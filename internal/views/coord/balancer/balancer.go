package balancer

import (
	"context"
	"sync"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/internal/views/coord/coordview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
)

// Balancer is the scheduling controller that reconciles dirty shards into
// QueryView prepare/release operations.
type Balancer interface {
	Start(ctx context.Context)
	Stop()
	Trigger(scopes ...TriggerScope)
}

type snapshotSource interface {
	build(ctx context.Context, pending triggerBatch) (*BalancerSnapshot, []qviews.ShardID)
}

// DefaultBalancer owns the trigger queue and reconcile loop. Business
// decisions are delegated to BalancePolicy; this type only builds snapshots,
// drains dirty work, and applies the resulting BalancePlan.
type DefaultBalancer struct {
	snapshotBuilder snapshotSource
	viewRegistry    *coordview.ShardViewRegistry
	policy          BalancePolicy
	queue           *triggerQueue
	tickerInterval  time.Duration

	mu     sync.Mutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewDefaultBalancer constructs the standard Balancer controller.
func NewDefaultBalancer(
	builder *SnapshotBuilder,
	registry *coordview.ShardViewRegistry,
	policy BalancePolicy,
) *DefaultBalancer {
	if policy == nil {
		policy = NewDefaultBalancePolicy()
	}
	var interval time.Duration
	if builder != nil && builder.config != nil {
		interval = builder.config.TickerInterval
	}
	var source snapshotSource
	if builder != nil {
		source = builder
	}
	balancer := &DefaultBalancer{
		snapshotBuilder: source,
		viewRegistry:    registry,
		policy:          policy,
		queue:           newTriggerQueue(),
		tickerInterval:  interval,
	}
	if builder != nil {
		balancer.registerNodeChangedNotifier(builder.nodeProvider)
	}
	return balancer
}

func (b *DefaultBalancer) registerNodeChangedNotifier(provider NodeProvider) {
	notifier, ok := provider.(NodeChangedNotifier)
	if !ok {
		return
	}
	notifier.RegisterNodeChangedNotifier(func() {
		b.Trigger(TriggerScope{NodeChanged: true})
	})
}

// Start launches the reconcile loop and enqueues an initial full scan.
func (b *DefaultBalancer) Start(ctx context.Context) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cancel != nil {
		return
	}
	loopCtx, cancel := context.WithCancel(ctx)
	b.cancel = cancel
	b.wg.Add(1)
	go b.loop(loopCtx)
	b.queue.add()
}

// Stop cancels the reconcile loop and waits for it to exit.
func (b *DefaultBalancer) Stop() {
	b.mu.Lock()
	cancel := b.cancel
	b.cancel = nil
	b.mu.Unlock()

	if cancel != nil {
		cancel()
		b.wg.Wait()
	}
}

// Trigger enqueues affected shards. Calling Trigger with no scopes enqueues a
// full scan.
func (b *DefaultBalancer) Trigger(scopes ...TriggerScope) {
	b.queue.add(scopes...)
}

func (b *DefaultBalancer) loop(ctx context.Context) {
	defer b.wg.Done()

	var ticker *time.Ticker
	var tickerCh <-chan time.Time
	if b.tickerInterval > 0 {
		ticker = time.NewTicker(b.tickerInterval)
		tickerCh = ticker.C
		defer ticker.Stop()
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-b.queue.signalCh():
		case <-tickerCh:
			b.queue.add()
			continue
		}

		_ = b.Reconcile(ctx)
	}
}

// Reconcile runs one reconcile cycle. It is exported primarily for tests and
// for callers that want a synchronous controller pass during startup.
func (b *DefaultBalancer) Reconcile(ctx context.Context) error {
	if b.snapshotBuilder == nil || b.viewRegistry == nil || b.policy == nil {
		return nil
	}
	totalStartedAt := time.Now()
	// Take this cycle's work before building the snapshot so triggers arriving
	// during snapshot construction remain queued for the next cycle.
	triggerStartedAt := time.Now()
	pending := b.queue.takePending()
	triggerDuration := time.Since(triggerStartedAt)
	if pending.empty() {
		return nil
	}
	snapshotStartedAt := time.Now()
	snap, dirty := b.snapshotBuilder.build(ctx, pending)
	snapshotDuration := time.Since(snapshotStartedAt)
	if len(dirty) == 0 {
		logReconcileStats(
			ctx,
			pending,
			0,
			nil,
			snapshotDuration,
			triggerDuration,
			0,
			0,
			time.Since(totalStartedAt),
			nil,
		)
		return nil
	}
	planStartedAt := time.Now()
	plan := b.policy.Plan(snap, dirty)
	planDuration := time.Since(planStartedAt)
	applyStartedAt := time.Now()
	err := b.apply(ctx, plan)
	applyDuration := time.Since(applyStartedAt)
	logReconcileStats(
		ctx,
		pending,
		len(dirty),
		plan,
		snapshotDuration,
		triggerDuration,
		planDuration,
		applyDuration,
		time.Since(totalStartedAt),
		err,
	)
	return err
}

func logReconcileStats(
	ctx context.Context,
	batch triggerBatch,
	reconciledShardCount int,
	plan *BalancePlan,
	snapshotDuration time.Duration,
	triggerDuration time.Duration,
	planDuration time.Duration,
	applyDuration time.Duration,
	totalDuration time.Duration,
	err error,
) {
	prepareCount := 0
	releaseCount := 0
	if plan != nil {
		prepareCount = len(plan.Prepares)
		releaseCount = len(plan.Releases)
	}
	fields := []mlog.Field{
		mlog.String("phase", "qc.recovery_reconcile"),
		mlog.Bool("fullScan", batch.full),
		mlog.Int("dirtyNodes", len(batch.dirtyNodes)),
		mlog.Int("dirtyShards", len(batch.dirtyShards)),
		mlog.Int("dirtyCollections", len(batch.dirtyColls)),
		mlog.Int("reconciledShards", reconciledShardCount),
		mlog.Int("prepareCount", prepareCount),
		mlog.Int("releaseCount", releaseCount),
		mlog.Duration("snapshotDuration", snapshotDuration),
		mlog.Duration("triggerDrainDuration", triggerDuration),
		mlog.Duration("planDuration", planDuration),
		mlog.Duration("applyDuration", applyDuration),
		mlog.Duration("totalDuration", totalDuration),
	}
	if err != nil {
		mlog.Warn(ctx, "[SN recovery] query view reconcile completed with errors", append(fields, mlog.Err(err))...)
		return
	}
	mlog.Info(ctx, "[SN recovery] query view reconcile completed", fields...)
}

func (b *DefaultBalancer) apply(ctx context.Context, plan *BalancePlan) error {
	if plan == nil {
		return nil
	}
	batch := b.viewRegistry.Begin()
	defer batch.Commit()
	var errs []error
	for _, shardID := range plan.Releases {
		mgr := b.viewRegistry.Get(shardID)
		if mgr == nil {
			continue
		}
		if err := mgr.RequestRelease(ctx); err != nil {
			errs = append(errs, err)
		}
	}
	for shardID, builder := range plan.Prepares {
		if builder == nil {
			continue
		}
		mgr := b.viewRegistry.Ensure(shardID)
		if err := mgr.AddPreparing(ctx, builder); err != nil {
			errs = append(errs, err)
		}
	}
	var err error
	for _, e := range errs {
		err = errors.CombineErrors(err, e)
	}
	return err
}
