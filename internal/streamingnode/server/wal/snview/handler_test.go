package snview

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/internal/views/worknode/handler"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// buildPersistKey constructs a unique persistence key from view metadata.
// Used only by mockCatalog to simulate catalog behavior.
func buildPersistKey(meta *viewpb.QueryViewMeta) string {
	return fmt.Sprintf("%d/%s/%d/%d/%d",
		meta.ReplicaId,
		meta.Vchannel,
		meta.Version.DataVersion.StreamingVersion,
		meta.Version.DataVersion.CompactVersion,
		meta.Version.QueryVersion,
	)
}

// ---------------------------------------------------------------------------
// Mock catalog
// ---------------------------------------------------------------------------

type mockCatalog struct {
	metastore.StreamingNodeCataLog
	mu    sync.Mutex
	saved map[string]*viewpb.QueryViewOfShard
}

func newMockCatalog() *mockCatalog {
	return &mockCatalog{
		saved: make(map[string]*viewpb.QueryViewOfShard),
	}
}

func (c *mockCatalog) SaveQueryViews(_ context.Context, _ string, views []*viewpb.QueryViewOfShard) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, view := range views {
		key := buildPersistKey(view.Meta)
		persistState := qviews.QueryViewState(view.Meta.State)
		switch persistState {
		case qviews.QueryViewStateUp:
			c.saved[key] = view
		default:
			delete(c.saved, key)
		}
	}
	return nil
}

func (c *mockCatalog) ListQueryViews(context.Context, string) ([]*viewpb.QueryViewOfShard, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	views := make([]*viewpb.QueryViewOfShard, 0, len(c.saved))
	for _, view := range c.saved {
		views = append(views, view)
	}
	return views, nil
}

func (c *mockCatalog) savedCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.saved)
}

// ---------------------------------------------------------------------------
// Mock ResourceManager
// ---------------------------------------------------------------------------

type mockResourceManager struct {
	runtime         QueryRuntime
	mu              sync.Mutex
	acquired        map[qviews.QueryViewKey]AcquireResource
	acquiredOrder   []qviews.QueryViewKey
	released        []qviews.QueryViewKey
	releaseCallback map[qviews.QueryViewKey]func() // captured OnDropped callbacks
}

func newMockResourceManager() *mockResourceManager {
	return &mockResourceManager{
		acquired:        make(map[qviews.QueryViewKey]AcquireResource),
		releaseCallback: make(map[qviews.QueryViewKey]func()),
	}
}

func (m *mockResourceManager) Acquire(req AcquireResource) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acquired[req.Key] = req
	m.acquiredOrder = append(m.acquiredOrder, req.Key)
}

func (m *mockResourceManager) Release(req ReleaseResource) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.released = append(m.released, req.Key)
	m.releaseCallback[req.Key] = req.OnDropped
}

func (m *mockResourceManager) invokeReleaseCallback(key qviews.QueryViewKey) {
	m.mu.Lock()
	cb := m.releaseCallback[key]
	delete(m.releaseCallback, key)
	m.mu.Unlock()
	if cb != nil {
		cb()
	}
}

func (m *mockResourceManager) getAcquired(key qviews.QueryViewKey) (AcquireResource, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.acquired[key]
	return req, ok
}

func (m *mockResourceManager) acquiredCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.acquired)
}

func (m *mockResourceManager) acquiredKeys() []qviews.QueryViewKey {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]qviews.QueryViewKey{}, m.acquiredOrder...)
}

func (m *mockResourceManager) releasedCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.released)
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

func buildHandlerTestMeta(version int64) *viewpb.QueryViewMeta {
	return &viewpb.QueryViewMeta{
		CollectionId: testCollectionID,
		ReplicaId:    testReplicaID,
		Vchannel:     testVChannel,
		Version: &viewpb.QueryViewVersion{
			DataVersion:  &viewpb.DataVersion{StreamingVersion: version, CompactVersion: 1},
			QueryVersion: version,
		},
		State: viewpb.QueryViewState_QueryViewStatePreparing,
	}
}

func newPreparingSNView(version int64) qviews.QueryViewAtWorkNode {
	return qviews.NewQueryViewAtStreamingNode(buildHandlerTestMeta(version), &viewpb.QueryViewOfStreamingNode{})
}

func newSNViewWithState(version int64, state viewpb.QueryViewState) qviews.QueryViewAtWorkNode {
	meta := buildHandlerTestMeta(version)
	meta.State = state
	return qviews.NewQueryViewAtStreamingNode(meta, &viewpb.QueryViewOfStreamingNode{})
}

func newFullSNViewWithState(version int64, state viewpb.QueryViewState, qnNodeIDs ...int64) qviews.QueryViewAtWorkNode {
	meta := buildHandlerTestMeta(version)
	meta.State = state
	queryNodes := make([]*viewpb.QueryViewOfQueryNode, 0, len(qnNodeIDs))
	for _, nodeID := range qnNodeIDs {
		queryNodes = append(queryNodes, &viewpb.QueryViewOfQueryNode{
			NodeId: nodeID,
			Partitions: []*viewpb.QueryViewOfPartition{
				{PartitionId: 10, SegmentIds: []int64{nodeID * 100}},
			},
		})
	}
	return qviews.NewFullQueryViewAtStreamingNode(meta, &viewpb.QueryViewOfStreamingNode{}, queryNodes)
}

type reportCollector struct {
	mu      sync.Mutex
	reports []qviews.QueryViewAtWorkNode
}

func (c *reportCollector) onReport(report qviews.QueryViewAtWorkNode) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reports = append(c.reports, report)
}

func (c *reportCollector) get() []qviews.QueryViewAtWorkNode {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]qviews.QueryViewAtWorkNode{}, c.reports...)
}

func (c *reportCollector) last() qviews.QueryViewAtWorkNode {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.reports) == 0 {
		return nil
	}
	return c.reports[len(c.reports)-1]
}

func (c *reportCollector) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.reports)
}

// ---------------------------------------------------------------------------
// 1. ApplyViews — new Preparing view triggers Acquire
// ---------------------------------------------------------------------------

func TestSNHandler_ApplyViews_NewPreparing(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})

	// SN SM generates Preparing report on construction.
	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStatePreparing, rc.last().State())
	// No persistence for Preparing.
	assert.Equal(t, 0, cat.savedCount())
	// ResourceManager should have been called with Acquire.
	key := view.QueryViewKey()
	_, ok := mgr.getAcquired(key)
	require.True(t, ok)
}

func TestSNHandler_AcquireUnrecoverableReportsUnrecoverable(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{{View: view, OnReport: rc.onReport}})
	req, ok := mgr.getAcquired(view.QueryViewKey())
	require.True(t, ok)
	require.NotNil(t, req.OnUnrecoverable)

	req.OnUnrecoverable()

	require.Equal(t, 2, rc.count())
	assert.Equal(t, qviews.QueryViewStateUnrecoverable, rc.last().State())
	assert.Equal(t, 0, cat.savedCount())
}

func TestSNHandler_ApplyViews_UnknownViewReportsUnrecoverable(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	// Non-Preparing, non-Dropped state for unknown view → Unrecoverable.
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateReady), OnReport: rc.onReport},
	})

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateUnrecoverable, rc.last().State())
	assert.Equal(t, 0, mgr.acquiredCount())
}

func TestSNHandler_ApplyViews_UnknownDownReportsDropped(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDown), OnReport: rc.onReport},
	})

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())
	assert.Equal(t, 0, mgr.acquiredCount())
}

func TestSNHandler_ApplyViews_DroppedOnUnknownViewReportsBack(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	// Coord pushes Dropped for a view SN doesn't know (e.g., after restart).
	// SN must report Dropped back so Coord can finish cleanup.
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())
	assert.Equal(t, 0, mgr.acquiredCount())
}

// ---------------------------------------------------------------------------
// 2. ResourceManager callback → Ready
// ---------------------------------------------------------------------------

func TestSNHandler_ResourceManagerCallback_TransitionToReady(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})

	key := view.QueryViewKey()
	req, _ := mgr.getAcquired(key)

	// ResourceManager calls OnReady.
	req.OnReady()

	require.Equal(t, 2, rc.count()) // Preparing + Ready
	assert.Equal(t, qviews.QueryViewStateReady, rc.last().State())
	// No persistence for Ready.
	assert.Equal(t, 0, cat.savedCount())
}

// ---------------------------------------------------------------------------
// 3. Coord Up — Ready → Up (persists)
// ---------------------------------------------------------------------------

func TestSNHandler_CoordUp_PersistsRecoveryInfo(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})

	key := view.QueryViewKey()
	req, _ := mgr.getAcquired(key)
	req.OnReady()

	// Coord pushes Up.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp), OnReport: rc.onReport},
	})

	require.Equal(t, 3, rc.count()) // Preparing + Ready + Up
	assert.Equal(t, qviews.QueryViewStateUp, rc.last().State())
	assert.Equal(t, 1, cat.savedCount())
}

func TestSNHandler_PersistCancellationDoesNotReport(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	ctx, cancel := context.WithCancel(context.Background())
	h := recoverSNQueryViewHandler(ctx, testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{{View: view, OnReport: rc.onReport}})
	req, ok := mgr.getAcquired(view.QueryViewKey())
	require.True(t, ok)
	req.OnReady()
	require.Equal(t, 2, rc.count())

	mockSave := mockey.Mock((*mockCatalog).SaveQueryViews).
		To(func(_ *mockCatalog, ctx context.Context, _ string, _ []*viewpb.QueryViewOfShard) error {
			return ctx.Err()
		}).Build()
	t.Cleanup(func() { mockSave.UnPatch() })
	cancel()
	require.NotPanics(t, func() {
		h.ApplyViews([]handler.ApplyView{{
			View:     newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp),
			OnReport: rc.onReport,
		}})
	})
	assert.Equal(t, 2, rc.count(), "unpersisted Up must not be reported")
}

func TestSNHandler_PersistFailureIsTerminal(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{{View: view, OnReport: rc.onReport}})
	req, ok := mgr.getAcquired(view.QueryViewKey())
	require.True(t, ok)
	req.OnReady()
	require.Equal(t, 2, rc.count())

	mockSave := mockey.Mock((*mockCatalog).SaveQueryViews).
		Return(merr.WrapErrServiceInternalMsg("injected persist failure")).Build()
	t.Cleanup(func() { mockSave.UnPatch() })
	require.Panics(t, func() {
		h.ApplyViews([]handler.ApplyView{{
			View:     newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp),
			OnReport: rc.onReport,
		}})
	})
	assert.Equal(t, 2, rc.count(), "unpersisted Up must not be reported")
}

func TestSNHandler_ApplyRetriesAfterShardDetached(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)
	first := newPreparingSNView(1)
	firstKey := first.QueryViewKey()
	h.ApplyViews([]handler.ApplyView{{View: first}})
	h.ApplyViews([]handler.ApplyView{{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped)}})
	require.Equal(t, 1, mgr.releasedCount())

	resolved := make(chan struct{})
	resume := make(chan struct{})
	var blockOnce sync.Once
	var origin func(*SNQueryViewHandler, qviews.ShardID) *snShardView
	mock := mockey.Mock((*SNQueryViewHandler).getOrCreateShard).
		To(func(handler *SNQueryViewHandler, shardID qviews.ShardID) *snShardView {
			shard := origin(handler, shardID)
			blockOnce.Do(func() {
				close(resolved)
				<-resume
			})
			return shard
		}).Origin(&origin).Build()
	t.Cleanup(func() { mock.UnPatch() })

	second := newPreparingSNView(2)
	applyDone := make(chan struct{})
	go func() {
		h.ApplyViews([]handler.ApplyView{{View: second}})
		close(applyDone)
	}()
	<-resolved

	mgr.invokeReleaseCallback(firstKey)
	close(resume)
	<-applyDone

	h.ApplyViews([]handler.ApplyView{{View: newSNViewWithState(2, viewpb.QueryViewState_QueryViewStateDropped)}})
	assert.Equal(t, 2, mgr.releasedCount(), "replacement view must remain reachable for release")
}

func TestSNHandler_RecoveryPublishesShardBeforeAcquire(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	meta := buildHandlerTestMeta(1)
	meta.State = viewpb.QueryViewState_QueryViewStateUp
	persistedView := &viewpb.QueryViewOfShard{
		Meta:          meta,
		StreamingNode: &viewpb.QueryViewOfStreamingNode{},
	}
	key := newPreparingSNView(1).QueryViewKey()

	var origin func(
		context.Context,
		string,
		qviews.ShardID,
		map[qviews.QueryViewVersion]*snQueryViewStateMachine,
		metastore.StreamingNodeCataLog,
		StreamingNodeResourceManager,
	) *snShardView
	mock := mockey.Mock(recoverSnShardView).To(func(
		ctx context.Context,
		pchannel string,
		shardID qviews.ShardID,
		views map[qviews.QueryViewVersion]*snQueryViewStateMachine,
		catalog metastore.StreamingNodeCataLog,
		resMgr StreamingNodeResourceManager,
	) *snShardView {
		shard := origin(ctx, pchannel, shardID, views, catalog, resMgr)
		if _, acquired := mgr.getAcquired(key); acquired {
			shard.ApplyViews([]handler.ApplyView{{
				View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped),
			}})
			mgr.invokeReleaseCallback(key)
		}
		return shard
	}).Origin(&origin).Build()
	t.Cleanup(func() { mock.UnPatch() })

	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{persistedView})
	h.mu.Lock()
	shard := h.shards[key.ShardID]
	h.mu.Unlock()
	require.NotNil(t, shard)
	shard.mu.Lock()
	viewCount := len(shard.views)
	shard.mu.Unlock()
	assert.Equal(t, 1, viewCount, "recovery callback must not empty a shard before it is published")
	assert.Equal(t, 1, mgr.acquiredCount())
}

func TestSNHandler_CloseForHandoffWaitsForExistingRelease(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)
	view := newPreparingSNView(1)
	key := view.QueryViewKey()
	h.ApplyViews([]handler.ApplyView{{View: view}})
	h.ApplyViews([]handler.ApplyView{{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped)}})
	require.Equal(t, 1, mgr.releasedCount())

	handoffDone := make(chan struct{})
	go func() {
		h.CloseForHandoff()
		close(handoffDone)
	}()

	assert.Never(t, func() bool {
		return mgr.releasedCount() > 1
	}, 100*time.Millisecond, 5*time.Millisecond, "handoff must reuse the in-flight release")
	mgr.invokeReleaseCallback(key)
	require.Eventually(t, func() bool {
		select {
		case <-handoffDone:
			return true
		default:
			return false
		}
	}, time.Second, 5*time.Millisecond)
	assert.Equal(t, 1, mgr.releasedCount())
}

// ---------------------------------------------------------------------------
// 4. Coord Down — Up → Down (deletes recovery info)
// ---------------------------------------------------------------------------

func TestSNHandler_CoordDown_DeletesRecoveryInfo(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})

	key := view.QueryViewKey()
	req, _ := mgr.getAcquired(key)
	req.OnReady()

	// Coord Up.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp), OnReport: rc.onReport},
	})
	assert.Equal(t, 1, cat.savedCount())

	// Coord Down.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDown), OnReport: rc.onReport},
	})
	assert.Equal(t, qviews.QueryViewStateDown, rc.last().State())
	assert.Equal(t, 0, cat.savedCount()) // deleted
}

// ---------------------------------------------------------------------------
// 5. Coord Dropped — Dropping → Release → Dropped
// ---------------------------------------------------------------------------

func TestSNHandler_CoordDropped_DroppingThenDropped(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	key := view.QueryViewKey()
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})
	assert.Equal(t, 1, rc.count()) // Preparing

	// Push Dropped → SM enters Dropping, Release called.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})

	// No report yet — SM is in Dropping, waiting for Release callback.
	assert.Equal(t, 1, rc.count()) // still just Preparing
	assert.Equal(t, 1, mgr.releasedCount())

	// ResourceManager completes release → SM transitions Dropping → Dropped.
	mgr.invokeReleaseCallback(key)

	require.Equal(t, 2, rc.count())
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())

	// Entry removed: stale OnReady callback is a no-op.
	req, _ := mgr.getAcquired(key)
	req.OnReady()
	assert.Equal(t, 2, rc.count())
}

func TestSNHandler_CoordDropped_WhileDropping_Ignored(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	key := view.QueryViewKey()
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})

	// Push Dropped → SM enters Dropping.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})
	assert.Equal(t, 1, mgr.releasedCount())

	// Coord re-pushes Dropped while SM is Dropping → no additional Release.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})
	assert.Equal(t, 1, mgr.releasedCount()) // no double Release

	// Release callback completes → Dropped report.
	mgr.invokeReleaseCallback(key)
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())
}

// ---------------------------------------------------------------------------
// 7. Recover — crash recovery
// ---------------------------------------------------------------------------

func TestSNHandler_Recover_CreatesUpRecoveringViews(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()

	meta := buildHandlerTestMeta(1)
	meta.State = viewpb.QueryViewState_QueryViewStateUp
	persistedView := &viewpb.QueryViewOfShard{
		Meta:          meta,
		StreamingNode: &viewpb.QueryViewOfStreamingNode{},
	}

	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{persistedView})

	// ResourceManager should have Acquire called for recovered Up views.
	key := newPreparingSNView(1).QueryViewKey()
	acquireReq, ok := mgr.getAcquired(key)
	require.True(t, ok)

	// Register callback via ApplyViews (simulating Coord re-push).
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newPreparingSNView(1), OnReport: rc.onReport},
	})

	// UpRecovering: Coord re-push Preparing → no report (SM suppresses).
	assert.Equal(t, 0, rc.count())

	// WAL catches up via ResourceManager callback.
	acquireReq.OnReady()

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateUp, rc.last().State())
	// Already persisted as Up — no new save (catalog save count unchanged).
}

func TestSNHandler_Recover_AcquiresUpViewsInVersionOrder(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()

	meta2 := buildHandlerTestMeta(2)
	meta2.State = viewpb.QueryViewState_QueryViewStateUp
	meta1 := buildHandlerTestMeta(1)
	meta1.State = viewpb.QueryViewState_QueryViewStateUp

	recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{
		{Meta: meta2, StreamingNode: &viewpb.QueryViewOfStreamingNode{}},
		{Meta: meta1, StreamingNode: &viewpb.QueryViewOfStreamingNode{}},
	})

	keys := mgr.acquiredKeys()
	require.Len(t, keys, 2)
	assert.True(t, keys[1].QueryViewVersion.GT(keys[0].QueryViewVersion))
}

// ---------------------------------------------------------------------------
// 8. Callback replacement on re-apply
// ---------------------------------------------------------------------------

func TestSNHandler_CallbackReplacement(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc1 := &reportCollector{}
	view := newPreparingSNView(1)
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc1.onReport},
	})
	assert.Equal(t, 1, rc1.count()) // Preparing

	// Make Ready via callback.
	key := view.QueryViewKey()
	req, _ := mgr.getAcquired(key)
	req.OnReady()
	assert.Equal(t, 2, rc1.count()) // Preparing + Ready

	// Re-apply with new callback.
	rc2 := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newPreparingSNView(1), OnReport: rc2.onReport},
	})

	// rc1 unchanged, rc2 gets re-report.
	assert.Equal(t, 2, rc1.count())
	require.Equal(t, 1, rc2.count())
	assert.Equal(t, qviews.QueryViewStateReady, rc2.last().State())
}

// ---------------------------------------------------------------------------
// 9. Full lifecycle
// ---------------------------------------------------------------------------

func TestSNHandler_FullLifecycle(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	key := view.QueryViewKey()

	// 1. Apply Preparing → report Preparing, Acquire called.
	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})
	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStatePreparing, rc.last().State())

	// 2. ResourceManager OnReady → Ready.
	req, _ := mgr.getAcquired(key)
	req.OnReady()
	require.Equal(t, 2, rc.count())
	assert.Equal(t, qviews.QueryViewStateReady, rc.last().State())

	// 3. Coord Up → persist.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp), OnReport: rc.onReport},
	})
	require.Equal(t, 3, rc.count())
	assert.Equal(t, qviews.QueryViewStateUp, rc.last().State())
	assert.Equal(t, 1, cat.savedCount())

	// 4. Coord Down → delete persist.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDown), OnReport: rc.onReport},
	})
	require.Equal(t, 4, rc.count())
	assert.Equal(t, qviews.QueryViewStateDown, rc.last().State())
	assert.Equal(t, 0, cat.savedCount())

	// 5. Coord Dropped → Dropping, Release called.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})
	assert.Equal(t, 4, rc.count()) // no report yet (Dropping)
	assert.Equal(t, 1, mgr.releasedCount())

	// 6. Release callback → Dropped.
	mgr.invokeReleaseCallback(key)
	require.Equal(t, 5, rc.count())
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())

	// 7. Further callbacks are no-op.
	req.OnReady()
	assert.Equal(t, 5, rc.count())
}

// ---------------------------------------------------------------------------
// 10. Multiple versions in same shard
// ---------------------------------------------------------------------------

func TestSNHandler_MultipleVersions(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc1 := &reportCollector{}
	rc2 := &reportCollector{}
	view1 := newPreparingSNView(1)
	view2 := newPreparingSNView(2)

	h.ApplyViews([]handler.ApplyView{
		{View: view1, OnReport: rc1.onReport},
		{View: view2, OnReport: rc2.onReport},
	})

	// Both get Preparing report.
	assert.Equal(t, 1, rc1.count())
	assert.Equal(t, 1, rc2.count())

	// Only notify version 1 ready via callback.
	key1 := view1.QueryViewKey()
	req1, _ := mgr.getAcquired(key1)
	req1.OnReady()

	assert.Equal(t, 2, rc1.count())
	assert.Equal(t, qviews.QueryViewStateReady, rc1.last().State())
	assert.Equal(t, 1, rc2.count()) // version 2 unaffected
}

// ---------------------------------------------------------------------------
// 11. Concurrency safety
// ---------------------------------------------------------------------------

func TestSNHandler_ConcurrentApplyAndCallback(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	const numViews = 20
	var wg sync.WaitGroup
	var readyCount atomic.Int32

	for i := int64(1); i <= numViews; i++ {
		view := newPreparingSNView(i)
		h.ApplyViews([]handler.ApplyView{
			{View: view, OnReport: func(report qviews.QueryViewAtWorkNode) {
				if report.State() == qviews.QueryViewStateReady {
					readyCount.Add(1)
				}
			}},
		})
	}

	// Invoke all callbacks concurrently.
	for i := int64(1); i <= numViews; i++ {
		wg.Add(1)
		go func(version int64) {
			defer wg.Done()
			view := newPreparingSNView(version)
			key := view.QueryViewKey()
			req, ok := mgr.getAcquired(key)
			if ok {
				req.OnReady()
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int32(numViews), readyCount.Load())
}

// ---------------------------------------------------------------------------
// 13. Recover with callback via ApplyViews re-push, then Coord Down
// ---------------------------------------------------------------------------

func TestSNHandler_Recover_ThenCoordDown(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()

	meta := buildHandlerTestMeta(1)
	meta.State = viewpb.QueryViewState_QueryViewStateUp
	persistedView := &viewpb.QueryViewOfShard{
		Meta:          meta,
		StreamingNode: &viewpb.QueryViewOfStreamingNode{},
	}
	cat.SaveQueryViews(context.Background(), testPChannel, []*viewpb.QueryViewOfShard{persistedView})

	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{persistedView})

	// Coord pushes Down to recovered view.
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDown), OnReport: rc.onReport},
	})

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateDown, rc.last().State())
	// Recovery info deleted.
	assert.Equal(t, 0, cat.savedCount())
}

// ---------------------------------------------------------------------------
// 14. Recover callback on unknown shard/version ignored
// ---------------------------------------------------------------------------

func TestSNHandler_RecoverCallback_AfterDropped_Ignored(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()

	// Recover a view, then drop it, then invoke the stale Recover callback.
	meta := buildHandlerTestMeta(1)
	meta.State = viewpb.QueryViewState_QueryViewStateUp
	persistedView := &viewpb.QueryViewOfShard{
		Meta:          meta,
		StreamingNode: &viewpb.QueryViewOfStreamingNode{},
	}

	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{persistedView})

	key := newPreparingSNView(1).QueryViewKey()
	acquireReq, ok := mgr.getAcquired(key)
	require.True(t, ok)

	// Drop the view via Coord push.
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})
	mgr.invokeReleaseCallback(key)
	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())

	// Stale recovered Acquire callback after entry removed should be no-op.
	acquireReq.OnReady()
	assert.Equal(t, 1, rc.count())
}

// ---------------------------------------------------------------------------
// 15. Dropped from Up — deletes recovery info via Dropping
// ---------------------------------------------------------------------------

func TestSNHandler_DroppedFromUp_DeletesRecoveryInfo(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()
	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, nil)

	rc := &reportCollector{}
	view := newPreparingSNView(1)
	key := view.QueryViewKey()

	h.ApplyViews([]handler.ApplyView{
		{View: view, OnReport: rc.onReport},
	})
	req, _ := mgr.getAcquired(key)
	req.OnReady()

	// Up.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp), OnReport: rc.onReport},
	})
	assert.Equal(t, 1, cat.savedCount())

	// Dropped from Up → Dropping (persist deleted immediately), Release called.
	h.ApplyViews([]handler.ApplyView{
		{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDropped), OnReport: rc.onReport},
	})
	assert.Equal(t, 0, cat.savedCount()) // recovery info deleted immediately
	assert.Equal(t, 1, mgr.releasedCount())

	// Release callback → Dropped.
	mgr.invokeReleaseCallback(key)
	assert.Equal(t, qviews.QueryViewStateDropped, rc.last().State())
}

// ---------------------------------------------------------------------------
// 16. Recover async callback flow
// ---------------------------------------------------------------------------

func TestSNHandler_Recover_AcquireCallbackFlow(t *testing.T) {
	cat := newMockCatalog()
	mgr := newMockResourceManager()

	meta := buildHandlerTestMeta(1)
	meta.State = viewpb.QueryViewState_QueryViewStateUp
	persistedView := &viewpb.QueryViewOfShard{
		Meta:          meta,
		StreamingNode: &viewpb.QueryViewOfStreamingNode{},
	}

	h := recoverSNQueryViewHandler(context.Background(), testPChannel, cat, mgr, []*viewpb.QueryViewOfShard{persistedView})

	// Recovered views acquire resources through the same ordered Acquire path.
	assert.Equal(t, 1, mgr.acquiredCount())

	key := newPreparingSNView(1).QueryViewKey()
	acquireReq, ok := mgr.getAcquired(key)
	require.True(t, ok)
	assert.NotNil(t, acquireReq.OnReady)

	// Register callback via ApplyViews.
	rc := &reportCollector{}
	h.ApplyViews([]handler.ApplyView{
		{View: newPreparingSNView(1), OnReport: rc.onReport},
	})
	assert.Equal(t, 0, rc.count()) // UpRecovering suppresses

	// WAL catch-up completes.
	acquireReq.OnReady()

	require.Equal(t, 1, rc.count())
	assert.Equal(t, qviews.QueryViewStateUp, rc.last().State())
}

func (m *mockResourceManager) QueryRuntime(qviews.QueryViewKey) (QueryRuntime, bool) {
	return m.runtime, m.runtime != nil
}

func TestSNHandler_UnknownViewsLeaveNoShard(t *testing.T) {
	for _, state := range []qviews.QueryViewState{
		qviews.QueryViewStateUp, qviews.QueryViewStateReady,
		qviews.QueryViewStateDown, qviews.QueryViewStateDropped,
	} {
		t.Run(state.String(), func(t *testing.T) {
			// Unknown views must not acquire, release, or persist resources.
			h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, nil, nil)
			view := newSNViewWithState(1, viewpb.QueryViewState(state))
			before := view.IntoProto()
			expected := qviews.QueryViewStateUnrecoverable
			if state == qviews.QueryViewStateDown || state == qviews.QueryViewStateDropped {
				expected = qviews.QueryViewStateDropped
			}
			for range 3 {
				rc := &reportCollector{}
				h.ApplyViews([]handler.ApplyView{{View: view, OnReport: rc.onReport}})
				require.Equal(t, 1, rc.count())
				require.Equal(t, expected, rc.last().State())
				require.Equal(t, view.QueryViewKey(), rc.last().QueryViewKey())
				require.True(t, proto.Equal(before, view.IntoProto()))
				require.Empty(t, h.shards)
				require.Empty(t, h.shardsByVChannel)
			}
		})
	}
}

func TestSNHandler_AcquireLatestUpViewSkipsUnavailableReplicas(t *testing.T) {
	mockey.PatchConvey("replica lookup skips shards without an Up view", t, func() {
		acquires := make(map[qviews.QueryViewKey]AcquireResource)
		mockey.Mock((*mockResourceManager).Acquire).To(func(_ *mockResourceManager, req AcquireResource) {
			acquires[req.Key] = req
		}).Build()
		up := newFullSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp, 101, 102)
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, newMockResourceManager(), []*viewpb.QueryViewOfShard{up.IntoProto()})
		acquires[up.QueryViewKey()].OnReady()
		unknown := qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: testVChannel}
		// Include an empty shard (the old bug), plus valid non-serving states.
		for i, state := range []qviews.QueryViewState{
			qviews.QueryViewStateNil, qviews.QueryViewStatePreparing, qviews.QueryViewStateReady,
			qviews.QueryViewStateUpRecovering, qviews.QueryViewStateDown,
			qviews.QueryViewStateDropping, qviews.QueryViewStateUnrecoverable,
		} {
			id := qviews.ShardID{ReplicaID: testReplicaID + int64(i) + 1, VChannel: testVChannel}
			shard := h.getOrCreateShard(id)
			if state != qviews.QueryViewStateNil {
				pb := up.IntoProto()
				pb.Meta.ReplicaId = id.ReplicaID
				view := qviews.NewQueryViewAtWorkNodeFromProto(pb)
				sm := recoverSNQueryViewStateMachine(pb.Meta, pb.StreamingNode, pb.QueryNode)
				sm.state = state
				shard.views[view.Version()] = &snViewEntry{ApplyView: handler.ApplyView{View: view}, sm: sm}
			}
			// Explicit replica requests must never fall back to another replica.
			// UpRecovering waits instead; covered by the recovery wait tests.
			if state != qviews.QueryViewStateUpRecovering {
				_, err := h.AcquireLatestUpView(context.Background(), id)
				require.True(t, viewerror.AsViewError(err).IsViewNotFound())
			}
		}
		for range 128 {
			lease, err := h.AcquireLatestUpView(context.Background(), unknown)
			require.NoError(t, err)
			require.Equal(t, testReplicaID, lease.Meta.GetReplicaId())
			require.True(t, proto.Equal(up.IntoProto(), lease.View))
			lease.Release()
			lease.Release()
		}
		require.Zero(t, h.shards[up.ShardID()].views[up.Version()].queryRefs)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := h.AcquireLatestUpView(ctx, unknown)
		require.ErrorIs(t, err, context.Canceled)
		// If no replica is Up or recovering, preserve the retryable error code.
		for _, shard := range h.shards {
			for _, entry := range shard.views {
				entry.sm.state = qviews.QueryViewStateDown
			}
		}
		_, err = h.AcquireLatestUpView(context.Background(), unknown)
		require.True(t, viewerror.AsViewError(err).IsViewNotFound())
	})
}

func TestSNHandler_UnknownOldReplicaDoesNotRemoveServingReplica(t *testing.T) {
	mockey.PatchConvey("late old-replica reports preserve the serving replica", t, func() {
		var acquire AcquireResource
		mockey.Mock((*mockResourceManager).Acquire).To(func(_ *mockResourceManager, req AcquireResource) {
			acquire = req
		}).Build()
		up := newFullSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp, 101)
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, newMockResourceManager(), []*viewpb.QueryViewOfShard{up.IntoProto()})
		acquire.OnReady()
		for _, state := range []qviews.QueryViewState{qviews.QueryViewStateUp, qviews.QueryViewStateDown, qviews.QueryViewStateDropped} {
			old := up.IntoProto()
			old.Meta.ReplicaId++
			old.Meta.State = viewpb.QueryViewState(state)
			rc := &reportCollector{}
			h.ApplyViews([]handler.ApplyView{{View: qviews.NewQueryViewAtWorkNodeFromProto(old), OnReport: rc.onReport}})
			require.Equal(t, 1, rc.count())
			require.Len(t, h.shards, 1)
			require.Equal(t, map[qviews.ShardID]struct{}{up.ShardID(): {}}, h.shardsByVChannel[testVChannel])
			lease, err := h.AcquireLatestUpView(context.Background(), qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: testVChannel})
			require.NoError(t, err)
			require.True(t, proto.Equal(up.IntoProto(), lease.View))
			lease.Release()
		}
	})
}

func TestSNHandler_AcquireLatestUpViewRetriesCandidateThatGoesDown(t *testing.T) {
	mockey.PatchConvey("the first replica can go Down after the index snapshot", t, func() {
		mockey.Mock((*mockResourceManager).Acquire).Return().Build()
		first := newFullSNViewWithState(1, viewpb.QueryViewState_QueryViewStateUp, 101).IntoProto()
		second := proto.Clone(first).(*viewpb.QueryViewOfShard)
		second.Meta.ReplicaId++
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, newMockResourceManager(), []*viewpb.QueryViewOfShard{first, second})
		for _, shard := range h.shards {
			for version := range shard.views {
				shard.notifyRecoveringDone(version)
			}
		}
		var original func(*snShardView, context.Context, *int32) (*QueryViewLease, <-chan struct{}, error)
		var unavailable *snShardView
		attempts := 0
		mockey.Mock((*snShardView).tryAcquireUpViewForSchema).Origin(&original).To(func(shard *snShardView, ctx context.Context, schema *int32) (*QueryViewLease, <-chan struct{}, error) {
			attempts++
			if attempts == 1 {
				unavailable = shard
				shard.mu.Lock()
				for _, entry := range shard.views {
					entry.sm.OnCoordStateDelivered(qviews.QueryViewStateDown)
				}
				shard.mu.Unlock()
			}
			return original(shard, ctx, schema)
		}).Build()
		lease, err := h.AcquireLatestUpView(context.Background(), qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: testVChannel})
		require.NoError(t, err)
		defer lease.Release()
		require.Equal(t, 2, attempts)
		require.NotEqual(t, unavailable.shardID.ReplicaID, lease.Meta.GetReplicaId())
	})
}

func TestSNHandler_AcquireLatestUpViewCanceledDuringReplicaLookup(t *testing.T) {
	mockey.PatchConvey("cancellation during replica lookup is returned unchanged", t, func() {
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, nil, nil)
		h.getOrCreateShard(qviews.ShardID{ReplicaID: testReplicaID, VChannel: testVChannel})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var original func(*snShardView, context.Context, *int32) (*QueryViewLease, <-chan struct{}, error)
		mockey.Mock((*snShardView).tryAcquireUpViewForSchema).Origin(&original).To(func(shard *snShardView, ctx context.Context, schema *int32) (*QueryViewLease, <-chan struct{}, error) {
			cancel()
			return original(shard, ctx, schema)
		}).Build()
		_, err := h.AcquireLatestUpView(ctx, qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: testVChannel})
		require.ErrorIs(t, err, context.Canceled)
	})
}

func TestSNHandler_ApplyRetriesDetachedShard(t *testing.T) {
	mockey.PatchConvey("cleanup between lookup and apply cannot orphan a Preparing view", t, func() {
		var acquire AcquireResource
		acquireMock := mockey.Mock((*mockResourceManager).Acquire).To(func(_ *mockResourceManager, req AcquireResource) {
			acquire = req
		}).Build()
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, newMockResourceManager(), nil)
		view := newPreparingSNView(1)
		old := h.getOrCreateShard(view.ShardID())
		// Force cleanup after the handler has looked up old but before it
		// acquires the shard lock, independent of goroutine scheduling.
		var original func(*snShardView, []handler.ApplyView) bool
		first := true
		mockey.Mock((*snShardView).ApplyViews).Origin(&original).To(func(shard *snShardView, views []handler.ApplyView) bool {
			if first {
				first = false
				require.Same(t, old, shard)
				require.True(t, original(old, []handler.ApplyView{{View: newSNViewWithState(2, viewpb.QueryViewState_QueryViewStateUp)}}))
			}
			return original(shard, views)
		}).Build()
		rc := &reportCollector{}
		h.ApplyViews([]handler.ApplyView{{View: view, OnReport: rc.onReport}})
		replacement := h.shards[view.ShardID()]
		require.NotNil(t, replacement)
		require.NotSame(t, old, replacement)
		require.True(t, old.detached)
		require.Empty(t, old.views)
		require.EqualValues(t, 1, acquireMock.Times())
		acquire.OnReady()
		require.Equal(t, qviews.QueryViewStateReady, rc.last().State())
		// A delayed callback from the old instance must not erase the new one.
		old.onEmpty(old)
		require.Same(t, replacement, h.shards[view.ShardID()])
		require.Contains(t, h.shardsByVChannel[testVChannel], view.ShardID())
	})
}

func TestSNHandler_UnknownDownInMixedBatchKeepsPreparingView(t *testing.T) {
	mockey.PatchConvey("unknown teardown and Preparing coexist in one shard batch", t, func() {
		var acquire AcquireResource
		mockey.Mock((*mockResourceManager).Acquire).To(func(_ *mockResourceManager, req AcquireResource) {
			acquire = req
		}).Build()
		h := recoverSNQueryViewHandler(context.Background(), testPChannel, nil, newMockResourceManager(), nil)
		preparing := newPreparingSNView(2)
		downReports, preparingReports := &reportCollector{}, &reportCollector{}
		h.ApplyViews([]handler.ApplyView{
			{View: newSNViewWithState(1, viewpb.QueryViewState_QueryViewStateDown), OnReport: downReports.onReport},
			{View: preparing, OnReport: preparingReports.onReport},
		})
		require.Equal(t, qviews.QueryViewStateDropped, downReports.last().State())
		require.Len(t, h.shards, 1)
		shard := h.shards[preparing.ShardID()]
		require.False(t, shard.detached)
		require.Len(t, shard.views, 1)
		acquire.OnReady()
		require.Equal(t, qviews.QueryViewStateReady, preparingReports.last().State())
	})
}
