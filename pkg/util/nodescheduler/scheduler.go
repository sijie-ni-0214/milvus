// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nodescheduler

import (
	"container/list"
	"context"
	"math"
	"sync"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/pkg/v3/config"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type Scheduler interface {
	// Submit enqueues the task into an unbounded queue and returns without
	// waiting for queue capacity or task execution. Tasks may call Submit from
	// Execute, so implementations must preserve this non-blocking contract to
	// avoid exhausting all workers on nested submissions.
	Submit(Task) TaskHandle
}

type Task interface {
	Execute(context.Context) error
}

type TaskHandle interface {
	Cancel()
	Wait(context.Context) error
}

type scheduleErrorKind int

const scheduleErrorKindDelay scheduleErrorKind = iota + 1

type ScheduleError struct {
	kind scheduleErrorKind
}

func (e *ScheduleError) Error() string {
	if e != nil && e.kind == scheduleErrorKindDelay {
		return "delay node scheduler task"
	}
	return "unknown node scheduler error"
}

func (e *ScheduleError) Is(target error) bool {
	other, ok := target.(*ScheduleError)
	return ok && e != nil && other != nil && e.kind == other.kind
}

var ErrDelay = &ScheduleError{kind: scheduleErrorKindDelay}

type nodeScheduler struct {
	ctx    context.Context
	cancel context.CancelFunc
	stats  *schedulerStats

	mu     sync.Mutex
	cond   *sync.Cond
	queue  *list.List
	closed bool

	concurrency int
	workerCount int
	workers     sync.WaitGroup
	reporter    sync.WaitGroup
}

type taskEntry struct {
	task     Task
	taskType string
	queuedAt time.Time
	ctx      context.Context
	cancel   context.CancelFunc
	done     chan struct{}
	once     sync.Once
	wakeup   func()
}

func (e *taskEntry) finish() {
	e.once.Do(func() {
		e.cancel()
		close(e.done)
	})
}

type taskHandle struct {
	entry *taskEntry
}

func (h *taskHandle) Cancel() {
	h.entry.cancel()
	h.entry.wakeup()
}

func (h *taskHandle) Wait(ctx context.Context) error {
	select {
	case <-h.entry.done:
		return nil
	default:
	}

	select {
	case <-h.entry.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func New(concurrency int) *nodeScheduler {
	if concurrency <= 0 {
		panic("node scheduler concurrency must be greater than zero")
	}

	ctx, cancel := context.WithCancel(context.Background())
	scheduler := &nodeScheduler{
		ctx:    ctx,
		cancel: cancel,
		stats:  newSchedulerStats(concurrency),
		queue:  list.New(),
	}
	scheduler.cond = sync.NewCond(&scheduler.mu)
	scheduler.resize(concurrency)
	scheduler.reporter.Add(1)
	go scheduler.reportStats()
	return scheduler
}

func (s *nodeScheduler) resize(concurrency int) {
	if concurrency <= 0 {
		panic("node scheduler concurrency must be greater than zero")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.concurrency == concurrency {
		return
	}

	s.concurrency = concurrency
	s.stats.setCapacity(concurrency)
	if concurrency > s.workerCount {
		additional := concurrency - s.workerCount
		s.workerCount += additional
		s.workers.Add(additional)
		for i := 0; i < additional; i++ {
			go s.runWorker()
		}
	}
	s.cond.Broadcast()
}

func (s *nodeScheduler) Submit(task Task) TaskHandle {
	ctx, cancel := context.WithCancel(s.ctx)
	entry := &taskEntry{
		task:     task,
		taskType: schedulerTaskType(task),
		queuedAt: time.Now(),
		ctx:      ctx,
		cancel:   cancel,
		done:     make(chan struct{}),
		wakeup:   s.wakeup,
	}
	handle := &taskHandle{entry: entry}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		entry.finish()
		return handle
	}
	// The queue is intentionally unbounded: Submit must never wait for capacity.
	s.queue.PushBack(entry)
	s.stats.submit(entry.taskType)
	s.cond.Signal()
	s.mu.Unlock()
	return handle
}

func (s *nodeScheduler) Close() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		s.workers.Wait()
		s.reporter.Wait()
		return
	}

	s.closed = true
	for element := s.queue.Front(); element != nil; element = element.Next() {
		entry := element.Value.(*taskEntry)
		s.stats.cancelQueued(entry.taskType)
		entry.finish()
	}
	s.queue.Init()
	s.cancel()
	s.cond.Broadcast()
	s.mu.Unlock()

	s.workers.Wait()
	s.reporter.Wait()
}

func (s *nodeScheduler) wakeup() {
	s.mu.Lock()
	s.cond.Broadcast()
	s.mu.Unlock()
}

func (s *nodeScheduler) runWorker() {
	defer s.workers.Done()
	for {
		entry := s.dequeue()
		if entry == nil {
			return
		}
		if entry.ctx.Err() != nil {
			s.stats.cancelStarted(entry.taskType)
			entry.finish()
			continue
		}

		startedAt := time.Now()
		err := entry.task.Execute(entry.ctx)
		executeDuration := time.Since(startedAt)
		if entry.ctx.Err() != nil {
			s.stats.finishCanceled(entry.taskType, executeDuration)
			entry.finish()
			continue
		}
		if errors.Is(err, ErrDelay) {
			if s.requeue(entry, executeDuration) {
				continue
			}
			s.stats.finishDelayed(entry.taskType, executeDuration, false)
			entry.finish()
			continue
		}
		if err != nil {
			s.stats.finishFailed(entry.taskType, executeDuration)
			mlog.Error(entry.ctx, "node scheduler task failed",
				mlog.String("taskType", entry.taskType),
				mlog.Err(err))
		} else {
			s.stats.finishCompleted(entry.taskType, executeDuration)
		}
		entry.finish()
	}
}

func (s *nodeScheduler) dequeue() *taskEntry {
	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		if s.closed || s.workerCount > s.concurrency {
			s.workerCount--
			return nil
		}
		if s.queue.Len() > 0 {
			element := s.queue.Front()
			s.queue.Remove(element)
			entry := element.Value.(*taskEntry)
			s.stats.start(entry.taskType, time.Since(entry.queuedAt))
			return entry
		}
		s.cond.Wait()
	}
}

func (s *nodeScheduler) requeue(entry *taskEntry, executeDuration time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || entry.ctx.Err() != nil {
		return false
	}
	entry.queuedAt = time.Now()
	s.queue.PushBack(entry)
	s.stats.finishDelayed(entry.taskType, executeDuration, true)
	s.cond.Signal()
	return true
}

var getGlobalScheduler = sync.OnceValue(func() *nodeScheduler {
	params := paramtable.Get()
	ratioParam := &params.CommonCfg.NodeSchedulerMaxConcurrencyRatio
	cpu := hardware.GetCPUNum()
	concurrency, ok := concurrencyFromRatio(cpu, ratioParam.GetAsFloat())
	if !ok {
		concurrency = cpu
		mlog.Warn(context.TODO(), "invalid node scheduler concurrency ratio, use default concurrency",
			mlog.String("value", ratioParam.GetValue()),
			mlog.Int("concurrency", concurrency))
	}

	scheduler := New(concurrency)
	params.Watch(ratioParam.Key, config.NewHandler("node-scheduler-concurrency", func(event *config.Event) {
		if !event.HasUpdated {
			return
		}
		ratio := ratioParam.GetAsFloat()
		concurrency, ok := concurrencyFromRatio(hardware.GetCPUNum(), ratio)
		if !ok {
			mlog.Warn(context.TODO(), "ignore invalid node scheduler concurrency ratio",
				mlog.String("value", ratioParam.GetValue()))
			return
		}
		scheduler.resize(concurrency)
		mlog.Info(context.TODO(), "node scheduler concurrency resized",
			mlog.Float64("ratio", ratio),
			mlog.Int("concurrency", concurrency))
	}))
	return scheduler
})

func concurrencyFromRatio(cpu int, ratio float64) (int, bool) {
	if cpu <= 0 || ratio <= 0 || math.IsNaN(ratio) || math.IsInf(ratio, 0) {
		return 0, false
	}
	return max(1, int(float64(cpu)*ratio)), true
}

func Get() Scheduler {
	return getGlobalScheduler()
}
