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

package job

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// JobScheduler schedules jobs,
// all jobs within the same collection will run sequentially
const (
	collectionQueueCap = 64
	waitQueueCap       = 512
)

type jobQueue chan Job

type Scheduler struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	processors *typeutil.ConcurrentSet[int64] // Collections of having processor
	queues     map[int64]jobQueue             // CollectionID -> Queue
	waitQueue  jobQueue

	stopOnce sync.Once
}

func NewScheduler(parent context.Context) *Scheduler {
	ctx, cancel := context.WithCancel(parent) //nolint:gosec // Stop calls cancel before waiting for the workers.
	return &Scheduler{
		ctx: ctx, cancel: cancel,
		processors: typeutil.NewConcurrentSet[int64](),
		queues:     make(map[int64]jobQueue),
		waitQueue:  make(jobQueue, waitQueueCap),
	}
}

func (scheduler *Scheduler) Context() context.Context { return scheduler.ctx }

func (scheduler *Scheduler) Start() {
	scheduler.wg.Add(1)
	go func() {
		defer scheduler.wg.Done()
		scheduler.schedule(scheduler.ctx)
	}()
}

func (scheduler *Scheduler) Stop() {
	scheduler.stopOnce.Do(func() {
		if scheduler.cancel != nil {
			scheduler.cancel()
		}
		scheduler.wg.Wait()
	})
}

func (scheduler *Scheduler) schedule(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			mlog.Info(ctx, "JobManager stopped")
			for _, queue := range scheduler.queues {
				close(queue)
			}
			return

		case job := <-scheduler.waitQueue:
			queue, ok := scheduler.queues[job.CollectionID()]
			if !ok {
				queue = make(jobQueue, collectionQueueCap)
				scheduler.queues[job.CollectionID()] = queue
			}
			select {
			case queue <- job:
				scheduler.startProcessor(job.CollectionID(), queue)
			case <-ctx.Done():
				return
			}

		case <-ticker.C:
			for collection, queue := range scheduler.queues {
				if len(queue) > 0 {
					scheduler.startProcessor(collection, queue)
				} else {
					// Release resource if no job for the collection
					close(queue)
					delete(scheduler.queues, collection)
				}
			}
		}
	}
}

func (scheduler *Scheduler) Add(job Job) error {
	select {
	case <-scheduler.ctx.Done():
		return scheduler.ctx.Err()
	case scheduler.waitQueue <- job:
		return nil
	}
}

func (scheduler *Scheduler) startProcessor(collection int64, queue jobQueue) {
	if !scheduler.processors.Insert(collection) {
		return
	}

	scheduler.wg.Add(1)
	go scheduler.processQueue(collection, queue)
}

// processQueue processes jobs in the given queue,
// it only processes jobs with the number of the length of queue at the time,
// to avoid leaking goroutines
func (scheduler *Scheduler) processQueue(collection int64, queue jobQueue) {
	defer scheduler.wg.Done()
	defer scheduler.processors.Remove(collection)

	len := len(queue)
	for i := 0; i < len; i++ {
		scheduler.process(<-queue)
	}
}

func (scheduler *Scheduler) process(job Job) {
	log := mlog.With(
		mlog.FieldCollectionID(job.CollectionID()))

	defer func() {
		log.Info(context.TODO(), "start to post-execute job")
		job.PostExecute()
		log.Info(context.TODO(), "job finished")
		job.Done()
	}()

	log.Info(context.TODO(), "start to pre-execute job")
	err := job.PreExecute()
	if err != nil {
		log.Warn(context.TODO(), "failed to pre-execute job", mlog.Err(err))
		job.SetError(err)
		return
	}

	log.Info(context.TODO(), "start to execute job")
	err = job.Execute()
	if err != nil {
		log.Warn(context.TODO(), "failed to execute job", mlog.Err(err))
		job.SetError(err)
	}
}
