// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nodescheduler

import (
	"context"
	"reflect"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
)

// TaskTypeProvider lets wrapper tasks expose the stable task type that should
// be used by node scheduler metrics.
type TaskTypeProvider interface {
	SchedulerTaskType() string
}

// TaskTypeName returns the stable, low-cardinality type name used by node
// scheduler metrics.
func TaskTypeName(task Task) string {
	if provider, ok := task.(TaskTypeProvider); ok {
		if taskType := provider.SchedulerTaskType(); taskType != "" {
			return taskType
		}
	}
	if task == nil {
		return "<nil>"
	}
	return reflect.TypeOf(task).String()
}

type schedulerMetrics struct {
	nodeID string
}

func newSchedulerMetrics(nodeID string) schedulerMetrics {
	return schedulerMetrics{nodeID: nodeID}
}

func (m schedulerMetrics) setConcurrency(concurrency int) {
	metrics.NodeSchedulerConcurrency.WithLabelValues(m.nodeID).Set(float64(concurrency))
}

func (m schedulerMetrics) observeEnqueued(taskType string) {
	metrics.NodeSchedulerPendingTaskCount.WithLabelValues(m.nodeID, taskType).Inc()
}

func (m schedulerMetrics) observeDequeued(taskType string, queueDuration time.Duration) {
	metrics.NodeSchedulerPendingTaskCount.WithLabelValues(m.nodeID, taskType).Dec()
	metrics.NodeSchedulerTaskQueueDurationSeconds.WithLabelValues(m.nodeID, taskType).Observe(queueDuration.Seconds())
}

func (m schedulerMetrics) observeExecutionStarted(taskType string) {
	metrics.NodeSchedulerRunningTaskCount.WithLabelValues(m.nodeID, taskType).Inc()
}

func (m schedulerMetrics) observeExecutionFinished(taskType, status string, executionDuration time.Duration) {
	metrics.NodeSchedulerRunningTaskCount.WithLabelValues(m.nodeID, taskType).Dec()
	metrics.NodeSchedulerTaskExecutionDurationSeconds.WithLabelValues(m.nodeID, taskType).Observe(executionDuration.Seconds())
	metrics.NodeSchedulerTaskExecutionTotal.WithLabelValues(m.nodeID, taskType, status).Inc()
}

func taskExecutionStatus(ctxErr, taskErr error) string {
	if ctxErr != nil {
		return metrics.CancelLabel
	}
	if taskErr == nil {
		return metrics.SuccessLabel
	}
	switch {
	case errors.Is(taskErr, context.Canceled),
		errors.Is(taskErr, context.DeadlineExceeded):
		return metrics.CancelLabel
	case errors.Is(taskErr, ErrDelay):
		return metrics.RetryLabel
	default:
		return metrics.FailLabel
	}
}
