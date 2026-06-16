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

#include "ThreadPool.h"

#include <chrono>
#include <algorithm>

#include "log/Log.h"
#include "storage/SafeQueue.h"

namespace milvus {

int CPU_NUM = DEFAULT_CPU_NUM;
std::atomic<int> THREAD_POOL_MAX_THREADS_SIZE(
    DEFAULT_THREAD_POOL_MAX_THREADS_SIZE);

std::atomic<float> HIGH_PRIORITY_THREAD_CORE_COEFFICIENT(
    DEFAULT_HIGH_PRIORITY_THREAD_CORE_COEFFICIENT);
std::atomic<float> MIDDLE_PRIORITY_THREAD_CORE_COEFFICIENT(
    DEFAULT_MIDDLE_PRIORITY_THREAD_CORE_COEFFICIENT);
std::atomic<float> LOW_PRIORITY_THREAD_CORE_COEFFICIENT(
    DEFAULT_LOW_PRIORITY_THREAD_CORE_COEFFICIENT);

void
SetHighPriorityThreadCoreCoefficient(const float coefficient) {
    HIGH_PRIORITY_THREAD_CORE_COEFFICIENT.store(coefficient);
    LOG_INFO("set high priority thread pool core coefficient: {}",
             HIGH_PRIORITY_THREAD_CORE_COEFFICIENT.load());
}

void
SetMiddlePriorityThreadCoreCoefficient(const float coefficient) {
    MIDDLE_PRIORITY_THREAD_CORE_COEFFICIENT.store(coefficient);
    LOG_INFO("set middle priority thread pool core coefficient: {}",
             MIDDLE_PRIORITY_THREAD_CORE_COEFFICIENT.load());
}

void
SetLowPriorityThreadCoreCoefficient(const float coefficient) {
    LOW_PRIORITY_THREAD_CORE_COEFFICIENT.store(coefficient);
    LOG_INFO("set low priority thread pool core coefficient: {}",
             LOW_PRIORITY_THREAD_CORE_COEFFICIENT.load());
}

void
InitCpuNum(const int num) {
    CPU_NUM = num;
}

void
SetThreadPoolMaxThreadsSize(const int size) {
    THREAD_POOL_MAX_THREADS_SIZE.store(size);
    LOG_INFO("set thread pool max threads size: {}", size);
}

namespace {

constexpr int64_t POOL_STATS_LOG_INTERVAL_NS = 5LL * 1000 * 1000 * 1000;
constexpr bool kSnRecoveryThreadPoolStatsEnabled = false;

void
UpdateMax(std::atomic<int64_t>& target, int64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

double
NsToMs(int64_t ns) {
    return static_cast<double>(ns) / 1000.0 / 1000.0;
}

}  // namespace

int64_t
ThreadPool::NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void
ThreadPool::RecordTaskSubmitted(size_t queue_depth) {
    if (!kSnRecoveryThreadPoolStatsEnabled) {
        return;
    }
    submitted_tasks_delta_.fetch_add(1, std::memory_order_relaxed);
    UpdateMax(max_queue_depth_,
              static_cast<int64_t>(queue_depth));
}

void
ThreadPool::RecordTaskStarted(int64_t submit_time_ns) {
    if (!kSnRecoveryThreadPoolStatsEnabled) {
        return;
    }
    auto queue_wait_ns = std::max<int64_t>(0, NowNs() - submit_time_ns);
    queue_wait_samples_delta_.fetch_add(1, std::memory_order_relaxed);
    total_queue_wait_ns_delta_.fetch_add(queue_wait_ns,
                                         std::memory_order_relaxed);
    UpdateMax(max_queue_wait_ns_, queue_wait_ns);

    auto running =
        running_tasks_.fetch_add(1, std::memory_order_relaxed) + 1;
    UpdateMax(max_running_tasks_, running);
    LogStatsIfNeeded();
}

void
ThreadPool::RecordTaskFinished() {
    if (!kSnRecoveryThreadPoolStatsEnabled) {
        return;
    }
    running_tasks_.fetch_sub(1, std::memory_order_relaxed);
    completed_tasks_delta_.fetch_add(1, std::memory_order_relaxed);
    LogStatsIfNeeded();
}

void
ThreadPool::LogStatsIfNeeded() {
    if (!kSnRecoveryThreadPoolStatsEnabled) {
        return;
    }
    auto now_ns = NowNs();
    auto last_ns = last_pool_stats_log_ns_.load(std::memory_order_relaxed);
    if (last_ns == 0) {
        if (last_pool_stats_log_ns_.compare_exchange_strong(
                last_ns,
                now_ns,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
    if (now_ns - last_ns < POOL_STATS_LOG_INTERVAL_NS) {
        return;
    }
    if (!last_pool_stats_log_ns_.compare_exchange_strong(
            last_ns,
            now_ns,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
        return;
    }

    auto running = running_tasks_.load(std::memory_order_relaxed);
    auto submitted =
        submitted_tasks_delta_.exchange(0, std::memory_order_relaxed);
    auto completed =
        completed_tasks_delta_.exchange(0, std::memory_order_relaxed);
    auto samples =
        queue_wait_samples_delta_.exchange(0, std::memory_order_relaxed);
    auto total_queue_wait_ns =
        total_queue_wait_ns_delta_.exchange(0, std::memory_order_relaxed);
    auto max_queue_wait_ns =
        max_queue_wait_ns_.exchange(0, std::memory_order_relaxed);
    auto max_running =
        max_running_tasks_.exchange(running, std::memory_order_relaxed);

    int current_threads = 0;
    int idle_threads = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_threads = current_threads_size_;
        idle_threads = idle_threads_size_;
    }
    auto queue_depth = static_cast<int64_t>(work_queue_.size());
    auto max_queue =
        max_queue_depth_.exchange(queue_depth, std::memory_order_relaxed);
    max_running = std::max(max_running, running);
    max_queue = std::max(max_queue, queue_depth);

    auto avg_queue_wait_ns = samples == 0 ? 0 : total_queue_wait_ns / samples;
    LOG_WARN(
        "[SN recovery] pool stats phase=segcore.thread_pool pool={} "
        "capacity={} threads={} active={} idle={} queue={} submittedDelta={} "
        "completedDelta={} maxActive={} maxQueue={} avgQueueWaitMs={:.3f} "
        "maxQueueWaitMs={:.3f}",
        name_,
        max_threads_size_.load(std::memory_order_relaxed),
        current_threads,
        running,
        idle_threads,
        queue_depth,
        submitted,
        completed,
        max_running,
        max_queue,
        NsToMs(avg_queue_wait_ns),
        NsToMs(max_queue_wait_ns));
}

void
ThreadPool::Init() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < min_threads_size_; i++) {
        std::thread t(&ThreadPool::Worker, this);
        assert(threads_.find(t.get_id()) == threads_.end());
        threads_[t.get_id()] = std::move(t);
        current_threads_size_++;
    }
}

void
ThreadPool::ShutDown() {
    LOG_INFO("Start shutting down {}", name_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    condition_lock_.notify_all();
    for (auto& thread : threads_) {
        if (thread.second.joinable()) {
            thread.second.join();
        }
    }
    LOG_INFO("Finish shutting down {}", name_);
}

void
ThreadPool::FinishThreads() {
    while (!need_finish_threads_.empty()) {
        std::thread::id id;
        auto dequeue = need_finish_threads_.dequeue(id);
        if (dequeue) {
            auto iter = threads_.find(id);
            assert(iter != threads_.end());
            if (iter->second.joinable()) {
                iter->second.join();
            }
            threads_.erase(iter);
        }
    }
}

void
ThreadPool::Worker() {
    std::function<void()> func;
    bool dequeue;
    SetThreadName(name_);
    while (!shutdown_) {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_threads_size_++;
        if (metric_idle_) {
            metric_idle_->Set(idle_threads_size_);
        }
        if (metric_active_) {
            metric_active_->Set(current_threads_size_ - idle_threads_size_);
        }
        auto is_timeout = !condition_lock_.wait_for(
            lock, std::chrono::seconds(WAIT_SECONDS), [this]() {
                return shutdown_ || !work_queue_.empty();
            });
        idle_threads_size_--;
        if (metric_idle_) {
            metric_idle_->Set(idle_threads_size_);
        }
        if (metric_active_) {
            metric_active_->Set(current_threads_size_ - idle_threads_size_);
        }
        if (work_queue_.empty()) {
            // Dynamic reduce thread number
            if (shutdown_) {
                current_threads_size_--;
                if (metric_active_) {
                    metric_active_->Set(current_threads_size_ -
                                        idle_threads_size_);
                }
                return;
            }
            if (is_timeout) {
                FinishThreads();
                if (current_threads_size_ > min_threads_size_) {
                    need_finish_threads_.enqueue(std::this_thread::get_id());
                    current_threads_size_--;
                    if (metric_active_) {
                        metric_active_->Set(current_threads_size_ -
                                            idle_threads_size_);
                    }
                    return;
                }
                continue;
            }
        }
        dequeue = work_queue_.dequeue(func);
        if (metric_queue_depth_) {
            metric_queue_depth_->Set(work_queue_.size());
        }
        lock.unlock();
        if (dequeue) {
            func();
            func = nullptr;
            if (metric_completed_) {
                metric_completed_->Increment();
            }
        }
    }
}
};  // namespace milvus
