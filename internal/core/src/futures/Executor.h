// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>

#include <folly/CancellationToken.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/task_queue/PriorityLifoSemMPMCQueue.h>
#include <folly/system/HardwareConcurrency.h>

#include "Future.h"

namespace milvus::futures {

namespace ExecutePriority {
const int LOW = 2;
const int NORMAL = 1;
const int HIGH = 0;
}  // namespace ExecutePriority

folly::CPUThreadPoolExecutor*
getGlobalCPUExecutor();

folly::CPUThreadPoolExecutor*
getSearchCPUExecutor();

folly::CPUThreadPoolExecutor*
getLoadCPUExecutor();

folly::CPUThreadPoolExecutor*
getReduceCPUExecutor();

template <typename F>
auto
SubmitReduceTask(const folly::CancellationToken& token, F&& task)
    -> std::future<std::invoke_result_t<std::decay_t<F>&>> {
    using Task = std::decay_t<F>;
    using Result = std::invoke_result_t<Task&>;
    using TaskMetrics = Metrics<std::chrono::microseconds>;

    auto metrics = std::make_shared<TaskMetrics>(PoolType::kReduce);
    auto packaged_task = std::make_shared<std::packaged_task<Result()>>(
        [token,
         task = Task(std::forward<F>(task)),
         metrics = std::move(metrics)]() mutable -> Result {
            if (token.isCancellationRequested()) {
                metrics->withEarlyCancel();
                throw folly::FutureCancellation();
            }

            typename TaskMetrics::ExecutionGuard execution_guard(*metrics);
            try {
                return std::invoke(task);
            } catch (const folly::FutureCancellation&) {
                metrics->withDuringCancel();
                throw;
            } catch (const milvus::SegcoreError& e) {
                if (e.get_error_code() == milvus::ErrorCode::FollyCancel) {
                    metrics->withDuringCancel();
                }
                throw;
            }
        });
    auto future = packaged_task->get_future();
    getReduceCPUExecutor()->add(
        [packaged_task = std::move(packaged_task)]() mutable {
            (*packaged_task)();
        });
    return future;
}

};  // namespace milvus::futures
