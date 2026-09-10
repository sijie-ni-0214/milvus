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

#include <folly/CancellationToken.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/ScopeGuard.h>
#include <folly/system/ThreadName.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/common_type_c.h"
#include "futures/Executor.h"
#include "futures/Future.h"
#include "futures/LeakyResult.h"
#include "futures/Ready.h"
#include "futures/future_c.h"
#include "futures/future_c_types.h"
#include "gtest/gtest.h"
#include "monitor/Monitor.h"
#include "storage/ThreadPools.h"
#include "storage/Util.h"

using namespace milvus::futures;

TEST(Futures, LeakyResult) {
    {
        LeakyResult<int> leaky_result;
        ASSERT_ANY_THROW(leaky_result.leakyGet());
    }

    {
        auto leaky_result = LeakyResult<int>(1, "error");
        auto [r, s] = leaky_result.leakyGet();
        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, 1);
        ASSERT_STREQ(s.error_msg, "error");
        free((char*)(s.error_msg));
    }
    {
        auto leaky_result = LeakyResult<int>(new int(1));
        auto [r, s] = leaky_result.leakyGet();
        ASSERT_NE(r, nullptr);
        ASSERT_EQ(*(int*)(r), 1);
        ASSERT_EQ(s.error_code, 0);
        ASSERT_EQ(s.error_msg, nullptr);
        delete (int*)(r);
    }
    {
        LeakyResult<int> leaky_result(1, "error");
        LeakyResult<int> leaky_result_moved(std::move(leaky_result));
        auto [r, s] = leaky_result_moved.leakyGet();
        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, 1);
        ASSERT_STREQ(s.error_msg, "error");
        free((char*)(s.error_msg));
    }
    {
        LeakyResult<int> leaky_result(1, "error");
        LeakyResult<int> leaky_result_moved;
        leaky_result_moved = std::move(leaky_result);
        auto [r, s] = leaky_result_moved.leakyGet();
        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, 1);
        ASSERT_STREQ(s.error_msg, "error");
        free((char*)(s.error_msg));
    }
}

TEST(Futures, Ready) {
    Ready<int> ready;
    int a = 0;
    ready.callOrRegisterCallback([&a]() { a++; });
    ASSERT_EQ(a, 0);
    ASSERT_FALSE(ready.isReady());
    ready.setValue(1);
    ASSERT_EQ(a, 1);
    ASSERT_TRUE(ready.isReady());
    ready.callOrRegisterCallback([&a]() { a++; });
    ASSERT_EQ(a, 2);

    ASSERT_EQ(std::move(ready).getValue(), 1);
}

TEST(Futures, Future) {
    folly::CPUThreadPoolExecutor executor(2);

    // success path.
    {
        // try a async function
        auto future = milvus::futures::Future<int>::async(
            &executor, 0, [](folly::CancellationToken token) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                return new int(1);
            });
        ASSERT_FALSE(future->isReady());

        std::mutex mu;
        mu.lock();
        future->registerReadyCallback(
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        ASSERT_TRUE(future->isReady());
        auto [r, s] = future->leakyGet();

        ASSERT_NE(r, nullptr);
        ASSERT_EQ(*(int*)(r), 1);
        ASSERT_EQ(s.error_code, 0);
        ASSERT_EQ(s.error_msg, nullptr);
        delete (int*)(r);
    }

    // error path.
    {
        // try a async function
        auto future = milvus::futures::Future<int>::async(
            &executor, 0, [](folly::CancellationToken token) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                throw milvus::SegcoreError(milvus::NotImplemented,
                                           "unimplemented");
                return new int(1);
            });
        ASSERT_FALSE(future->isReady());

        std::mutex mu;
        mu.lock();
        future->registerReadyCallback(
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        ASSERT_TRUE(future->isReady());
        auto [r, s] = future->leakyGet();

        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, milvus::NotImplemented);
        ASSERT_STREQ(s.error_msg, "unimplemented");
        free((char*)(s.error_msg));
    }

    {
        // try a async function
        auto future = milvus::futures::Future<int>::async(
            &executor, 0, [](folly::CancellationToken token) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                throw std::runtime_error("unimplemented");
                return new int(1);
            });
        ASSERT_FALSE(future->isReady());

        std::mutex mu;
        mu.lock();
        future->registerReadyCallback(
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        ASSERT_TRUE(future->isReady());
        auto [r, s] = future->leakyGet();

        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, milvus::UnexpectedError);
        free((char*)(s.error_msg));
    }

    {
        // try a async function
        auto future = milvus::futures::Future<int>::async(
            &executor, 0, [](folly::CancellationToken token) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                throw folly::FutureNotReady();
                return new int(1);
            });
        ASSERT_FALSE(future->isReady());

        std::mutex mu;
        mu.lock();
        future->registerReadyCallback(
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        ASSERT_TRUE(future->isReady());
        auto [r, s] = future->leakyGet();

        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, milvus::FollyOtherException);
        free((char*)(s.error_msg));
    }

    // cancellation path.
    {
        // try a async function
        auto future = milvus::futures::Future<int>::async(
            &executor, 0, [](folly::CancellationToken token) {
                for (int i = 0; i < 10; i++) {
                    milvus::futures::throwIfCancelled(token);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                return new int(1);
            });
        ASSERT_FALSE(future->isReady());
        future->cancel();

        std::mutex mu;
        mu.lock();
        future->registerReadyCallback(
            [](CLockedGoMutex* mutex) { ((std::mutex*)(mutex))->unlock(); },
            (CLockedGoMutex*)(&mu));
        mu.lock();
        ASSERT_TRUE(future->isReady());
        auto [r, s] = future->leakyGet();

        ASSERT_EQ(r, nullptr);
        ASSERT_EQ(s.error_code, milvus::FollyCancel);
        free((char*)(s.error_msg));
    }
}

TEST(Futures, SubmitReduceTaskUsesReduceExecutor) {
    auto future = SubmitReduceTask(folly::CancellationToken(), [] {
        return folly::getCurrentThreadName().value_or("");
    });

    auto thread_name = future.get();
    EXPECT_EQ(thread_name.rfind("MILVUS_REDUCE_", 0), 0);
    EXPECT_EQ(getReduceCPUExecutor()->getNumPriorities(), 1);
}

TEST(Futures, SubmitReduceTaskHonorsCancellationBeforeExecution) {
    folly::CancellationSource source;
    source.requestCancellation();
    std::atomic<bool> executed{false};

    auto future = SubmitReduceTask(source.getToken(),
                                   [&executed] { executed.store(true); });

    EXPECT_THROW(future.get(), folly::FutureCancellation);
    EXPECT_FALSE(executed.load());
}

TEST(Futures, SubmitReduceTaskPreservesSegcoreError) {
    auto future = SubmitReduceTask(folly::CancellationToken(), []() -> int {
        throw milvus::SegcoreError(milvus::NotImplemented,
                                   "reduce task failed");
    });

    try {
        static_cast<void>(future.get());
        FAIL() << "expected SegcoreError";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), milvus::NotImplemented);
        EXPECT_STREQ(error.what(), "reduce task failed");
    }
}

TEST(Futures, ResizeReduceExecutorClampsAndUpdatesMetric) {
    auto* executor = getReduceCPUExecutor();
    const auto original_size = executor->numThreads();
    auto restore = folly::makeGuard([original_size] {
        executor_set_reduce_thread_num(static_cast<int>(original_size));
    });

    executor_set_reduce_thread_num(0);
    EXPECT_EQ(executor->numThreads(), 1);
    EXPECT_EQ(milvus::monitor::internal_cgo_pool_size_reduce.Value(), 1);

    executor_set_reduce_thread_num(2);
    EXPECT_EQ(executor->numThreads(), 2);
    EXPECT_EQ(milvus::monitor::internal_cgo_pool_size_reduce.Value(), 2);
}

TEST(Futures, ReduceTasksCanWaitForSaturatedMiddlePool) {
    auto* reduce_executor = getReduceCPUExecutor();
    auto& middle_pool =
        milvus::ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    const auto original_reduce_size = reduce_executor->numThreads();
    const auto original_middle_size = middle_pool.GetMaxThreadNum();
    auto restore = folly::makeGuard([original_reduce_size,
                                     original_middle_size,
                                     &middle_pool] {
        executor_set_reduce_thread_num(static_cast<int>(original_reduce_size));
        middle_pool.Resize(original_middle_size);
    });

    executor_set_reduce_thread_num(1);
    middle_pool.Resize(1);
    std::atomic<int> completed_fields{0};
    std::vector<std::future<void>> segment_tasks;
    for (int segment = 0; segment < 2; ++segment) {
        segment_tasks.emplace_back(SubmitReduceTask(
            folly::CancellationToken(), [&middle_pool, &completed_fields] {
                std::vector<std::future<void>> field_tasks;
                for (int field = 0; field < 2; ++field) {
                    field_tasks.emplace_back(middle_pool.Submit(
                        [&completed_fields] { ++completed_fields; }));
                }
                milvus::storage::WaitAllFutures(field_tasks);
            }));
    }

    milvus::storage::WaitAllFutures(segment_tasks);
    EXPECT_EQ(completed_fields.load(), 4);
}
