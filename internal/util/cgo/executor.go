package cgo

/*
#cgo pkg-config: milvus_core

#include "futures/future_c.h"
#include "segcore/segcore_init_c.h"
*/
import "C"

import (
	"context"
	"math"

	"github.com/milvus-io/milvus/pkg/v3/config"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// initExecutor initialize underlying cgo thread pools.
func initExecutor() {
	pt := paramtable.Get()

	// Initialize search executor pool.
	searchPoolSize := int(math.Ceil(pt.QueryNodeCfg.MaxReadConcurrency.GetAsFloat() * pt.QueryNodeCfg.CGOPoolSizeRatio.GetAsFloat()))
	setSearchThreadNum(searchPoolSize)

	resetSearchThreadNum := func(evt *config.Event) {
		if evt.HasUpdated {
			pt := paramtable.Get()
			newSize := int(math.Ceil(pt.QueryNodeCfg.MaxReadConcurrency.GetAsFloat() * pt.QueryNodeCfg.CGOPoolSizeRatio.GetAsFloat()))
			mlog.Info(context.TODO(), "reset cgo search thread num", mlog.Int("thread_num", newSize))
			setSearchThreadNum(newSize)
		}
	}
	pt.Watch(pt.QueryNodeCfg.MaxReadConcurrency.Key, config.NewHandler("cgo.search."+pt.QueryNodeCfg.MaxReadConcurrency.Key, resetSearchThreadNum))
	pt.Watch(pt.QueryNodeCfg.CGOPoolSizeRatio.Key, config.NewHandler("cgo.search."+pt.QueryNodeCfg.CGOPoolSizeRatio.Key, resetSearchThreadNum))

	// Initialize reduce executor pool.
	setReduceThreadNum(calculateReducePoolSize(
		pt.QueryNodeCfg.MaxReadConcurrency.GetAsFloat(),
		pt.QueryNodeCfg.ReducePoolSizeRatio.GetAsFloat(),
	))

	resetReduceThreadNum := func(evt *config.Event) {
		if evt.HasUpdated {
			pt := paramtable.Get()
			newSize := calculateReducePoolSize(
				pt.QueryNodeCfg.MaxReadConcurrency.GetAsFloat(),
				pt.QueryNodeCfg.ReducePoolSizeRatio.GetAsFloat(),
			)
			mlog.Info(context.TODO(), "reset cgo reduce thread num", mlog.Int("thread_num", newSize))
			setReduceThreadNum(newSize)
		}
	}
	pt.Watch(pt.QueryNodeCfg.MaxReadConcurrency.Key, config.NewHandler("cgo.reduce."+pt.QueryNodeCfg.MaxReadConcurrency.Key, resetReduceThreadNum))
	pt.Watch(pt.QueryNodeCfg.ReducePoolSizeRatio.Key, config.NewHandler("cgo.reduce."+pt.QueryNodeCfg.ReducePoolSizeRatio.Key, resetReduceThreadNum))

	// Initialize load executor pool.
	loadPoolSize := hardware.GetCPUNum() * pt.CommonCfg.MiddlePriorityThreadCoreCoefficient.GetAsInt()
	C.executor_set_load_thread_num(C.int(loadPoolSize))

	resetLoadThreadNum := func(evt *config.Event) {
		if evt.HasUpdated {
			pt := paramtable.Get()
			newSize := hardware.GetCPUNum() * pt.CommonCfg.MiddlePriorityThreadCoreCoefficient.GetAsInt()
			mlog.Info(context.TODO(), "reset cgo load thread num", mlog.Int("thread_num", newSize))
			C.executor_set_load_thread_num(C.int(newSize))
		}
	}
	pt.Watch(pt.CommonCfg.MiddlePriorityThreadCoreCoefficient.Key, config.NewHandler("cgo.load."+pt.CommonCfg.MiddlePriorityThreadCoreCoefficient.Key, resetLoadThreadNum))
}

func setSearchThreadNum(size int) {
	if size <= 0 {
		size = 1
	}
	C.executor_set_search_thread_num(C.int(size))
	C.SegcoreSetPrefetchThreadPoolNum(C.uint32_t(size))
}

func calculateReducePoolSize(maxReadConcurrency, ratio float64) int {
	size := int(math.Ceil(maxReadConcurrency * ratio))
	if size < 1 {
		return 1
	}
	return size
}

func setReduceThreadNum(size int) {
	if size <= 0 {
		size = 1
	}
	C.executor_set_reduce_thread_num(C.int(size))
}
