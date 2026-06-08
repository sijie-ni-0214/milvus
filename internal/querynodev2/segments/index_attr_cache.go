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

package segments

/*
#cgo pkg-config: milvus_core

#include "segcore/load_index_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unsafe"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/indexparams"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var indexAttrCache = NewIndexAttrCache()

// getIndexAttrCache use a singleton to store index meta cache.
func getIndexAttrCache() *IndexAttrCache {
	return indexAttrCache
}

// IndexAttrCache index meta cache stores calculated attribute.
type IndexAttrCache struct {
	loadWithDisk  *typeutil.ConcurrentMap[typeutil.Pair[string, int32], bool]
	indexResource *typeutil.ConcurrentMap[string, ResourceEstimate]
	sf            conc.Singleflight[bool]
	resourceSF    conc.Singleflight[ResourceEstimate]
}

func NewIndexAttrCache() *IndexAttrCache {
	return &IndexAttrCache{
		loadWithDisk:  typeutil.NewConcurrentMap[typeutil.Pair[string, int32], bool](),
		indexResource: typeutil.NewConcurrentMap[string, ResourceEstimate](),
	}
}

func (c *IndexAttrCache) GetIndexResourceUsage(indexInfo *querypb.FieldIndexInfo, memoryIndexLoadPredictMemoryUsageFactor float64, fieldBinlog *datapb.FieldBinlog) (memory uint64, disk uint64, err error) {
	indexType, err := funcutil.GetAttrByKeyFromRepeatedKV(common.IndexTypeKey, indexInfo.IndexParams)
	if err != nil {
		return 0, 0, errors.New("index type not exist in index params")
	}
	if vecindexmgr.GetVecIndexMgrInstance().IsDiskANN(indexType) {
		neededMemSize := indexInfo.IndexSize / UsedDiskMemoryRatio
		neededDiskSize := indexInfo.IndexSize - neededMemSize
		return uint64(neededMemSize), uint64(neededDiskSize), nil
	}
	if vecindexmgr.GetVecIndexMgrInstance().IsAISAQ(indexType) {
		neededMemSize := indexInfo.IndexSize / UsedDiskMemoryRatioAisaq
		neededDiskSize := indexInfo.IndexSize
		return uint64(neededMemSize), uint64(neededDiskSize), nil
	}
	if indexType == indexparamcheck.IndexINVERTED {
		neededMemSize := 0
		// we will mmap the binlog if the index type is inverted index.
		neededDiskSize := indexInfo.IndexSize + getBinlogDataDiskSize(fieldBinlog)
		return uint64(neededMemSize), uint64(neededDiskSize), nil
	}

	engineVersion := indexInfo.GetCurrentIndexVersion()
	isLoadWithDisk, has := c.loadWithDisk.Get(typeutil.NewPair(indexType, engineVersion))
	if !has {
		isLoadWithDisk, _, _ = c.sf.Do(fmt.Sprintf("%s_%d", indexType, engineVersion), func() (bool, error) {
			var result bool
			GetDynamicPool().Submit(func() (any, error) {
				cIndexType := C.CString(indexType)
				defer C.free(unsafe.Pointer(cIndexType))
				cEngineVersion := C.int32_t(indexInfo.GetCurrentIndexVersion())
				result = bool(C.IsLoadWithDisk(cIndexType, cEngineVersion))
				return nil, nil
			}).Await()
			c.loadWithDisk.Insert(typeutil.NewPair(indexType, engineVersion), result)
			return result, nil
		})
	}

	factor := float64(1)
	diskUsage := uint64(0)
	if !isLoadWithDisk {
		factor = memoryIndexLoadPredictMemoryUsageFactor
	} else {
		diskUsage = uint64(indexInfo.IndexSize)
	}

	return uint64(float64(indexInfo.IndexSize) * factor), diskUsage, nil
}

func (c *IndexAttrCache) GetCIndexResourceEstimate(ctx context.Context, fieldSchema *schemapb.FieldSchema, loadInfo *querypb.SegmentLoadInfo, indexInfo *querypb.FieldIndexInfo) (ResourceEstimate, error) {
	key, err := indexResourceEstimateCacheKey(fieldSchema, indexInfo)
	if err != nil {
		return ResourceEstimate{}, err
	}
	if estimate, ok := c.indexResource.Get(key); ok {
		indexEstimateTiming.recordCgoCacheHit()
		return estimate, nil
	}

	estimate, err, _ := c.resourceSF.Do(key, func() (ResourceEstimate, error) {
		if estimate, ok := c.indexResource.Get(key); ok {
			indexEstimateTiming.recordCgoCacheHit()
			return estimate, nil
		}

		indexEstimateTiming.recordCgoCacheMiss()
		var estimateResult ResourceEstimate
		err := GetCLoadInfoWithFunc(ctx, fieldSchema, loadInfo, indexInfo, func(c *LoadIndexInfo) error {
			GetDynamicPool().Submit(func() (any, error) {
				loadResourceRequest := C.EstimateLoadIndexResource(c.cLoadIndexInfo)
				estimateResult = GetResourceEstimate(&loadResourceRequest)
				return nil, nil
			}).Await()
			return nil
		})
		if err != nil {
			return ResourceEstimate{}, err
		}

		c.indexResource.Insert(key, estimateResult)
		return estimateResult, nil
	})
	return estimate, err
}

func indexResourceEstimateCacheKey(fieldSchema *schemapb.FieldSchema, indexInfo *querypb.FieldIndexInfo) (string, error) {
	indexParams, err := prepareIndexLoadParamsForEstimate(fieldSchema, indexInfo)
	if err != nil {
		return "", err
	}

	dim := int64(1)
	if typeutil.IsVectorType(fieldSchema.GetDataType()) &&
		fieldSchema.GetDataType() != schemapb.DataType_SparseFloatVector {
		dim, err = typeutil.GetDim(fieldSchema)
		if err != nil {
			return "", err
		}
	}

	keys := make([]string, 0, len(indexParams))
	for key := range indexParams {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var builder strings.Builder
	_, _ = fmt.Fprintf(&builder, "%d|%d|%d|%d|%d|%d|%t",
		fieldSchema.GetDataType(),
		fieldSchema.GetElementType(),
		indexInfo.GetCurrentIndexVersion(),
		indexInfo.GetIndexSize(),
		indexInfo.GetNumRows(),
		dim,
		isIndexMmapEnable(fieldSchema, indexInfo),
	)
	for _, key := range keys {
		builder.WriteByte('|')
		builder.WriteString(key)
		builder.WriteByte('=')
		builder.WriteString(indexParams[key])
	}
	return builder.String(), nil
}

func prepareIndexLoadParamsForEstimate(fieldSchema *schemapb.FieldSchema, indexInfo *querypb.FieldIndexInfo) (map[string]string, error) {
	indexParams := funcutil.KeyValuePair2Map(indexInfo.GetIndexParams())
	delete(indexParams, common.MmapEnabledKey)

	indexType := indexParams[common.IndexTypeKey]
	if vecindexmgr.GetVecIndexMgrInstance().IsDiskANN(indexType) {
		if err := indexparams.SetDiskIndexLoadParams(paramtable.Get(), indexParams, indexInfo.GetNumRows()); err != nil {
			return nil, err
		}
	}

	if indexType == indexparamcheck.IndexBitmap {
		indexparams.SetBitmapIndexLoadParams(paramtable.Get(), indexParams)
	}

	if err := indexparams.AppendPrepareLoadParams(paramtable.Get(), indexParams); err != nil {
		return nil, err
	}

	if _, exists := indexParams[common.WarmupKey]; !exists {
		if warmupPolicy := getIndexWarmupPolicy(fieldSchema, indexInfo); warmupPolicy != "" {
			indexParams[common.WarmupKey] = warmupPolicy
		}
	}
	return indexParams, nil
}
