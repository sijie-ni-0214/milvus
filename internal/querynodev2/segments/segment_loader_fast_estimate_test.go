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

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestEstimateScalarIndexResourceFast(t *testing.T) {
	const indexSize = int64(4096)
	const expectedIndexSize = uint64(indexSize)
	scalarField := &schemapb.FieldSchema{DataType: schemapb.DataType_Int64}

	tests := []struct {
		name      string
		indexType string
		mmap      string
		expected  ResourceEstimate
	}{
		{
			name:      "stl sort",
			indexType: indexparamcheck.IndexSTLSORT,
			expected: ResourceEstimate{
				MaxMemoryCost:   2 * expectedIndexSize,
				FinalMemoryCost: expectedIndexSize,
				HasRawData:      true,
			},
		},
		{
			name:      "inverted",
			indexType: indexparamcheck.IndexINVERTED,
			expected: ResourceEstimate{
				MaxMemoryCost: expectedIndexSize,
				MaxDiskCost:   expectedIndexSize,
				FinalDiskCost: expectedIndexSize,
			},
		},
		{
			name:      "hybrid",
			indexType: indexparamcheck.IndexHybrid,
			expected: ResourceEstimate{
				MaxMemoryCost:   2 * expectedIndexSize,
				MaxDiskCost:     expectedIndexSize,
				FinalMemoryCost: expectedIndexSize,
				FinalDiskCost:   expectedIndexSize,
			},
		},
		{
			name:      "bitmap mmap",
			indexType: indexparamcheck.IndexBitmap,
			mmap:      "true",
			expected: ResourceEstimate{
				MaxMemoryCost: 2 * expectedIndexSize,
				MaxDiskCost:   2 * expectedIndexSize,
				FinalDiskCost: expectedIndexSize,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexInfo := &querypb.FieldIndexInfo{
				IndexSize: indexSize,
				IndexParams: []*commonpb.KeyValuePair{
					{Key: common.IndexTypeKey, Value: test.indexType},
				},
			}
			if test.mmap != "" {
				indexInfo.IndexParams = append(indexInfo.IndexParams, &commonpb.KeyValuePair{
					Key:   common.MmapEnabledKey,
					Value: test.mmap,
				})
			}

			actual, ok := estimateScalarIndexResourceFast(scalarField, indexInfo)
			require.True(t, ok)
			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestEstimateScalarIndexResourceFastSkipsVectorIndex(t *testing.T) {
	actual, ok := estimateScalarIndexResourceFast(
		&schemapb.FieldSchema{DataType: schemapb.DataType_FloatVector},
		&querypb.FieldIndexInfo{
			IndexSize: 1024,
			IndexParams: []*commonpb.KeyValuePair{
				{Key: common.IndexTypeKey, Value: "HNSW"},
			},
		},
	)
	require.False(t, ok)
	assert.Equal(t, ResourceEstimate{}, actual)
}

func TestEstimateKnowhereVectorIndexResourceFast(t *testing.T) {
	const indexSize = int64(4096)
	const expectedIndexSize = uint64(indexSize)
	vectorField := &schemapb.FieldSchema{DataType: schemapb.DataType_FloatVector}
	sparseField := &schemapb.FieldSchema{DataType: schemapb.DataType_SparseFloatVector}

	tests := []struct {
		name      string
		field     *schemapb.FieldSchema
		indexType string
		metric    string
		mmap      string
		expected  ResourceEstimate
	}{
		{
			name:      "hnsw mmap",
			field:     vectorField,
			indexType: "HNSW",
			mmap:      "true",
			expected: ResourceEstimate{
				MaxMemoryCost: expectedIndexSize,
				MaxDiskCost:   expectedIndexSize,
				FinalDiskCost: expectedIndexSize,
				HasRawData:    true,
			},
		},
		{
			name:      "sparse bm25 mmap",
			field:     sparseField,
			indexType: "SPARSE_INVERTED_INDEX",
			metric:    metric.BM25,
			mmap:      "true",
			expected: ResourceEstimate{
				MaxMemoryCost: expectedIndexSize,
				MaxDiskCost:   expectedIndexSize,
				FinalDiskCost: expectedIndexSize,
			},
		},
		{
			name:      "sparse ip in memory",
			field:     sparseField,
			indexType: "SPARSE_WAND",
			metric:    metric.IP,
			mmap:      "false",
			expected: ResourceEstimate{
				MaxMemoryCost:   2 * expectedIndexSize,
				FinalMemoryCost: expectedIndexSize,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexInfo := &querypb.FieldIndexInfo{
				IndexSize: indexSize,
				IndexParams: []*commonpb.KeyValuePair{
					{Key: common.IndexTypeKey, Value: test.indexType},
				},
			}
			if test.metric != "" {
				indexInfo.IndexParams = append(indexInfo.IndexParams, &commonpb.KeyValuePair{Key: common.MetricTypeKey, Value: test.metric})
			}
			if test.mmap != "" {
				indexInfo.IndexParams = append(indexInfo.IndexParams, &commonpb.KeyValuePair{Key: common.MmapEnabledKey, Value: test.mmap})
			}

			actual, ok := estimateIndexResourceFast(test.field, indexInfo)
			require.True(t, ok)
			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestEstimateIndexResourceFastMatchesCxxEstimate(t *testing.T) {
	paramtable.Init()
	require.NoError(t, initcore.InitRemoteChunkManager(paramtable.Get()))
	require.NoError(t, initcore.InitLocalChunkManager(t.TempDir()))

	const indexSize = int64(4096)
	tests := []struct {
		name      string
		field     *schemapb.FieldSchema
		indexType string
		metric    string
		mmap      string
	}{
		{
			name:      "stl sort",
			field:     fastEstimateTestField(100, schemapb.DataType_Int64),
			indexType: indexparamcheck.IndexSTLSORT,
		},
		{
			name:      "trie mmap",
			field:     fastEstimateTestField(101, schemapb.DataType_VarChar),
			indexType: indexparamcheck.IndexTRIE,
			mmap:      "true",
		},
		{
			name:      "legacy trie",
			field:     fastEstimateTestField(102, schemapb.DataType_VarChar),
			indexType: indexparamcheck.IndexTrie,
		},
		{
			name:      "inverted",
			field:     fastEstimateTestField(103, schemapb.DataType_VarChar),
			indexType: indexparamcheck.IndexINVERTED,
		},
		{
			name:      "ngram",
			field:     fastEstimateTestField(104, schemapb.DataType_VarChar),
			indexType: indexparamcheck.IndexNGRAM,
		},
		{
			name:      "rtree",
			field:     fastEstimateTestField(105, schemapb.DataType_Geometry),
			indexType: indexparamcheck.IndexRTREE,
		},
		{
			name:      "bitmap mmap",
			field:     fastEstimateTestField(106, schemapb.DataType_Int64),
			indexType: indexparamcheck.IndexBitmap,
			mmap:      "true",
		},
		{
			name:      "hybrid",
			field:     fastEstimateTestField(107, schemapb.DataType_Int64),
			indexType: indexparamcheck.IndexHybrid,
		},
		{
			name:      "hnsw",
			field:     fastEstimateTestField(108, schemapb.DataType_FloatVector),
			indexType: "HNSW",
			metric:    metric.L2,
		},
		{
			name:      "hnsw mmap",
			field:     fastEstimateTestField(109, schemapb.DataType_FloatVector),
			indexType: "HNSW",
			metric:    metric.L2,
			mmap:      "true",
		},
		{
			name:      "sparse bm25 mmap",
			field:     fastEstimateTestField(110, schemapb.DataType_SparseFloatVector),
			indexType: "SPARSE_INVERTED_INDEX",
			metric:    metric.BM25,
			mmap:      "true",
		},
		{
			name:      "sparse ip",
			field:     fastEstimateTestField(111, schemapb.DataType_SparseFloatVector),
			indexType: "SPARSE_WAND",
			metric:    metric.IP,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexInfo := fastEstimateTestIndexInfo(test.field.GetFieldID(), indexSize, test.indexType, test.metric, test.mmap)

			fastEstimate, ok := estimateIndexResourceFast(test.field, indexInfo)
			require.True(t, ok)

			cxxEstimate, err := estimateLoadIndexResource(
				context.Background(),
				test.field,
				fastEstimateTestLoadInfo(indexInfo),
				indexInfo,
			)
			require.NoError(t, err)
			assert.Equal(t, cxxEstimate, fastEstimate)
		})
	}
}

func fastEstimateTestField(fieldID int64, dataType schemapb.DataType) *schemapb.FieldSchema {
	field := &schemapb.FieldSchema{
		FieldID:  fieldID,
		Name:     "field",
		DataType: dataType,
	}
	if dataType == schemapb.DataType_FloatVector {
		field.TypeParams = []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "128"}}
	} else if dataType == schemapb.DataType_VarChar {
		field.TypeParams = []*commonpb.KeyValuePair{{Key: common.MaxLengthKey, Value: "128"}}
	}
	return field
}

func fastEstimateTestIndexInfo(fieldID int64, indexSize int64, indexType string, metricType string, mmap string) *querypb.FieldIndexInfo {
	params := []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: indexType}}
	if metricType != "" {
		params = append(params, &commonpb.KeyValuePair{Key: common.MetricTypeKey, Value: metricType})
	}
	if mmap != "" {
		params = append(params, &commonpb.KeyValuePair{Key: common.MmapEnabledKey, Value: mmap})
	}
	return &querypb.FieldIndexInfo{
		FieldID:        fieldID,
		IndexID:        1,
		BuildID:        2,
		IndexVersion:   3,
		IndexSize:      indexSize,
		NumRows:        120,
		IndexFilePaths: []string{"test-index-file"},
		IndexParams:    params,
	}
}

func fastEstimateTestLoadInfo(indexInfo *querypb.FieldIndexInfo) *querypb.SegmentLoadInfo {
	return &querypb.SegmentLoadInfo{
		CollectionID: 1,
		PartitionID:  2,
		SegmentID:    3,
		NumOfRows:    indexInfo.GetNumRows(),
		IndexInfos:   []*querypb.FieldIndexInfo{indexInfo},
	}
}

func TestEstimateKnowhereVectorIndexResourceFastSkipsSpecializedIndex(t *testing.T) {
	actual, ok := estimateIndexResourceFast(&schemapb.FieldSchema{DataType: schemapb.DataType_FloatVector}, &querypb.FieldIndexInfo{
		IndexSize: 1024,
		IndexParams: []*commonpb.KeyValuePair{
			{Key: common.IndexTypeKey, Value: "HNSW_SQ"},
		},
	})
	require.False(t, ok)
	assert.Equal(t, ResourceEstimate{}, actual)
}
