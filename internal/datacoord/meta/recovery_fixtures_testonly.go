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

// Package datacoord contains core functions in datacoord
//go:build test

package meta

import (
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

var (
	SegmentIndexFromManifestForTest      = segmentIndexFromManifest
	ManifestIndexUsableForTest           = manifestIndexUsable
	ManifestIndexUsableForSegmentForTest = manifestIndexUsableForSegment
	ManifestIndexFilePathInfoForTest     = manifestIndexFilePathInfo
	ProjectManifestIndexesForTest        = projectManifestIndexes
)

type ManifestReadResultForTest struct {
	Segment *SegmentInfo
	Entries []packed.ManifestIndexInfo
	Indexes []*model.SegmentIndex
	Err     error
	Invalid error
}

func (m *Meta) SetManifestReadPrefetchForTest(root string, results []ManifestReadResultForTest) {
	ch := make(chan manifestIndexReadResult, len(results))
	for _, result := range results {
		ch <- manifestIndexReadResult{request: manifestIndexReadRequest{segment: result.Segment}, entries: result.Entries, indexes: result.Indexes, err: result.Err, invalid: result.Invalid}
	}
	close(ch)
	m.manifestReadPrefetch = conc.Go(func() (*ManifestReadPrefetch, error) { return &ManifestReadPrefetch{rootPath: root, results: ch}, nil })
}

func (m *Meta) HasManifestReadPrefetchForTest() bool { return m.manifestReadPrefetch != nil }
