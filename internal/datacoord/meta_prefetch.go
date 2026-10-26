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

package datacoord

import (
	"context"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

// metaPrefetch holds the catalog reads that dominate meta recovery, started
// before DataCoord initializes: the index meta, the segment records and the
// channel checkpoints. The collection IDs they are given only split their key
// ranges, so they read what the same reads would read later. None of the keys
// they read changes before DataCoord recovers: only DataCoord writes them, and
// it serves no RPC and runs no ack callback until then.
type metaPrefetch struct {
	// store is the segment store the records were scanned through; the meta
	// keeps using it, as it notes the records still carrying their binlogs.
	store       *casSegmentStore
	indexMeta   *conc.Future[*indexMeta]
	segments    *conc.Future[prefetchedSegments]
	checkpoints *conc.Future[map[string]*msgpb.MsgPosition]
	// manifests reads the segment manifests that index recovery reads, from
	// the scanned segment records, while the index meta is still recovering.
	manifests *conc.Future[*manifestReadPrefetch]
}

type prefetchedSegments struct {
	records  []*datapb.SegmentInfo
	versions []int64
}

func startMetaPrefetch(ctx context.Context, catalog metastore.DataCoordCatalog, segmentPersist OptimisticTxnPersist[string, *datapb.SegmentInfo], metaRootPath string, collectionIDs []int64) *metaPrefetch {
	mlog.Info(ctx, "datacoord meta prefetch started", mlog.Int("numCollections", len(collectionIDs)))
	store := newCASSegmentStore(segmentPersist, catalog, metaRootPath)
	segments := conc.Go(func() (prefetchedSegments, error) {
		start := time.Now()
		records, versions, err := store.ListSegments(ctx, collectionIDs)
		if err == nil {
			mlog.Info(ctx, "datacoord segment catalog prefetch done",
				mlog.Int("numSegments", len(records)), mlog.Duration("duration", time.Since(start)))
		}
		return prefetchedSegments{records: records, versions: versions}, err
	})
	return &metaPrefetch{
		store: store,
		indexMeta: conc.Go(func() (*indexMeta, error) {
			return recoverIndexMeta(ctx, catalog, collectionIDs)
		}),
		segments: segments,
		checkpoints: conc.Go(func() (map[string]*msgpb.MsgPosition, error) {
			return catalog.ListChannelCheckpoint(ctx)
		}),
		manifests: conc.Go(func() (*manifestReadPrefetch, error) {
			scan, err := segments.Await()
			if err != nil {
				return nil, err
			}
			return startManifestReadPrefetch(ctx, scan.records), nil
		}),
	}
}

// manifestReadPrefetch reads ahead the segment manifests that index recovery
// reads. A manifest path names one immutable manifest version, so a read of a
// scanned record serves recovery for the installed segment with that manifest
// path; recovery reads the other segments itself.
type manifestReadPrefetch struct {
	rootPath string
	// results receives the final result of every prefetched read, then closes.
	results chan manifestIndexReadResult
}

// startManifestReadPrefetch starts reading the manifests of the records that
// are index recovery candidates. It reads nothing on local storage, where meta
// recovery rewrites manifest paths.
func startManifestReadPrefetch(ctx context.Context, records []*datapb.SegmentInfo) *manifestReadPrefetch {
	config := createStorageConfig()
	if config.GetStorageType() == "local" {
		return nil
	}
	var segments []*SegmentInfo
	for _, record := range records {
		// A bare wrapper: NewSegmentInfo fills in the record's stats, which
		// meta recovery reads meanwhile.
		segment := &SegmentInfo{SegmentInfo: record}
		if manifestIndexRecoveryCandidate(segment) && segment.GetManifestPath() != "" {
			segments = append(segments, segment)
		}
	}
	prefetch := &manifestReadPrefetch{rootPath: config.GetRootPath(), results: make(chan manifestIndexReadResult, len(segments))}
	if len(segments) == 0 {
		close(prefetch.results)
		return prefetch
	}
	start := time.Now()
	reader := newManifestIndexReaderWith(ctx, segments, min(len(segments), segmentIndexManifestReadConcurrency()), config, prefetch.rootPath, admitManifestReadDirectly)
	go func() {
		defer close(prefetch.results)
		defer reader.close()
		for {
			result, err := reader.next()
			if err != nil || result == nil {
				mlog.Info(ctx, "segment manifest prefetch done", mlog.Int("numManifests", len(segments)),
					mlog.Duration("duration", time.Since(start)), mlog.Bool("complete", err == nil))
				return
			}
			// The projected indexes are all recovery takes from a read; the
			// entries would stay alive in the buffer until it installs them.
			result.entries = nil
			prefetch.results <- *result
		}
	}()
	return prefetch
}

// StartMetaPrefetch starts reading the catalog data of meta recovery for the
// given collections ahead of Init; the first meta recovery attempt of Init
// uses what it reads. Only the first call starts the reads, and it does
// nothing unless segments are kept in the optimistic store, whose scans use
// the collection IDs only to split their key ranges. It does not block.
func (s *Server) StartMetaPrefetch(collectionIDs []int64) {
	s.metaPrefetchOnce.Do(func() {
		segmentPersist := s.newSegmentPersist()
		if s.catalog == nil || segmentPersist == nil {
			return
		}
		s.metaPrefetch.Store(startMetaPrefetch(s.ctx, s.catalog, segmentPersist, metaStoreRootPath(), collectionIDs))
	})
}
