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

package meta

import (
	"context"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"time"
)

// ManifestReadPrefetch reads ahead the segment manifests that index recovery
// reads. A manifest path names one immutable manifest version, so a read of a
// scanned record serves recovery for the installed segment with that manifest
// path; recovery reads the other segments itself.
type ManifestReadPrefetch struct {
	rootPath string
	cancel   context.CancelFunc
	done     chan struct{}
	// results receives the final result of every prefetched read, then closes.
	results chan manifestIndexReadResult
}

// StartManifestReadPrefetch starts reading the manifests of the records that
// are index recovery candidates. It reads nothing on local storage, where meta
// recovery rewrites manifest paths.
func StartManifestReadPrefetch(ctx context.Context, records []*datapb.SegmentInfo) *ManifestReadPrefetch {
	config := CreateStorageConfig()
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
	ctx, cancel := context.WithCancel(ctx)
	prefetch := &ManifestReadPrefetch{cancel: cancel, done: make(chan struct{}), rootPath: config.GetRootPath(), results: make(chan manifestIndexReadResult, len(segments))}
	if len(segments) == 0 {
		close(prefetch.results)
		close(prefetch.done)
		return prefetch
	}
	start := time.Now()
	reader := newManifestIndexReaderWith(ctx, segments, min(len(segments), SegmentIndexManifestReadConcurrency()), config, prefetch.rootPath, admitManifestReadDirectly)
	go func() {
		defer close(prefetch.done)
		defer close(prefetch.results)
		defer reader.Close()
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

// Close cancels speculative reads and joins their reader before startup exits.
func (p *ManifestReadPrefetch) Close() {
	if p == nil {
		return
	}
	p.cancel()
	<-p.done
}
