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
	"github.com/milvus-io/milvus/internal/datacoord/index"
	"github.com/milvus-io/milvus/internal/datacoord/meta"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"slices"
)

// metaPrefetch starts only catalog/object-store reads before DataCoord Init.
// No writer or RPC is active until recovery publishes the completed metadata.
type metaPrefetch struct {
	cancel    context.CancelFunc
	indexMeta *conc.Future[*index.IndexMeta]
	recovery  *meta.RecoveryPrefetch
}

func startMetaPrefetch(ctx context.Context, catalog metastore.DataCoordCatalog, collectionIDs []int64) *metaPrefetch {
	ctx, cancel := context.WithCancel(ctx)
	reads := metacache.StartRecoveryRead(ctx, catalog, collectionIDs)
	mlog.Info(ctx, "datacoord meta prefetch started", mlog.Int("numCollections", len(collectionIDs)))
	return &metaPrefetch{
		cancel:    cancel,
		indexMeta: conc.Go(func() (*index.IndexMeta, error) { return index.NewIndexMeta(ctx, catalog, collectionIDs) }),
		recovery: &meta.RecoveryPrefetch{
			Catalog: reads,
			Manifests: conc.Go(func() (*meta.ManifestReadPrefetch, error) {
				groups, err := reads.Segments.Await()
				if err != nil {
					return nil, err
				}
				return meta.StartManifestReadPrefetch(ctx, slices.Concat(groups...)), nil
			}),
		},
	}
}

func (p *metaPrefetch) close() {
	if p == nil {
		return
	}
	p.cancel()
	p.recovery.Catalog.Close()
	_, _ = p.indexMeta.Await()
	manifests, _ := p.recovery.Manifests.Await()
	manifests.Close()
}

// StartMetaPrefetch overlaps reads with RootCoord's child-metadata recovery.
// The collection IDs bound the recovery set as they do in normal MetaStore loading.
func (s *Server) StartMetaPrefetch(collectionIDs []int64) {
	s.metaPrefetchOnce.Do(func() {
		if s.catalog == nil {
			return
		}
		s.metaPrefetch.Store(startMetaPrefetch(s.ctx, s.catalog, collectionIDs))
	})
}
