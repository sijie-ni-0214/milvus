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

// Package metacache provides a centralized, in-memory store for segment and
// collection metadata that is shared across coordinators when running in
// MixCoord mode. It exposes a read-only MetaView (consumed by QueryCoord)
// and a read-write MetaStore (consumed by DataCoord) backed by the same
// underlying data, avoiding metadata duplication between coordinators.
package metacache

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

// RecoveryRead holds the independent segment and checkpoint scans. The owner
// must Close it on every exit, including a failed scan, before retrying.
type RecoveryRead struct {
	Segments    *conc.Future[[][]*datapb.SegmentInfo]
	Checkpoints *conc.Future[map[string]*msgpb.MsgPosition]
	cancel      context.CancelFunc
}

func StartRecoveryRead(ctx context.Context, catalog metastore.DataCoordCatalog, collectionIDs []int64) *RecoveryRead {
	ctx, cancel := context.WithCancel(ctx)
	return &RecoveryRead{
		cancel:      cancel,
		Segments:    conc.Go(func() ([][]*datapb.SegmentInfo, error) { return listRecoverySegments(ctx, catalog, collectionIDs) }),
		Checkpoints: conc.Go(func() (map[string]*msgpb.MsgPosition, error) { return catalog.ListChannelCheckpoint(ctx) }),
	}
}

func (r *RecoveryRead) Close() {
	r.cancel()
	_, _ = r.Segments.Await()
	_, _ = r.Checkpoints.Await()
}
