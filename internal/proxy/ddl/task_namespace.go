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

package ddl

import (
	"context"
	"strconv"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type CreateNamespaceTask struct {
	baseTask
	Condition
	*milvuspb.CreateNamespaceRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.CreateNamespaceResponse
}

func (t *CreateNamespaceTask) TraceCtx() context.Context { return t.ctx }
func (t *CreateNamespaceTask) ID() UniqueID              { return t.Base.MsgID }
func (t *CreateNamespaceTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *CreateNamespaceTask) Name() string              { return CreateNamespaceTaskName }
func (t *CreateNamespaceTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *CreateNamespaceTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *CreateNamespaceTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *CreateNamespaceTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *CreateNamespaceTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_CreatePartition
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *CreateNamespaceTask) PreExecute(ctx context.Context) error {
	if err := validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName()); err != nil {
		return err
	}
	return validatePartitionTag(t.GetNamespaceName(), true)
}

func (t *CreateNamespaceTask) Execute(ctx context.Context) error {
	resp, err := t.mixCoord.CreateNamespace(ctx, t.CreateNamespaceRequest)
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	collectionID, err := t.GetMetaCache().GetCollectionID(ctx, t.GetDbName(), t.GetCollectionName())
	if err != nil {
		return err
	}
	status, err := t.mixCoord.SyncNewCreatedPartition(ctx, &querypb.SyncNewCreatedPartitionRequest{
		Base: commonpbutil.NewMsgBase(
			commonpbutil.WithMsgType(commonpb.MsgType_CreatePartition),
		),
		CollectionID: collectionID,
		PartitionID:  resp.GetPartitionID(),
	})
	if err = merr.CheckRPCCall(status, err); err != nil {
		return err
	}
	t.result = &milvuspb.CreateNamespaceResponse{
		Status:    merr.Success(),
		Namespace: resp.GetNamespace(),
	}
	return nil
}

func (t *CreateNamespaceTask) PostExecute(ctx context.Context) error { return nil }

type DescribeNamespaceTask struct {
	baseTask
	Condition
	*milvuspb.DescribeNamespaceRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.DescribeNamespaceResponse
}

func (t *DescribeNamespaceTask) TraceCtx() context.Context { return t.ctx }
func (t *DescribeNamespaceTask) ID() UniqueID              { return t.Base.MsgID }
func (t *DescribeNamespaceTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *DescribeNamespaceTask) Name() string              { return DescribeNamespaceTaskName }
func (t *DescribeNamespaceTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *DescribeNamespaceTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *DescribeNamespaceTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *DescribeNamespaceTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *DescribeNamespaceTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_ShowPartitions
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *DescribeNamespaceTask) PreExecute(ctx context.Context) error {
	if err := validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName()); err != nil {
		return err
	}
	return validatePartitionTag(t.GetNamespaceName(), true)
}

func (t *DescribeNamespaceTask) Execute(ctx context.Context) error {
	resp, err := t.mixCoord.DescribeNamespace(ctx, t.DescribeNamespaceRequest)
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	t.result = resp
	return nil
}

func (t *DescribeNamespaceTask) PostExecute(ctx context.Context) error { return nil }

type ListNamespacesTask struct {
	baseTask
	Condition
	*milvuspb.ListNamespacesRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.ListNamespacesResponse
}

func (t *ListNamespacesTask) TraceCtx() context.Context { return t.ctx }
func (t *ListNamespacesTask) ID() UniqueID              { return t.Base.MsgID }
func (t *ListNamespacesTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *ListNamespacesTask) Name() string              { return ListNamespacesTaskName }
func (t *ListNamespacesTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *ListNamespacesTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *ListNamespacesTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *ListNamespacesTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *ListNamespacesTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_ShowPartitions
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *ListNamespacesTask) PreExecute(ctx context.Context) error {
	return validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName())
}

func (t *ListNamespacesTask) Execute(ctx context.Context) error {
	resp, err := t.mixCoord.ListNamespaces(ctx, t.ListNamespacesRequest)
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	t.result = resp
	return nil
}

func (t *ListNamespacesTask) PostExecute(ctx context.Context) error { return nil }

type DropNamespaceTask struct {
	baseTask
	Condition
	*milvuspb.DropNamespaceRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.DropNamespaceResponse
}

func (t *DropNamespaceTask) TraceCtx() context.Context { return t.ctx }
func (t *DropNamespaceTask) ID() UniqueID              { return t.Base.MsgID }
func (t *DropNamespaceTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *DropNamespaceTask) Name() string              { return DropNamespaceTaskName }
func (t *DropNamespaceTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *DropNamespaceTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *DropNamespaceTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *DropNamespaceTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *DropNamespaceTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_DropPartition
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *DropNamespaceTask) PreExecute(ctx context.Context) error {
	if err := validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName()); err != nil {
		return err
	}
	if err := validatePartitionTag(t.GetNamespaceName(), true); err != nil {
		return err
	}
	return checkNamespaceNotLoaded(ctx, t.GetMetaCache(), t.mixCoord, t.GetDbName(), t.GetCollectionName(), t.GetNamespaceName())
}

func (t *DropNamespaceTask) Execute(ctx context.Context) error {
	resp, err := t.mixCoord.DropNamespace(ctx, t.DropNamespaceRequest)
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	t.result = resp
	return nil
}

func (t *DropNamespaceTask) PostExecute(ctx context.Context) error { return nil }

type HasNamespaceTask struct {
	baseTask
	Condition
	*milvuspb.HasNamespaceRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.HasNamespaceResponse
}

func (t *HasNamespaceTask) TraceCtx() context.Context { return t.ctx }
func (t *HasNamespaceTask) ID() UniqueID              { return t.Base.MsgID }
func (t *HasNamespaceTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *HasNamespaceTask) Name() string              { return HasNamespaceTaskName }
func (t *HasNamespaceTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *HasNamespaceTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *HasNamespaceTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *HasNamespaceTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *HasNamespaceTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_HasPartition
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *HasNamespaceTask) PreExecute(ctx context.Context) error {
	if err := validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName()); err != nil {
		return err
	}
	return validatePartitionTag(t.GetNamespaceName(), true)
}

func (t *HasNamespaceTask) Execute(ctx context.Context) error {
	resp, err := t.mixCoord.HasNamespace(ctx, t.HasNamespaceRequest)
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	t.result = resp
	return nil
}

func (t *HasNamespaceTask) PostExecute(ctx context.Context) error { return nil }

type GetNamespaceStatsTask struct {
	baseTask
	Condition
	*milvuspb.GetNamespaceStatsRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *milvuspb.GetNamespaceStatsResponse
}

func (t *GetNamespaceStatsTask) TraceCtx() context.Context { return t.ctx }
func (t *GetNamespaceStatsTask) ID() UniqueID              { return t.Base.MsgID }
func (t *GetNamespaceStatsTask) SetID(uid UniqueID)        { t.Base.MsgID = uid }
func (t *GetNamespaceStatsTask) Name() string              { return GetNamespaceStatsTaskName }
func (t *GetNamespaceStatsTask) Type() commonpb.MsgType    { return t.Base.MsgType }
func (t *GetNamespaceStatsTask) BeginTs() Timestamp        { return t.Base.Timestamp }
func (t *GetNamespaceStatsTask) EndTs() Timestamp          { return t.Base.Timestamp }
func (t *GetNamespaceStatsTask) SetTs(ts Timestamp)        { t.Base.Timestamp = ts }

func (t *GetNamespaceStatsTask) OnEnqueue() error {
	if t.Base == nil {
		t.Base = commonpbutil.NewMsgBase()
	}
	t.Base.MsgType = commonpb.MsgType_GetPartitionStatistics
	t.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *GetNamespaceStatsTask) PreExecute(ctx context.Context) error {
	if t.GetExact() {
		return merr.WrapErrParameterInvalidMsg("exact namespace stats are not supported yet")
	}
	if err := validateNamespaceCollection(ctx, t.GetMetaCache(), t.GetDbName(), t.GetCollectionName()); err != nil {
		return err
	}
	return validatePartitionTag(t.GetNamespaceName(), true)
}

func (t *GetNamespaceStatsTask) Execute(ctx context.Context) error {
	collectionID, err := t.GetMetaCache().GetCollectionID(ctx, t.GetDbName(), t.GetCollectionName())
	if err != nil {
		return err
	}
	partitionID, err := t.GetMetaCache().GetPartitionID(ctx, t.GetDbName(), t.GetCollectionName(), t.GetNamespaceName())
	if err != nil {
		if errors.Is(err, merr.ErrPartitionNotFound) {
			return merr.WrapErrNamespaceNotFound(t.GetNamespaceName())
		}
		return err
	}
	if t.GetNamespaceName() == paramtable.Get().CommonCfg.DefaultPartitionName.GetValue() {
		return merr.WrapErrNamespaceNotFound(t.GetNamespaceName())
	}

	resp, err := t.mixCoord.GetPartitionStatistics(ctx, &datapb.GetPartitionStatisticsRequest{
		Base: commonpbutil.UpdateMsgBase(
			t.Base,
			commonpbutil.WithMsgType(commonpb.MsgType_GetPartitionStatistics),
		),
		CollectionID: collectionID,
		PartitionIDs: []int64{partitionID},
	})
	if err = merr.CheckRPCCall(resp, err); err != nil {
		return err
	}
	t.result = &milvuspb.GetNamespaceStatsResponse{
		Status:        merr.Success(),
		NamespaceName: t.GetNamespaceName(),
		Stats:         namespaceStatsFromKVs(ctx, resp.GetStats()),
	}
	return nil
}

func (t *GetNamespaceStatsTask) PostExecute(ctx context.Context) error { return nil }

func validateNamespaceCollection(ctx context.Context, cache Cache, dbName, collectionName string) error {
	if err := validateCollectionName(collectionName); err != nil {
		return err
	}
	collSchema, err := cache.GetCollectionSchema(ctx, dbName, collectionName)
	if err != nil {
		return err
	}
	schema := collSchema.CollectionSchema
	if !schema.GetEnableNamespace() || !common.IsNamespaceModePartition(schema.GetProperties()...) {
		return merr.WrapErrParameterInvalidMsg("namespace APIs require collection with namespace enabled and namespace.mode=partition")
	}
	return nil
}

func checkNamespaceNotLoaded(ctx context.Context, cache Cache, mixCoord types.MixCoordClient, dbName, collectionName, namespaceName string) error {
	collectionID, err := cache.GetCollectionID(ctx, dbName, collectionName)
	if err != nil {
		return err
	}
	partitionID, err := cache.GetPartitionID(ctx, dbName, collectionName, namespaceName)
	if err != nil {
		if errors.Is(err, merr.ErrPartitionNotFound) || errors.Is(err, merr.ErrCollectionNotFound) || errors.Is(err, merr.ErrDatabaseNotFound) {
			return nil
		}
		return err
	}

	collLoaded, err := isCollectionLoaded(ctx, mixCoord, collectionID)
	if err != nil {
		return err
	}
	if !collLoaded {
		return nil
	}
	loaded, err := isPartitionLoaded(ctx, mixCoord, collectionID, partitionID)
	if err != nil {
		return err
	}
	if loaded {
		return merr.WrapErrParameterInvalidMsg("namespace cannot be dropped, namespace is loaded, please release it first")
	}
	return nil
}

func namespaceStatsFromKVs(ctx context.Context, stats []*commonpb.KeyValuePair) *milvuspb.NamespaceStats {
	namespaceStats := &milvuspb.NamespaceStats{
		Stats:           stats,
		EntityCountType: "approximate",
	}
	for _, stat := range stats {
		if stat.GetKey() != "row_count" {
			continue
		}
		count, err := strconv.ParseInt(stat.GetValue(), 10, 64)
		if err != nil {
			mlog.Warn(ctx, "failed to parse namespace row_count stat", mlog.String("value", stat.GetValue()), mlog.Err(err))
			continue
		}
		namespaceStats.ApproxEntityCount = count
		return namespaceStats
	}
	return namespaceStats
}

func (t *CreateNamespaceTask) Result() *milvuspb.CreateNamespaceResponse     { return t.result }
func (t *DescribeNamespaceTask) Result() *milvuspb.DescribeNamespaceResponse { return t.result }
func (t *ListNamespacesTask) Result() *milvuspb.ListNamespacesResponse       { return t.result }
func (t *DropNamespaceTask) Result() *milvuspb.DropNamespaceResponse         { return t.result }
func (t *HasNamespaceTask) Result() *milvuspb.HasNamespaceResponse           { return t.result }
func (t *GetNamespaceStatsTask) Result() *milvuspb.GetNamespaceStatsResponse { return t.result }

// NewCreateNamespaceTask constructs a CreateNamespaceTask.
func NewCreateNamespaceTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.CreateNamespaceRequest) *CreateNamespaceTask {
	return &CreateNamespaceTask{
		baseTask:               baseTask{MetaCache: node.GetMetaCache()},
		Condition:              NewTaskCondition(ctx),
		CreateNamespaceRequest: request,
		ctx:                    ctx,
		mixCoord:               node.MixCoord(),
	}
}

// NewDescribeNamespaceTask constructs a DescribeNamespaceTask.
func NewDescribeNamespaceTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.DescribeNamespaceRequest) *DescribeNamespaceTask {
	return &DescribeNamespaceTask{
		baseTask:                 baseTask{MetaCache: node.GetMetaCache()},
		Condition:                NewTaskCondition(ctx),
		DescribeNamespaceRequest: request,
		ctx:                      ctx,
		mixCoord:                 node.MixCoord(),
	}
}

// NewListNamespacesTask constructs a ListNamespacesTask.
func NewListNamespacesTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.ListNamespacesRequest) *ListNamespacesTask {
	return &ListNamespacesTask{
		baseTask:              baseTask{MetaCache: node.GetMetaCache()},
		Condition:             NewTaskCondition(ctx),
		ListNamespacesRequest: request,
		ctx:                   ctx,
		mixCoord:              node.MixCoord(),
	}
}

// NewDropNamespaceTask constructs a DropNamespaceTask.
func NewDropNamespaceTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.DropNamespaceRequest) *DropNamespaceTask {
	return &DropNamespaceTask{
		baseTask:             baseTask{MetaCache: node.GetMetaCache()},
		Condition:            NewTaskCondition(ctx),
		DropNamespaceRequest: request,
		ctx:                  ctx,
		mixCoord:             node.MixCoord(),
	}
}

// NewHasNamespaceTask constructs a HasNamespaceTask.
func NewHasNamespaceTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.HasNamespaceRequest) *HasNamespaceTask {
	return &HasNamespaceTask{
		baseTask:            baseTask{MetaCache: node.GetMetaCache()},
		Condition:           NewTaskCondition(ctx),
		HasNamespaceRequest: request,
		ctx:                 ctx,
		mixCoord:            node.MixCoord(),
	}
}

// NewGetNamespaceStatsTask constructs a GetNamespaceStatsTask.
func NewGetNamespaceStatsTask(ctx context.Context, node taskmodel.TaskNode, request *milvuspb.GetNamespaceStatsRequest) *GetNamespaceStatsTask {
	return &GetNamespaceStatsTask{
		baseTask:                 baseTask{MetaCache: node.GetMetaCache()},
		Condition:                NewTaskCondition(ctx),
		GetNamespaceStatsRequest: request,
		ctx:                      ctx,
		mixCoord:                 node.MixCoord(),
	}
}
