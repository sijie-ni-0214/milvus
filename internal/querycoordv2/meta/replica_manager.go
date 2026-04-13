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
	"fmt"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/messagespb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util"
	"github.com/milvus-io/milvus/pkg/v2/util/lock"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// ReplicaManagerInterface defines core operations for replica management
type ReplicaManagerInterface interface {
	// Basic operations
	Recover(ctx context.Context, collections []int64) error
	Get(ctx context.Context, id typeutil.UniqueID) *Replica
	Spawn(ctx context.Context, collection int64,
		replicaNumInRG map[string]int, channels []string, loadPriority commonpb.LoadPriority, opts ...SpawnOption) ([]*Replica, error)

	// Replica manipulation
	TransferReplica(ctx context.Context, collectionID typeutil.UniqueID, srcRGName string, dstRGName string, replicaNum int) error
	MoveReplica(ctx context.Context, dstRGName string, toMove []*Replica) error
	RemoveCollection(ctx context.Context, collectionID typeutil.UniqueID) error
	RemoveReplicas(ctx context.Context, collectionID typeutil.UniqueID, replicas ...typeutil.UniqueID) error

	// Query operations
	GetByCollection(ctx context.Context, collectionID typeutil.UniqueID) []*Replica
	GetByCollectionAndNode(ctx context.Context, collectionID, nodeID typeutil.UniqueID) *Replica
	GetByNode(ctx context.Context, nodeID typeutil.UniqueID) []*Replica
	GetByResourceGroup(ctx context.Context, rgName string) []*Replica

	// Node management
	RecoverNodesInCollection(ctx context.Context, collectionID typeutil.UniqueID, rgs map[string]*ResourceGroup) error
	RemoveNode(ctx context.Context, collectionID typeutil.UniqueID, replicaID typeutil.UniqueID, nodes ...typeutil.UniqueID) error
	RemoveSQNode(ctx context.Context, collectionID typeutil.UniqueID, replicaID typeutil.UniqueID, nodes ...typeutil.UniqueID) error

	// Metadata access
	GetResourceGroupByCollection(ctx context.Context, collection typeutil.UniqueID) typeutil.Set[string]
	GetReplicasJSON(ctx context.Context, meta *Meta) string
}

// Add the interface implementation assertion
var _ ReplicaManagerInterface = (*ReplicaManager)(nil)

type ReplicaManager struct {
	// per-collection read-write lock, auto-created/recycled by KeyLock
	collLock *lock.KeyLock[int64]

	// flat index: replicaID → *Replica, lock-free read for Get()
	flatReplicas *typeutil.ConcurrentMap[int64, *Replica]

	// per-collection replica list, lock-free read via ConcurrentMap
	coll2Replicas *typeutil.ConcurrentMap[int64, []*Replica]

	idAllocator func() (int64, error)
	catalog     metastore.QueryCoordCatalog
}

func NewReplicaManager(idAllocator func() (int64, error), catalog metastore.QueryCoordCatalog) *ReplicaManager {
	return &ReplicaManager{
		collLock:      lock.NewKeyLock[int64](),
		flatReplicas:  typeutil.NewConcurrentMap[int64, *Replica](),
		coll2Replicas: typeutil.NewConcurrentMap[int64, []*Replica](),
		idAllocator:   idAllocator,
		catalog:       catalog,
	}
}

// getCollReplica returns the replica with the given ID from the collection's replica list.
// Caller must hold collLock for the collection.
func (m *ReplicaManager) getCollReplica(collectionID, replicaID typeutil.UniqueID) *Replica {
	replicas, ok := m.coll2Replicas.Get(collectionID)
	if !ok {
		return nil
	}
	for _, r := range replicas {
		if r.GetID() == replicaID {
			return r
		}
	}
	return nil
}

// putCollReplica inserts or replaces one or more replicas in the collection's replica list.
// The whole batch becomes visible to lock-free readers through a single slice replacement,
// so multi-replica updates (RecoverNodesInCollection / TransferReplica / MoveReplica) look
// atomic from the outside. flatReplicas is updated first so that the invariant
// "visible via GetByCollection ⇒ visible via Get" always holds.
// Caller must hold collLock for the collection.
func (m *ReplicaManager) putCollReplica(collectionID typeutil.UniqueID, replicas ...*Replica) {
	if len(replicas) == 0 {
		return
	}
	old, _ := m.coll2Replicas.Get(collectionID)
	updates := make(map[int64]*Replica, len(replicas))
	for _, r := range replicas {
		updates[r.GetID()] = r
	}
	newSlice := make([]*Replica, 0, len(old)+len(updates))
	for _, r := range old {
		if nr, ok := updates[r.GetID()]; ok {
			newSlice = append(newSlice, nr)
			delete(updates, r.GetID())
		} else {
			newSlice = append(newSlice, r)
		}
	}
	// Append brand-new replicas (not present in the old slice), preserving input order.
	for _, r := range replicas {
		if _, ok := updates[r.GetID()]; ok {
			newSlice = append(newSlice, r)
			delete(updates, r.GetID())
		}
	}
	// Step 1: publish to flat index first.
	for _, r := range replicas {
		m.flatReplicas.Insert(r.GetID(), r)
	}
	// Step 2: publish the new collection slice in one shot.
	m.coll2Replicas.Insert(collectionID, newSlice)
}

// removeCollReplicas removes replicas by ID from the collection's replica list.
// Returns true if the collection has no replicas left.
// Caller must hold collLock for the collection.
func (m *ReplicaManager) removeCollReplicas(collectionID typeutil.UniqueID, replicaIDs ...int64) bool {
	old, ok := m.coll2Replicas.Get(collectionID)
	if !ok {
		return true
	}
	removeSet := typeutil.NewSet(replicaIDs...)
	newSlice := make([]*Replica, 0, len(old))
	for _, r := range old {
		if !removeSet.Contain(r.GetID()) {
			newSlice = append(newSlice, r)
		}
	}
	if len(newSlice) == 0 {
		m.coll2Replicas.Remove(collectionID)
		return true
	}
	m.coll2Replicas.Insert(collectionID, newSlice)
	return false
}

// ============================================================
// Internal helpers
// ============================================================

// persistReplicas persists replicas to etcd (caller must hold collLock).
func (m *ReplicaManager) persistReplicas(ctx context.Context, replicas ...*Replica) error {
	if len(replicas) == 0 {
		return nil
	}
	replicaPBs := make([]*querypb.Replica, 0, len(replicas))
	for _, replica := range replicas {
		replicaPBs = append(replicaPBs, replica.replicaPB)
	}
	return m.catalog.SaveReplica(ctx, replicaPBs...)
}

// updateReplicasInCollection updates replicas in the collection's in-memory data.
// The slice replacement is batched through putCollReplica so lock-free readers never
// observe a mid-batch mix of old and new replicas.
// Caller must hold collLock.Lock for this collection.
func (m *ReplicaManager) updateReplicasInCollection(collectionID typeutil.UniqueID, replicas ...*Replica) {
	if len(replicas) == 0 {
		return
	}
	for _, replica := range replicas {
		if oldReplica := m.getCollReplica(collectionID, replica.GetID()); oldReplica != nil {
			metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(oldReplica.GetResourceGroup()).Dec()
			metrics.QueryCoordReplicaRONodeTotal.Add(-float64(oldReplica.RONodesCount()))
		}
		metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(replica.GetResourceGroup()).Inc()
		metrics.QueryCoordReplicaRONodeTotal.Add(float64(replica.RONodesCount()))
	}
	m.putCollReplica(collectionID, replicas...)
}

// removeReplicasLocked removes specific replicas while holding collLock.Lock.
// coll2Replicas is updated before flatReplicas so lock-free readers never observe
// a replica that is visible via GetByCollection but missing from Get(id).
func (m *ReplicaManager) removeReplicasLocked(ctx context.Context, collectionID int64, replicaIDs ...int64) error {
	if len(replicaIDs) == 0 {
		return nil
	}
	if err := m.catalog.ReleaseReplica(ctx, collectionID, replicaIDs...); err != nil {
		return err
	}
	m.removeCollReplicas(collectionID, replicaIDs...)
	for _, replicaID := range replicaIDs {
		if replica, ok := m.flatReplicas.Get(replicaID); ok {
			metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(replica.GetResourceGroup()).Dec()
			metrics.QueryCoordReplicaRONodeTotal.Add(-float64(replica.RONodesCount()))
		}
		m.flatReplicas.Remove(replicaID)
	}
	return nil
}

// validateResourceGroups checks if the resource groups are valid.
func (m *ReplicaManager) validateResourceGroups(rgs map[string]typeutil.UniqueSet) error {
	// make sure that node in resource group is mutual exclusive.
	node := typeutil.NewUniqueSet()
	for _, rg := range rgs {
		for id := range rg {
			if node.Contain(id) {
				return errors.New("node in resource group is not mutual exclusive")
			}
			node.Insert(id)
		}
	}
	return nil
}

// getCollectionAssignmentHelper builds an assignment helper from collection replicas.
// Caller must hold collLock for this collection.
func (m *ReplicaManager) getCollectionAssignmentHelper(collectionID typeutil.UniqueID, rgs map[string]typeutil.UniqueSet) (*collectionAssignmentHelper, error) {
	replicas, _ := m.coll2Replicas.Get(collectionID)
	rgToReplicas := make(map[string][]*Replica)
	for _, replica := range replicas {
		rgName := replica.GetResourceGroup()
		if _, ok := rgs[rgName]; !ok {
			return nil, errors.Errorf("lost resource group info, collectionID: %d, replicaID: %d, resourceGroup: %s", collectionID, replica.GetID(), rgName)
		}
		rgToReplicas[rgName] = append(rgToReplicas[rgName], replica)
	}
	return newCollectionAssignmentHelper(collectionID, rgToReplicas, rgs), nil
}

// getSrcReplicasAndCheckIfTransferable checks if the collection can be transferred.
// Caller must hold collLock for this collection.
func (m *ReplicaManager) getSrcReplicasAndCheckIfTransferable(collectionID typeutil.UniqueID, srcRGName string, replicaNum int) ([]*Replica, error) {
	replicas, _ := m.coll2Replicas.Get(collectionID)
	srcReplicas := lo.Filter(replicas, func(replica *Replica, _ int) bool {
		return replica.GetResourceGroup() == srcRGName
	})
	if len(srcReplicas) < replicaNum {
		err := merr.WrapErrParameterInvalid(
			"NumReplica not greater than the number of replica in source resource group", fmt.Sprintf("only found [%d] replicas of collection [%d] in source resource group [%s], but %d require",
				len(srcReplicas),
				collectionID,
				srcRGName,
				replicaNum))
		return nil, err
	}
	return srcReplicas, nil
}

// ============================================================
// Startup recovery
// ============================================================

// Recover recovers the replicas for given collections from meta store
func (m *ReplicaManager) Recover(ctx context.Context, collections []int64) error {
	replicas, err := m.catalog.GetReplicas(ctx)
	if err != nil {
		return fmt.Errorf("failed to recover replicas, err=%w", err)
	}

	collectionSet := typeutil.NewUniqueSet(collections...)
	for _, replica := range replicas {
		if len(replica.GetResourceGroup()) == 0 {
			replica.ResourceGroup = DefaultResourceGroupName
		}

		if collectionSet.Contain(replica.GetCollectionID()) {
			rep := NewReplicaWithPriority(replica, commonpb.LoadPriority_HIGH)
			collID := rep.GetCollectionID()
			m.putCollReplica(collID, rep)
			metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(rep.GetResourceGroup()).Inc()
			metrics.QueryCoordReplicaRONodeTotal.Add(float64(rep.RONodesCount()))
			log.Info("recover replica",
				zap.Int64("collectionID", replica.GetCollectionID()),
				zap.Int64("replicaID", replica.GetID()),
				zap.Int64s("rwNodes", replica.GetNodes()),
				zap.Int64s("roNodes", replica.GetRoNodes()),
				zap.Int64s("rwSQNodes", replica.GetRwSqNodes()),
				zap.Int64s("roSQNodes", replica.GetRoNodes()),
			)
		} else {
			err := m.catalog.ReleaseReplica(ctx, replica.GetCollectionID(), replica.GetID())
			if err != nil {
				return err
			}
			log.Info("clear stale replica",
				zap.Int64("collectionID", replica.GetCollectionID()),
				zap.Int64("replicaID", replica.GetID()),
				zap.Int64s("nodes", replica.GetNodes()),
			)
		}
	}
	return nil
}

// ============================================================
// Read methods
// ============================================================

// Get returns the replica by id.
// Replica should be read-only, do not modify it.
// Uses lock-free flat index for fast lookup; safe because Replica is COW-immutable.
func (m *ReplicaManager) Get(ctx context.Context, id typeutil.UniqueID) *Replica {
	replica, _ := m.flatReplicas.Get(id)
	return replica
}

func (m *ReplicaManager) GetByCollection(ctx context.Context, collectionID typeutil.UniqueID) []*Replica {
	replicas, _ := m.coll2Replicas.Get(collectionID)
	return replicas
}

func (m *ReplicaManager) GetByCollectionAndNode(ctx context.Context, collectionID, nodeID typeutil.UniqueID) *Replica {
	replicas, ok := m.coll2Replicas.Get(collectionID)
	if !ok {
		return nil
	}
	for _, replica := range replicas {
		if replica.Contains(nodeID) {
			return replica
		}
	}
	return nil
}

func (m *ReplicaManager) GetByNode(ctx context.Context, nodeID typeutil.UniqueID) []*Replica {
	replicas := make([]*Replica, 0)
	m.coll2Replicas.Range(func(_ int64, collReplicas []*Replica) bool {
		for _, replica := range collReplicas {
			if replica.Contains(nodeID) {
				replicas = append(replicas, replica)
			}
		}
		return true
	})
	return replicas
}

func (m *ReplicaManager) GetByResourceGroup(ctx context.Context, rgName string) []*Replica {
	ret := make([]*Replica, 0)
	m.coll2Replicas.Range(func(_ int64, collReplicas []*Replica) bool {
		for _, replica := range collReplicas {
			if replica.GetResourceGroup() == rgName {
				ret = append(ret, replica)
			}
		}
		return true
	})
	return ret
}

func (m *ReplicaManager) GetResourceGroupByCollection(ctx context.Context, collection typeutil.UniqueID) typeutil.Set[string] {
	replicas := m.GetByCollection(ctx, collection)
	ret := typeutil.NewSet(lo.Map(replicas, func(r *Replica, _ int) string { return r.GetResourceGroup() })...)
	return ret
}

// GetReplicasJSON returns a JSON representation of all replicas managed by the ReplicaManager.
func (m *ReplicaManager) GetReplicasJSON(ctx context.Context, meta *Meta) string {
	allReplicas := make([]*metricsinfo.Replica, 0)
	m.coll2Replicas.Range(func(_ int64, collReplicas []*Replica) bool {
		for _, r := range collReplicas {
			channelToRWNodes := make(map[string][]int64)
			for k, v := range r.replicaPB.GetChannelNodeInfos() {
				channelToRWNodes[k] = v.GetRwNodes()
			}

			collectionInfo := meta.GetCollection(ctx, r.GetCollectionID())
			dbID := util.InvalidDBID
			if collectionInfo == nil {
				log.Ctx(ctx).Warn("failed to get collection info", zap.Int64("collectionID", r.GetCollectionID()))
			} else {
				dbID = collectionInfo.GetDbID()
			}

			allReplicas = append(allReplicas, &metricsinfo.Replica{
				ID:               r.GetID(),
				CollectionID:     r.GetCollectionID(),
				DatabaseID:       dbID,
				RWNodes:          r.GetNodes(),
				ResourceGroup:    r.GetResourceGroup(),
				RONodes:          r.GetRONodes(),
				ChannelToRWNodes: channelToRWNodes,
			})
		}
		return true
	})
	ret, err := json.Marshal(allReplicas)
	if err != nil {
		log.Warn("failed to marshal replicas", zap.Error(err))
		return ""
	}
	return string(ret)
}

// ============================================================
// Write methods - collection-scoped (collLock.Lock)
// ============================================================

// RecoverNodesInCollection recovers all nodes in collection with latest resource group.
// Promise a node will be only assigned to one replica in same collection at same time.
// 1. Move the rw nodes to ro nodes if they are not in related resource group.
// 2. Add new incoming nodes into the replica if they are not in-used by other replicas of same collection.
// 3. replicas in same resource group will shared the nodes in resource group fairly.
func (m *ReplicaManager) RecoverNodesInCollection(ctx context.Context, collectionID typeutil.UniqueID, rgs map[string]*ResourceGroup) error {
	// Build node sets from resource groups (no lock needed).
	rgNodeSets := make(map[string]typeutil.UniqueSet, len(rgs))
	for rgName, rg := range rgs {
		if rg == nil {
			rgNodeSets[rgName] = typeutil.NewUniqueSet()
		} else {
			rgNodeSets[rgName] = typeutil.NewUniqueSet(rg.GetNodes()...)
		}
	}

	if err := m.validateResourceGroups(rgNodeSets); err != nil {
		return err
	}

	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	if _, ok := m.coll2Replicas.Get(collectionID); !ok {
		return errors.Errorf("collection %d not loaded", collectionID)
	}

	// create a helper to do the recover.
	helper, err := m.getCollectionAssignmentHelper(collectionID, rgNodeSets)
	if err != nil {
		return err
	}

	modifiedReplicas := make([]*Replica, 0)
	// recover node by resource group.
	helper.RangeOverResourceGroup(func(replicaHelper *replicasInSameRGAssignmentHelper) {
		replicaHelper.RangeOverReplicas(func(assignment *replicaAssignmentInfo) {
			replica := m.getCollReplica(collectionID, assignment.GetReplicaID())
			// For replicas with needWaitRGReady flag, skip assignment if the RG still has missing nodes.
			if replica.NeedWaitRGReady() {
				rgName := replica.GetResourceGroup()
				if rg := rgs[rgName]; rg != nil && rg.MissingNumOfNodes() > 0 {
					log.RatedInfo(10, "defer node assignment for new replica, resource group not ready",
						zap.Int64("collectionID", collectionID),
						zap.Int64("replicaID", replica.GetID()),
						zap.String("rgName", rgName),
						zap.Int("missingNodes", rg.MissingNumOfNodes()),
					)
					return
				}
			}

			roNodes := assignment.GetNewRONodes()
			recoverableNodes, incomingNodeCount := assignment.GetRecoverNodesAndIncomingNodeCount()
			// There may be not enough incoming nodes for current replica,
			// Even we filtering the nodes that are used by other replica of same collection in other resource group,
			// current replica's expected node may be still used by other replica of same collection in same resource group.
			incomingNode := replicaHelper.AllocateIncomingNodes(incomingNodeCount)
			if len(roNodes) == 0 && len(recoverableNodes) == 0 && len(incomingNode) == 0 {
				// nothing to do.
				return
			}
			mutableReplica := replica.CopyForWrite()
			mutableReplica.AddRONode(roNodes...)          // rw -> ro
			mutableReplica.AddRWNode(recoverableNodes...) // ro -> rw
			mutableReplica.AddRWNode(incomingNode...)     // unused -> rw
			// Clear waitRGReady after first successful node assignment.
			if mutableReplica.NeedWaitRGReady() {
				mutableReplica.SetWaitRGReadyAt(time.Time{})
			}
			log.Info(
				"new replica recovery found",
				zap.Int64("collectionID", collectionID),
				zap.Int64("replicaID", assignment.GetReplicaID()),
				zap.Int64s("newRONodes", roNodes),
				zap.Int64s("roToRWNodes", recoverableNodes),
				zap.Int64s("newIncomingNodes", incomingNode),
				zap.Bool("enableChannelExclusiveMode", mutableReplica.IsChannelExclusiveModeEnabled()),
				zap.Any("channelNodeInfos", mutableReplica.replicaPB.GetChannelNodeInfos()),
				zap.Int64s("rwNodes", mutableReplica.GetRWNodes()),
				zap.Int64s("roNodes", mutableReplica.GetRONodes()),
				zap.Int64s("rwSQNodes", mutableReplica.GetRWSQNodes()),
				zap.Int64s("roSQNodes", mutableReplica.GetROSQNodes()),
			)
			modifiedReplicas = append(modifiedReplicas, mutableReplica.IntoReplica())
		})
	})

	if len(modifiedReplicas) == 0 {
		return nil
	}
	if err := m.persistReplicas(ctx, modifiedReplicas...); err != nil {
		return err
	}
	m.updateReplicasInCollection(collectionID, modifiedReplicas...)
	return nil
}

// RecoverSQNodesInCollection recovers all sq nodes in collection with latest node list.
func (m *ReplicaManager) RecoverSQNodesInCollection(ctx context.Context, collectionID int64, sqnNodeIDs typeutil.UniqueSet) error {
	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	replicas, ok := m.coll2Replicas.Get(collectionID)
	if !ok {
		return errors.Errorf("collection %d not loaded", collectionID)
	}

	helper := newReplicaSQNAssignmentHelper(replicas, sqnNodeIDs)
	helper.updateExpectedNodeCountForReplicas(len(sqnNodeIDs))

	modifiedReplicas := make([]*Replica, 0)
	// recover node by given sqn node list.
	helper.RangeOverReplicas(func(assignment *replicaAssignmentInfo) {
		roNodes := assignment.GetNewRONodes()
		recoverableNodes, incomingNodeCount := assignment.GetRecoverNodesAndIncomingNodeCount()
		incomingNode := helper.AllocateIncomingNodes(incomingNodeCount)
		if len(roNodes) == 0 && len(recoverableNodes) == 0 && len(incomingNode) == 0 {
			// nothing to do.
			return
		}
		mutableReplica := m.getCollReplica(collectionID, assignment.GetReplicaID()).CopyForWrite()
		mutableReplica.AddROSQNode(roNodes...)          // rw -> ro
		mutableReplica.AddRWSQNode(recoverableNodes...) // ro -> rw
		mutableReplica.AddRWSQNode(incomingNode...)     // unused -> rw
		log.Info(
			"new replica recovery streaming query node found",
			zap.Int64("collectionID", collectionID),
			zap.Int64("replicaID", assignment.GetReplicaID()),
			zap.Int64s("newRONodes", roNodes),
			zap.Int64s("roToRWNodes", recoverableNodes),
			zap.Int64s("newIncomingNodes", incomingNode),
			zap.Int64s("rwNodes", mutableReplica.GetRWNodes()),
			zap.Int64s("roNodes", mutableReplica.GetRONodes()),
			zap.Int64s("rwSQNodes", mutableReplica.GetRWSQNodes()),
			zap.Int64s("roSQNodes", mutableReplica.GetROSQNodes()),
		)
		modifiedReplicas = append(modifiedReplicas, mutableReplica.IntoReplica())
	})

	if len(modifiedReplicas) == 0 {
		return nil
	}
	if err := m.persistReplicas(ctx, modifiedReplicas...); err != nil {
		return err
	}
	m.updateReplicasInCollection(collectionID, modifiedReplicas...)
	return nil
}

// RemoveNode removes the node from the given replica.
func (m *ReplicaManager) RemoveNode(ctx context.Context, collectionID typeutil.UniqueID, replicaID typeutil.UniqueID, nodes ...typeutil.UniqueID) error {
	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	replica := m.getCollReplica(collectionID, replicaID)
	if replica == nil {
		return merr.WrapErrReplicaNotFound(replicaID)
	}

	mutableReplica := replica.CopyForWrite()
	mutableReplica.RemoveNode(nodes...) // ro -> unused
	modified := mutableReplica.IntoReplica()

	if err := m.persistReplicas(ctx, modified); err != nil {
		return err
	}
	m.updateReplicasInCollection(collectionID, modified)
	return nil
}

// RemoveSQNode removes the sq node from the given replica.
func (m *ReplicaManager) RemoveSQNode(ctx context.Context, collectionID typeutil.UniqueID, replicaID typeutil.UniqueID, nodes ...typeutil.UniqueID) error {
	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	replica := m.getCollReplica(collectionID, replicaID)
	if replica == nil {
		return merr.WrapErrReplicaNotFound(replicaID)
	}

	mutableReplica := replica.CopyForWrite()
	mutableReplica.RemoveSQNode(nodes...) // ro -> unused
	modified := mutableReplica.IntoReplica()

	if err := m.persistReplicas(ctx, modified); err != nil {
		return err
	}
	m.updateReplicasInCollection(collectionID, modified)
	return nil
}

// TransferReplica transfers N replicas from srcRGName to dstRGName.
func (m *ReplicaManager) TransferReplica(ctx context.Context, collectionID typeutil.UniqueID, srcRGName string, dstRGName string, replicaNum int) error {
	if srcRGName == dstRGName {
		return merr.WrapErrParameterInvalidMsg("source resource group and target resource group should not be the same, resource group: %s", srcRGName)
	}
	if replicaNum <= 0 {
		return merr.WrapErrParameterInvalid("NumReplica > 0", fmt.Sprintf("invalid NumReplica %d", replicaNum))
	}

	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	if _, ok := m.coll2Replicas.Get(collectionID); !ok {
		return merr.WrapErrParameterInvalid(
			"Collection not loaded",
			fmt.Sprintf("collectionID %d", collectionID),
		)
	}

	// Check if replica can be transferred.
	srcReplicas, err := m.getSrcReplicasAndCheckIfTransferable(collectionID, srcRGName, replicaNum)
	if err != nil {
		return err
	}

	// Transfer N replicas from srcRGName to dstRGName.
	// Node Change will be executed by replica_observer in background.
	replicas := make([]*Replica, 0, replicaNum)
	for i := 0; i < replicaNum; i++ {
		mutableReplica := srcReplicas[i].CopyForWrite()
		mutableReplica.SetResourceGroup(dstRGName)
		replicas = append(replicas, mutableReplica.IntoReplica())
	}

	if err := m.persistReplicas(ctx, replicas...); err != nil {
		return err
	}
	m.updateReplicasInCollection(collectionID, replicas...)
	return nil
}

func (m *ReplicaManager) MoveReplica(ctx context.Context, dstRGName string, toMove []*Replica) error {
	if len(toMove) == 0 {
		return nil
	}

	// Group by collectionID so each collection is locked and updated independently.
	grouped := make(map[int64][]*Replica)
	for _, replica := range toMove {
		collID := replica.GetCollectionID()
		grouped[collID] = append(grouped[collID], replica)
	}

	for collectionID, collToMove := range grouped {
		m.collLock.Lock(collectionID)

		if _, ok := m.coll2Replicas.Get(collectionID); !ok {
			m.collLock.Unlock(collectionID)
			continue
		}

		replicas := make([]*Replica, 0, len(collToMove))
		replicaIDs := make([]int64, 0, len(collToMove))
		for _, replica := range collToMove {
			mutableReplica := replica.CopyForWrite()
			mutableReplica.SetResourceGroup(dstRGName)
			replicas = append(replicas, mutableReplica.IntoReplica())
			replicaIDs = append(replicaIDs, replica.GetID())
		}
		log.Info("move replicas to resource group", zap.String("dstRGName", dstRGName), zap.Int64s("replicas", replicaIDs))

		if err := m.persistReplicas(ctx, replicas...); err != nil {
			m.collLock.Unlock(collectionID)
			return err
		}
		m.updateReplicasInCollection(collectionID, replicas...)
		m.collLock.Unlock(collectionID)
	}
	return nil
}

// ============================================================
// Write methods - structural (create/delete replicas)
// ============================================================

type SpawnWithReplicaConfigParams struct {
	CollectionID int64
	Channels     []string
	Configs      []*messagespb.LoadReplicaConfig
}

// SpawnOption is a functional option for Spawn.
type SpawnOption func(*spawnConfig)

type spawnConfig struct {
	waitRGReady bool
}

// WithNeedWaitRGReady returns a SpawnOption that enables waiting for resource group readiness.
func WithNeedWaitRGReady() SpawnOption {
	return func(cfg *spawnConfig) {
		cfg.waitRGReady = true
	}
}

// AllocateReplicaID allocates a replica ID.
func (m *ReplicaManager) AllocateReplicaID(ctx context.Context) (int64, error) {
	return m.idAllocator()
}

// Spawn spawns N replicas at resource group for given collection in ReplicaManager.
func (m *ReplicaManager) Spawn(ctx context.Context, collection int64, replicaNumInRG map[string]int,
	channels []string, loadPriority commonpb.LoadPriority, opts ...SpawnOption,
) ([]*Replica, error) {
	cfg := &spawnConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	// ID allocation and replica construction (no lock, no side effects)
	balancePolicy := paramtable.Get().QueryCoordCfg.Balancer.GetValue()
	enableChannelExclusiveMode := balancePolicy == ChannelLevelScoreBalancerName

	replicas := make([]*Replica, 0)
	for rgName, replicaNum := range replicaNumInRG {
		for ; replicaNum > 0; replicaNum-- {
			id, err := m.idAllocator()
			if err != nil {
				return nil, err
			}

			replica := NewReplicaWithPriority(&querypb.Replica{
				ID:            id,
				CollectionID:  collection,
				ResourceGroup: rgName,
			}, loadPriority)
			mutableReplica := replica.CopyForWrite()
			if cfg.waitRGReady {
				mutableReplica.SetWaitRGReadyAt(time.Now())
			}
			if enableChannelExclusiveMode {
				mutableReplica.TryEnableChannelExclusiveMode(channels...)
			}
			replica = mutableReplica.IntoReplica()
			replicas = append(replicas, replica)
		}
	}

	m.collLock.Lock(collection)
	defer m.collLock.Unlock(collection)

	// Persist + memory update (under lock)
	if err := m.persistReplicas(ctx, replicas...); err != nil {
		return nil, err
	}
	m.putCollReplica(collection, replicas...)
	for _, r := range replicas {
		metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(r.GetResourceGroup()).Inc()
		metrics.QueryCoordReplicaRONodeTotal.Add(float64(r.RONodesCount()))
	}
	return replicas, nil
}

// SpawnWithReplicaConfig spawns replicas with replica config.
func (m *ReplicaManager) SpawnWithReplicaConfig(ctx context.Context, params SpawnWithReplicaConfigParams) ([]*Replica, error) {
	balancePolicy := paramtable.Get().QueryCoordCfg.Balancer.GetValue()
	enableChannelExclusiveMode := balancePolicy == ChannelLevelScoreBalancerName

	m.collLock.Lock(params.CollectionID)
	defer m.collLock.Unlock(params.CollectionID)

	replicas := make([]*Replica, 0)
	for _, config := range params.Configs {
		existedReplica := m.getCollReplica(params.CollectionID, config.GetReplicaId())
		if existedReplica != nil {
			// if the replica already exists, just update the resource group
			mutableReplica := existedReplica.CopyForWrite()
			mutableReplica.SetResourceGroup(config.GetResourceGroupName())
			replicas = append(replicas, mutableReplica.IntoReplica())
			continue
		}
		replica := NewReplicaWithPriority(&querypb.Replica{
			ID:            config.GetReplicaId(),
			CollectionID:  params.CollectionID,
			ResourceGroup: config.ResourceGroupName,
		}, config.GetPriority())
		if enableChannelExclusiveMode {
			mutableReplica := replica.CopyForWrite()
			mutableReplica.TryEnableChannelExclusiveMode(params.Channels...)
			replica = mutableReplica.IntoReplica()
		}
		replicas = append(replicas, replica)
		log.Ctx(ctx).Info("spawn replica for collection",
			zap.Int64("collectionID", params.CollectionID),
			zap.Int64("replicaID", config.GetReplicaId()),
			zap.String("resourceGroup", config.GetResourceGroupName()),
		)
	}
	if err := m.persistReplicas(ctx, replicas...); err != nil {
		return nil, errors.Wrap(err, "failed to put replicas")
	}
	// Persist succeeded, now safe to update in-memory state.
	m.updateReplicasInCollection(params.CollectionID, replicas...)

	// Remove redundant replicas
	if err := m.removeRedundantReplicas(ctx, params); err != nil {
		return nil, errors.Wrap(err, "failed to remove redundant replicas")
	}
	return replicas, nil
}

// removeRedundantReplicas removes redundant replicas that are not in the new replica config.
func (m *ReplicaManager) removeRedundantReplicas(ctx context.Context, params SpawnWithReplicaConfigParams) error {
	replicas, _ := m.coll2Replicas.Get(params.CollectionID)
	toRemoveReplicas := make([]int64, 0)
	for _, replica := range replicas {
		found := false
		replicaID := replica.GetID()
		for _, config := range params.Configs {
			if config.GetReplicaId() == replicaID {
				found = true
				break
			}
		}
		if !found {
			toRemoveReplicas = append(toRemoveReplicas, replicaID)
		}
	}
	return m.removeReplicasLocked(ctx, params.CollectionID, toRemoveReplicas...)
}

// Deprecated: Warning, break the consistency of ReplicaManager,
// never use it in non-test code, use Spawn instead.
func (m *ReplicaManager) Put(ctx context.Context, replicas ...*Replica) error {
	if len(replicas) == 0 {
		return nil
	}
	// Group by collectionID, each group under its own collLock
	grouped := make(map[int64][]*Replica)
	for _, r := range replicas {
		grouped[r.GetCollectionID()] = append(grouped[r.GetCollectionID()], r)
	}
	for collID, groupedReplicas := range grouped {
		m.collLock.Lock(collID)
		if err := m.persistReplicas(ctx, groupedReplicas...); err != nil {
			m.collLock.Unlock(collID)
			return err
		}
		m.updateReplicasInCollection(collID, groupedReplicas...)
		m.collLock.Unlock(collID)
	}
	return nil
}

// RemoveCollection removes replicas of given collection,
// returns error if failed to remove replica from KV
func (m *ReplicaManager) RemoveCollection(ctx context.Context, collectionID typeutil.UniqueID) error {
	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	replicas, ok := m.coll2Replicas.Get(collectionID)
	if !ok {
		return nil
	}

	err := m.catalog.ReleaseReplicas(ctx, collectionID)
	if err != nil {
		return err
	}

	// coll2Replicas is updated before flatReplicas so the invariant
	// "visible via GetByCollection ⇒ visible via Get" holds during deletion.
	m.coll2Replicas.Remove(collectionID)
	for _, replica := range replicas {
		metrics.QueryCoordResourceGroupReplicaTotal.WithLabelValues(replica.GetResourceGroup()).Dec()
		metrics.QueryCoordReplicaRONodeTotal.Add(-float64(replica.RONodesCount()))
		m.flatReplicas.Remove(replica.GetID())
	}
	return nil
}

func (m *ReplicaManager) RemoveReplicas(ctx context.Context, collectionID typeutil.UniqueID, replicaIDs ...typeutil.UniqueID) error {
	m.collLock.Lock(collectionID)
	defer m.collLock.Unlock(collectionID)

	if _, ok := m.coll2Replicas.Get(collectionID); !ok {
		return nil
	}

	log.Info("release replicas", zap.Int64("collectionID", collectionID), zap.Int64s("replicas", replicaIDs))
	return m.removeReplicasLocked(ctx, collectionID, replicaIDs...)
}
