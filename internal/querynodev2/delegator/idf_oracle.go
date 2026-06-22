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

package delegator

/*
#cgo pkg-config: milvus_core

#include "segcore/load_index_c.h"
*/
import "C"

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"runtime"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/pathutil"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const memoryHeadroom = 4 * 1024 * 1024 // 4MB headroom for Insert path, ~50K unique tokens

type IDFOracle interface {
	SetNext(snapshot *snapshot)
	TargetVersion() int64

	UpdateGrowing(segmentID int64, stats bm25Stats)
	// mark growing segment remove target version
	LazyRemoveGrowings(targetVersion int64, segmentIDs ...int64)

	RegisterGrowing(segmentID int64, stats bm25Stats, partitionID ...int64)
	// LoadSealed loads BM25 stats for a sealed segment from remote storage.
	// Internally handles: streaming download → local disk → optional parse → register.
	// Idempotent: skips if segment already loaded.
	LoadSealed(ctx context.Context, segmentID int64, loadInfo *querypb.SegmentLoadInfo, cm storage.ChunkManager) error

	BuildIDF(ctx context.Context, fieldID int64, tfs *schemapb.SparseFloatArray, partitions ...int64) ([][]byte, float64, error)

	DirPath() string

	Start()
	Close()
}

type bm25Stats map[int64]*storage.BM25Stats

func (s bm25Stats) Clone() bm25Stats {
	result := make(bm25Stats, len(s))
	for fieldID, stats := range s {
		result[fieldID] = stats.Clone()
	}
	return result
}

func (s bm25Stats) MemSize() int64 {
	size := int64(0)
	for _, stats := range s {
		size += stats.MemSize()
	}
	return size
}

func (s bm25Stats) Merge(stats bm25Stats) {
	for fieldID, newstats := range stats {
		if stats, ok := s[fieldID]; ok {
			stats.Merge(newstats)
		} else {
			s[fieldID] = storage.NewBM25Stats()
			s[fieldID].Merge(newstats)
		}
	}
}

func (s bm25Stats) Minus(stats bm25Stats) {
	for fieldID, newstats := range stats {
		if stats, ok := s[fieldID]; ok {
			stats.Minus(newstats)
		} else {
			s[fieldID] = storage.NewBM25Stats()
			s[fieldID].Minus(newstats)
		}
	}
}

func (s bm25Stats) GetStats(fieldID int64) (*storage.BM25Stats, error) {
	stats, ok := s[fieldID]
	if !ok {
		return nil, errors.New("field not found in idf oracle BM25 stats")
	}
	return stats, nil
}

func (s bm25Stats) NumRow() int64 {
	for _, stats := range s {
		return stats.NumRow()
	}
	return 0
}

type sealedBm25Stats struct {
	sync.RWMutex // Protect all data in struct except activate

	activate *atomic.Bool

	removed         bool
	segmentID       int64
	partitionID     int64
	ts              time.Time // Time of segment register
	localDir        string
	remoteFetchOnly bool
	remotePaths     map[int64][]string
	pathsResolved   bool
	localReady      bool
	cm              storage.ChunkManager
	fieldList       []int64 // bm25 field list
	source          sealedBm25Source
	diskSize        int64 // total disk size of local files
	diskSizeTracker *atomic.Int64
}

type sealedBm25Source struct {
	manifestPath string
	bm25Logs     []*datapb.FieldBinlog
}

func newSealedBm25Source(loadInfo *querypb.SegmentLoadInfo) sealedBm25Source {
	source := sealedBm25Source{
		manifestPath: loadInfo.GetManifestPath(),
	}
	if source.manifestPath == "" && len(loadInfo.GetBm25Logs()) > 0 {
		source.bm25Logs = make([]*datapb.FieldBinlog, 0, len(loadInfo.GetBm25Logs()))
		for _, log := range loadInfo.GetBm25Logs() {
			source.bm25Logs = append(source.bm25Logs, typeutil.Clone(log))
		}
	}
	return source
}

func (s sealedBm25Source) hasSource() bool {
	return s.manifestPath != "" || len(s.bm25Logs) > 0
}

func (s sealedBm25Source) BM25StatsPaths() (map[int64][]string, error) {
	if s.manifestPath != "" {
		return packed.NewStatsResolver(s.manifestPath, packed.CreateStorageConfig()).BM25StatsPaths()
	}
	return packed.NewStatsResolver("", nil).WithBM25Logs(s.bm25Logs).BM25StatsPaths()
}

func (s *sealedBm25Stats) Remove() {
	s.Lock()
	defer s.Unlock()
	s.removed = true

	if s.localDir != "" {
		err := os.RemoveAll(s.localDir)
		if err != nil {
			log.Warn("remove local bm25 stats failed", zap.Error(err), zap.String("path", s.localDir))
		}
	}
}

// FetchStats reads stats from the configured source and merges per field.
// Local directory structure: {localDir}/{fieldID}/0.data, 1.data, ...
func (s *sealedBm25Stats) FetchStats(ctxs ...context.Context) (map[int64]*storage.BM25Stats, error) {
	ctx := context.Background()
	if len(ctxs) > 0 && ctxs[0] != nil {
		ctx = ctxs[0]
	}

	s.Lock()
	defer s.Unlock()

	if s.removed {
		return nil, errors.Newf("sealed bm25 stats for segment %d already removed", s.segmentID)
	}

	if err := s.resolveRemotePathsLocked(); err != nil {
		return nil, err
	}

	if s.remoteFetchOnly {
		return s.fetchRemoteStats(ctx)
	}
	if err := s.ensureLocalStatsLocked(ctx); err != nil {
		return nil, err
	}
	return s.fetchLocalStats()
}

func (s *sealedBm25Stats) resolveRemotePathsLocked() error {
	if s.pathsResolved || !s.source.hasSource() {
		return nil
	}

	logpaths, err := s.source.BM25StatsPaths()
	if err != nil {
		return err
	}

	s.remotePaths = logpaths
	s.fieldList = bm25FieldList(logpaths)
	s.pathsResolved = true
	return nil
}

func (s *sealedBm25Stats) ensureLocalStatsLocked(ctx context.Context) error {
	if s.localReady {
		return nil
	}
	if s.localDir == "" {
		return errors.Newf("local dir is empty for segment %d", s.segmentID)
	}
	if s.cm == nil {
		return errors.Newf("remote chunk manager is nil for segment %d", s.segmentID)
	}

	var totalDiskSize int64
	for _, fieldID := range s.fieldList {
		fieldDir := path.Join(s.localDir, fmt.Sprintf("%d", fieldID))
		if err := os.MkdirAll(fieldDir, os.ModePerm); err != nil {
			return err
		}

		for i, remotePath := range s.remotePaths[fieldID] {
			localFile := path.Join(fieldDir, fmt.Sprintf("%d.data", i))
			written, err := streamOneFile(ctx, s.cm, remotePath, localFile, nil)
			if err != nil {
				return errors.Wrapf(err, "stream bm25 stats file %s", remotePath)
			}
			totalDiskSize += written
		}
	}

	s.localReady = true
	s.diskSize += totalDiskSize
	if s.diskSizeTracker != nil && totalDiskSize > 0 {
		s.diskSizeTracker.Add(totalDiskSize)
	}
	return nil
}

func (s *sealedBm25Stats) fetchLocalStats() (map[int64]*storage.BM25Stats, error) {
	stats := make(map[int64]*storage.BM25Stats)
	for _, fieldID := range s.fieldList {
		fieldDir := path.Join(s.localDir, fmt.Sprintf("%d", fieldID))
		entries, err := os.ReadDir(fieldDir)
		if err != nil {
			return nil, errors.Newf("read local dir %s failed: %v", fieldDir, err)
		}

		fieldStats := storage.NewBM25Stats()
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			filePath := path.Join(fieldDir, entry.Name())
			f, err := os.Open(filePath)
			if err != nil {
				return nil, errors.Newf("open local file %s failed: %v", filePath, err)
			}
			err = fieldStats.DeserializeFromReader(bufio.NewReader(f))
			f.Close()
			if err != nil {
				return nil, errors.Newf("deserialize local file %s failed: %v", filePath, err)
			}
		}
		stats[fieldID] = fieldStats
	}

	return stats, nil
}

func (s *sealedBm25Stats) fetchRemoteStats(ctx context.Context) (map[int64]*storage.BM25Stats, error) {
	if s.cm == nil {
		return nil, errors.Newf("remote chunk manager is nil for segment %d", s.segmentID)
	}

	stats := make(map[int64]*storage.BM25Stats, len(s.remotePaths))
	for _, fieldID := range s.fieldList {
		fieldStats := storage.NewBM25Stats()
		for _, remotePath := range s.remotePaths[fieldID] {
			if err := readRemoteBM25Stats(ctx, s.cm, remotePath, fieldStats); err != nil {
				return nil, errors.Wrapf(err, "read remote bm25 stats %s", remotePath)
			}
		}
		stats[fieldID] = fieldStats
	}
	return stats, nil
}

type growingBm25Stats struct {
	bm25Stats

	partitionID    int64
	activate       bool
	droppedVersion int64
}

type partitionBm25Stats struct {
	targetVersion int64
	stats         bm25Stats
}

func newBm25Stats(functions []*schemapb.FunctionSchema) bm25Stats {
	stats := make(map[int64]*storage.BM25Stats)

	for _, function := range functions {
		if function.GetType() == schemapb.FunctionType_BM25 {
			stats[function.GetOutputFieldIds()[0]] = storage.NewBM25Stats()
		}
	}
	return stats
}

type idfTarget struct {
	sync.RWMutex
	snapshot *snapshot
	ts       time.Time // time of target generate
}

func (t *idfTarget) SetSnapshot(snapshot *snapshot) {
	t.Lock()
	defer t.Unlock()
	t.snapshot = snapshot
	t.ts = time.Now()
}

func (t *idfTarget) GetSnapshot() (*snapshot, time.Time) {
	t.RLock()
	defer t.RUnlock()
	return t.snapshot, t.ts
}

type idfOracle struct {
	sync.RWMutex   // protect current and growing segment stats
	current        bm25Stats
	growing        map[int64]*growingBm25Stats
	partitionStats map[int64]*partitionBm25Stats

	sealed         typeutil.ConcurrentMap[int64, *sealedBm25Stats]
	sealedDiskSize *atomic.Int64

	channel string

	// for sync distribution
	next          idfTarget
	targetVersion *atomic.Int64
	syncNotify    chan struct{}

	dirPath string

	closeCh chan struct{}
	sf      conc.Singleflight[any]
	wg      sync.WaitGroup

	// resource tracking for caching layer
	resourceMu    sync.Mutex
	chargedMemory int64
	chargedDisk   int64
}

// now only used for test
func (o *idfOracle) TargetVersion() int64 {
	return o.targetVersion.Load()
}

func (o *idfOracle) DirPath() string {
	return o.dirPath
}

func (o *idfOracle) preloadSealed(segmentID int64, stats *sealedBm25Stats, statsToMerge bm25Stats) {
	o.Lock()
	defer o.Unlock()

	// skip preload if first target was loaded.
	if o.targetVersion.Load() != 0 {
		o.sealed.Insert(segmentID, stats)
		return
	}
	o.sealed.Insert(segmentID, stats)
	o.current.Merge(statsToMerge)
	stats.activate.Store(true)
}

func (o *idfOracle) RegisterGrowing(segmentID int64, stats bm25Stats, partitionIDs ...int64) {
	partitionID := int64(0)
	if len(partitionIDs) > 0 {
		partitionID = partitionIDs[0]
	}
	partitionLevel, partitionLazyLoad, _ := idfPartitionLoadConfig()

	o.Lock()
	if _, ok := o.growing[segmentID]; ok {
		o.Unlock()
		return
	}
	o.growing[segmentID] = &growingBm25Stats{
		bm25Stats:   stats,
		partitionID: partitionID,
		activate:    true,
	}
	o.current.Merge(stats)
	if partitionLevel && !partitionLazyLoad {
		o.mergePartitionStatsLocked(partitionID, o.targetVersion.Load(), stats)
	} else {
		o.clearPartitionStatsLocked(partitionID)
	}
	o.Unlock()
	o.syncResource()
}

// LoadSealed loads BM25 stats for a sealed segment from remote storage to local disk.
// Idempotent: skips if segment already loaded.
func (o *idfOracle) LoadSealed(ctx context.Context, segmentID int64, loadInfo *querypb.SegmentLoadInfo, cm storage.ChunkManager) error {
	_, err, _ := o.sf.Do(fmt.Sprintf("load_sealed_%d", segmentID), func() (any, error) {
		if o.sealed.Contain(segmentID) {
			return nil, nil
		}

		partitionLevel, partitionLazyLoad, err := idfPartitionLoadConfig()
		if err != nil {
			return nil, err
		}

		remoteFetchOnly := paramtable.Get().QueryNodeCfg.IDFRemoteFetchOnly.GetAsBool()
		if partitionLevel && partitionLazyLoad {
			if !hasBM25StatsSource(loadInfo) {
				return nil, nil
			}

			segStats := &sealedBm25Stats{
				ts:              time.Now(),
				activate:        atomic.NewBool(false),
				segmentID:       segmentID,
				partitionID:     loadInfo.GetPartitionID(),
				localDir:        partitionLocalDir(o.dirPath, segmentID, remoteFetchOnly),
				remoteFetchOnly: remoteFetchOnly,
				cm:              cm,
				source:          newSealedBm25Source(loadInfo),
				diskSizeTracker: o.sealedDiskSize,
			}
			o.sealed.Insert(segmentID, segStats)

			o.Lock()
			o.clearPartitionStatsLocked(loadInfo.GetPartitionID())
			o.Unlock()

			o.syncResource()
			return nil, nil
		}

		logpaths, err := packed.NewStatsResolverFromLoadInfo(loadInfo).BM25StatsPaths()
		if err != nil {
			log.Warn("load remote segment bm25 stats failed",
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			return nil, err
		}

		if len(logpaths) == 0 {
			return nil, nil
		}

		needParse := !partitionLevel && o.targetVersion.Load() == 0 && paramtable.Get().QueryNodeCfg.IDFPreload.GetAsBool()

		result, err := o.streamLoad(ctx, segmentID, logpaths, cm, needParse, remoteFetchOnly)
		if err != nil {
			// cleanup on failure
			cleanupPath := path.Join(o.dirPath, fmt.Sprintf("%d", segmentID))
			if rmErr := os.RemoveAll(cleanupPath); rmErr != nil {
				log.Warn("failed to cleanup bm25 stats dir on load failure", zap.Error(rmErr), zap.String("path", cleanupPath))
			}
			return nil, err
		}

		segStats := &sealedBm25Stats{
			ts:              time.Now(),
			activate:        atomic.NewBool(false),
			segmentID:       segmentID,
			partitionID:     loadInfo.GetPartitionID(),
			localDir:        result.localDir,
			remoteFetchOnly: result.remoteFetchOnly,
			remotePaths:     result.remotePaths,
			pathsResolved:   true,
			localReady:      !result.remoteFetchOnly,
			cm:              result.cm,
			fieldList:       result.fieldList,
			diskSize:        result.diskSize,
		}

		if needParse && !partitionLevel && result.stats != nil {
			o.preloadSealed(segmentID, segStats, result.stats)
		} else {
			o.sealed.Insert(segmentID, segStats)
		}
		o.sealedDiskSize.Add(result.diskSize)

		o.syncResource()
		return nil, nil
	})
	return err
}

func hasBM25StatsSource(loadInfo *querypb.SegmentLoadInfo) bool {
	return loadInfo.GetManifestPath() != "" || len(loadInfo.GetBm25Logs()) > 0
}

func idfPartitionLoadConfig() (partitionLevel bool, partitionLazyLoad bool, err error) {
	partitionLevel = paramtable.Get().QueryNodeCfg.IDFPartitionLevel.GetAsBool()
	if !partitionLevel {
		return false, false, nil
	}
	partitionLazyLoad = paramtable.Get().QueryNodeCfg.IDFPartitionLazyLoad.GetAsBool()
	return partitionLevel, partitionLazyLoad, nil
}

func partitionLocalDir(baseDir string, segmentID int64, remoteFetchOnly bool) string {
	if remoteFetchOnly {
		return ""
	}
	return path.Join(baseDir, fmt.Sprintf("%d", segmentID))
}

type streamLoadResult struct {
	localDir        string
	remoteFetchOnly bool
	remotePaths     map[int64][]string
	cm              storage.ChunkManager
	fieldList       []int64
	stats           bm25Stats // non-nil only when needParse=true
	diskSize        int64
}

// streamLoad downloads BM25 stats from remote storage to local disk.
// When needParse is true, also parses stats using TeeReader.
func (o *idfOracle) streamLoad(ctx context.Context, segmentID int64, binlogPaths map[int64][]string, cm storage.ChunkManager, needParse bool, remoteFetchOnly bool) (streamLoadResult, error) {
	log := log.Ctx(ctx).With(zap.Int64("segmentID", segmentID))
	startTs := time.Now()

	segDir := path.Join(o.dirPath, fmt.Sprintf("%d", segmentID))
	var totalDiskSize int64
	var stats map[int64]*storage.BM25Stats
	fieldList := make([]int64, 0, len(binlogPaths))

	if remoteFetchOnly {
		if needParse {
			stats = make(map[int64]*storage.BM25Stats, len(binlogPaths))
			for fieldID, paths := range binlogPaths {
				fieldList = append(fieldList, fieldID)
				fieldStats := storage.NewBM25Stats()
				for _, remotePath := range paths {
					if err := readRemoteBM25Stats(ctx, cm, remotePath, fieldStats); err != nil {
						return streamLoadResult{}, errors.Wrapf(err, "read remote bm25 stats %s", remotePath)
					}
				}
				stats[fieldID] = fieldStats
				log.Info("loaded remote bm25 stats", zap.Duration("time", time.Since(startTs)), zap.Int64("numRow", fieldStats.NumRow()), zap.Int64("fieldID", fieldID))
			}
		} else {
			for fieldID := range binlogPaths {
				fieldList = append(fieldList, fieldID)
			}
		}

		log.Info("stream load bm25 stats done", zap.Duration("time", time.Since(startTs)), zap.Int64("diskSize", 0), zap.Bool("parsed", needParse), zap.Bool("remoteFetchOnly", true))
		return streamLoadResult{
			remoteFetchOnly: true,
			remotePaths:     binlogPaths,
			cm:              cm,
			fieldList:       fieldList,
			stats:           stats,
		}, nil
	}

	if needParse {
		stats = make(map[int64]*storage.BM25Stats, len(binlogPaths))
	}

	for fieldID, paths := range binlogPaths {
		fieldList = append(fieldList, fieldID)
		fieldDir := path.Join(segDir, fmt.Sprintf("%d", fieldID))
		if err := os.MkdirAll(fieldDir, os.ModePerm); err != nil {
			return streamLoadResult{}, err
		}

		var fieldStats *storage.BM25Stats
		if needParse {
			fieldStats = storage.NewBM25Stats()
		}

		for i, remotePath := range paths {
			localFile := path.Join(fieldDir, fmt.Sprintf("%d.data", i))
			written, err := streamOneFile(ctx, cm, remotePath, localFile, fieldStats)
			if err != nil {
				return streamLoadResult{}, errors.Wrapf(err, "stream bm25 stats file %s", remotePath)
			}
			totalDiskSize += written
		}

		if needParse {
			stats[fieldID] = fieldStats
			log.Info("loaded bm25 stats", zap.Duration("time", time.Since(startTs)), zap.Int64("numRow", fieldStats.NumRow()), zap.Int64("fieldID", fieldID))
		}
	}

	log.Info("stream load bm25 stats done", zap.Duration("time", time.Since(startTs)), zap.Int64("diskSize", totalDiskSize), zap.Bool("parsed", needParse))

	return streamLoadResult{
		localDir:  segDir,
		fieldList: fieldList,
		stats:     stats,
		diskSize:  totalDiskSize,
	}, nil
}

func readRemoteBM25Stats(ctx context.Context, cm storage.ChunkManager, remotePath string, parseInto *storage.BM25Stats) error {
	reader, err := cm.Reader(ctx, remotePath)
	if err != nil {
		return err
	}
	defer reader.Close()

	br := bufio.NewReaderSize(reader, paramtable.Get().QueryNodeCfg.IDFReadBufferSize.GetAsInt())
	return parseInto.DeserializeFromReader(br)
}

// streamOneFile streams a single remote file to a local file.
// If parseInto is non-nil, uses TeeReader to simultaneously parse stats.
func streamOneFile(ctx context.Context, cm storage.ChunkManager, remotePath, localPath string, parseInto *storage.BM25Stats) (int64, error) {
	reader, err := cm.Reader(ctx, remotePath)
	if err != nil {
		return 0, err
	}
	defer reader.Close()

	f, err := os.Create(localPath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if parseInto != nil {
		br := bufio.NewReaderSize(reader, paramtable.Get().QueryNodeCfg.IDFReadBufferSize.GetAsInt())
		bw := bufio.NewWriter(f)
		tee := io.TeeReader(br, bw)
		err = parseInto.DeserializeFromReader(tee)
		if err != nil {
			return 0, err
		}
		if err := bw.Flush(); err != nil {
			return 0, err
		}
		if err := f.Sync(); err != nil {
			return 0, err
		}
		info, err := f.Stat()
		if err != nil {
			return 0, err
		}
		return info.Size(), nil
	}

	written, err := io.Copy(f, reader)
	if err != nil {
		return 0, err
	}
	if err := f.Sync(); err != nil {
		return 0, err
	}
	return written, nil
}

func (o *idfOracle) UpdateGrowing(segmentID int64, stats bm25Stats) {
	if len(stats) == 0 {
		return
	}

	partitionLevel, partitionLazyLoad, _ := idfPartitionLoadConfig()

	o.Lock()

	old, ok := o.growing[segmentID]
	if !ok {
		o.Unlock()
		return
	}

	old.Merge(stats)
	if old.activate {
		o.current.Merge(stats)
	}
	if partitionLevel && !partitionLazyLoad && old.activate {
		o.mergePartitionStatsLocked(old.partitionID, o.targetVersion.Load(), stats)
	} else {
		o.clearPartitionStatsLocked(old.partitionID)
	}
	if old.activate {
		o.checkMemoryResource()
	}
	o.Unlock()
}

func (o *idfOracle) LazyRemoveGrowings(targetVersion int64, segmentIDs ...int64) {
	partitionLevel, partitionLazyLoad, _ := idfPartitionLoadConfig()

	o.Lock()
	defer o.Unlock()

	for _, segmentID := range segmentIDs {
		if stats, ok := o.growing[segmentID]; ok && stats.droppedVersion == 0 {
			stats.droppedVersion = targetVersion
			if !partitionLevel || partitionLazyLoad {
				o.clearPartitionStatsLocked(stats.partitionID)
			}
		}
	}
}

func (o *idfOracle) getOrCreatePartitionStatsLocked(partitionID int64, targetVersion int64) *partitionBm25Stats {
	stats, ok := o.partitionStats[partitionID]
	if ok {
		stats.targetVersion = targetVersion
		return stats
	}

	stats = &partitionBm25Stats{
		targetVersion: targetVersion,
		stats:         o.emptyStatsLocked(),
	}
	o.partitionStats[partitionID] = stats
	return stats
}

func (o *idfOracle) mergePartitionStatsLocked(partitionID int64, targetVersion int64, stats bm25Stats) {
	if partitionID == 0 || len(stats) == 0 {
		return
	}
	o.getOrCreatePartitionStatsLocked(partitionID, targetVersion).stats.Merge(stats)
}

func (o *idfOracle) minusPartitionStatsLocked(partitionID int64, targetVersion int64, stats bm25Stats) {
	if partitionID == 0 || len(stats) == 0 {
		return
	}
	current, ok := o.partitionStats[partitionID]
	if !ok {
		return
	}
	current.targetVersion = targetVersion
	current.stats.Minus(stats)
}

func (o *idfOracle) clearPartitionStatsLocked(partitionIDs ...int64) {
	if len(partitionIDs) == 0 {
		o.partitionStats = make(map[int64]*partitionBm25Stats)
		return
	}
	for _, partitionID := range partitionIDs {
		delete(o.partitionStats, partitionID)
	}
}

// memSize estimates total in-memory size of current + all growing stats.
// Caller must hold RLock or Lock.
func (o *idfOracle) memSize() int64 {
	size := o.current.MemSize()
	for _, g := range o.growing {
		size += g.bm25Stats.MemSize()
	}
	for _, stats := range o.partitionStats {
		size += stats.stats.MemSize()
	}
	return size
}

// MemorySize returns the estimated in-memory size with RLock protection.
func (o *idfOracle) MemorySize() int64 {
	o.RLock()
	defer o.RUnlock()
	return o.memSize()
}

// diskSize returns total disk size of all sealed segment local files.
func (o *idfOracle) diskSize() int64 {
	return o.sealedDiskSize.Load()
}

// resourceTrackingEnabled reports whether to charge/refund the C++ caching layer.
// When tiered storage eviction is disabled, the caching layer's resource accounting is
// inert (no eviction will be driven by it), so we skip the cgo calls entirely.
func resourceTrackingEnabled() bool {
	return paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool()
}

// syncResource precisely syncs resource usage to the caching layer.
// Used for segment lifecycle events (Register/Unregister/SyncDistribution).
// Caller must NOT hold the RWMutex.
func (o *idfOracle) syncResource() {
	if !resourceTrackingEnabled() {
		return
	}
	actualMem := o.MemorySize()
	actualDisk := o.diskSize()

	o.resourceMu.Lock()
	defer o.resourceMu.Unlock()
	o.doSyncResource(actualMem, actualDisk)
}

// checkMemoryResource checks if memory usage exceeds charged amount.
// Only charges (with headroom), never refunds. Used in Insert path (UpdateGrowing).
// Caller must hold RWMutex.Lock (so memSize is safe to call without RLock).
func (o *idfOracle) checkMemoryResource() {
	if !resourceTrackingEnabled() {
		return
	}
	actualMem := o.memSize()

	o.resourceMu.Lock()
	defer o.resourceMu.Unlock()

	if actualMem > o.chargedMemory {
		charge := actualMem + memoryHeadroom - o.chargedMemory
		C.ChargeLoadedResource(C.CResourceUsage{
			memory_bytes: C.int64_t(charge),
			disk_bytes:   0,
		})
		o.chargedMemory = actualMem + memoryHeadroom
	}
}

// doSyncResource performs the actual Charge/Refund. Caller must hold resourceMu.
func (o *idfOracle) doSyncResource(actualMem, actualDisk int64) {
	memDelta := actualMem - o.chargedMemory
	diskDelta := actualDisk - o.chargedDisk

	if memDelta > 0 || diskDelta > 0 {
		C.ChargeLoadedResource(C.CResourceUsage{
			memory_bytes: C.int64_t(max(memDelta, 0)),
			disk_bytes:   C.int64_t(max(diskDelta, 0)),
		})
	}
	if memDelta < 0 || diskDelta < 0 {
		C.RefundLoadedResource(C.CResourceUsage{
			memory_bytes: C.int64_t(max(-memDelta, 0)),
			disk_bytes:   C.int64_t(max(-diskDelta, 0)),
		})
	}

	o.chargedMemory = actualMem
	o.chargedDisk = actualDisk
}

func (o *idfOracle) Start() {
	o.wg.Add(1)
	go o.syncloop()
}

func (o *idfOracle) Close() {
	close(o.closeCh)
	o.wg.Wait()

	// Refund all charged resources
	o.resourceMu.Lock()
	if o.chargedMemory > 0 || o.chargedDisk > 0 {
		C.RefundLoadedResource(C.CResourceUsage{
			memory_bytes: C.int64_t(o.chargedMemory),
			disk_bytes:   C.int64_t(o.chargedDisk),
		})
		o.chargedMemory = 0
		o.chargedDisk = 0
	}
	o.resourceMu.Unlock()

	if err := os.RemoveAll(o.dirPath); err != nil {
		log.Warn("failed to remove bm25 stats dir on close", zap.Error(err), zap.String("path", o.dirPath))
	}
}

func (o *idfOracle) SetNext(snapshot *snapshot) {
	o.next.SetSnapshot(snapshot)

	// sync SyncDistibution when first load target
	if o.targetVersion.Load() == 0 {
		o.SyncDistribution()
	} else {
		o.NotifySync()
	}
}

func (o *idfOracle) NotifySync() {
	select {
	case o.syncNotify <- struct{}{}:
	default:
	}
}

func (o *idfOracle) syncloop() {
	defer o.wg.Done()
	for {
		select {
		case <-o.syncNotify:
			err := o.SyncDistribution()
			if err != nil {
				log.Warn("idf oracle sync distribution failed", zap.Error(err))
				time.Sleep(time.Second * 10)
				o.NotifySync()
			}
		case <-o.closeCh:
			return
		}
	}
}

// WARN: SyncDistribution not concurrent safe.
// SyncDistribution sync current target to idf oracle.
func (o *idfOracle) SyncDistribution() error {
	snapshot, snapshotTs := o.next.GetSnapshot()
	if snapshot.targetVersion <= o.targetVersion.Load() {
		return nil
	}

	sealed, _ := snapshot.Peek()

	// intarget segment map
	targetMap := typeutil.NewSet[UniqueID]()
	// segment with unreadable target version was not been used,
	// not remove them till it update version or remove from snapshot(released)
	reserveMap := typeutil.NewSet[UniqueID]()

	for _, item := range sealed {
		for _, segment := range item.Segments {
			if segment.Level == datapb.SegmentLevel_L0 {
				continue
			}

			switch segment.TargetVersion {
			case snapshot.targetVersion:
				targetMap.Insert(segment.SegmentID)
				if !o.sealed.Contain(segment.SegmentID) {
					log.Warn("idf oracle lack some sealed segment", zap.Int64("segment", segment.SegmentID))
				}
			case unreadableTargetVersion:
				reserveMap.Insert(segment.SegmentID)
			}
		}
	}

	partitionLevel, partitionLazyLoad, err := idfPartitionLoadConfig()
	if err != nil {
		return err
	}
	if partitionLevel {
		return o.syncDistributionPartitionLevel(snapshot, snapshotTs, targetMap, reserveMap, partitionLazyLoad)
	}

	diff := bm25Stats{}

	var rangeErr error
	o.sealed.Range(func(segmentID int64, stats *sealedBm25Stats) bool {
		intarget := targetMap.Contain(segmentID)

		activate := stats.activate.Load()
		// activate segment if segment in target
		if intarget && !activate {
			stats, err := stats.FetchStats()
			if err != nil {
				rangeErr = fmt.Errorf("fetch stats failed with error: %v", err)
				return false
			}
			diff.Merge(stats)
		} else
		// deactivate segment if segment not in target.
		if !intarget && activate {
			stats, err := stats.FetchStats()
			if err != nil {
				rangeErr = fmt.Errorf("fetch stats failed with error: %v", err)
				return false
			}
			diff.Minus(stats)
		}
		return true
	})

	if rangeErr != nil {
		return rangeErr
	}

	o.Lock()

	for segmentID, stats := range o.growing {
		// drop growing segment bm25 stats
		if stats.droppedVersion != 0 && stats.droppedVersion <= snapshot.targetVersion {
			if stats.activate {
				o.current.Minus(stats.bm25Stats)
			}
			delete(o.growing, segmentID)
		}
	}
	o.current.Merge(diff)

	// remove sealed segment not in target
	o.sealed.Range(func(segmentID int64, stats *sealedBm25Stats) bool {
		reserve := reserveMap.Contain(segmentID)
		intarget := targetMap.Contain(segmentID)

		activate := stats.activate.Load()
		// save activate if segment in target.
		if intarget && !activate {
			stats.activate.Store(true)
		}

		// deactivate if segment not in target.
		if !intarget && activate {
			stats.activate.Store(false)
		}

		// remove
		// if segment not in target and not in reserve list
		// (means segment target version was old version or segment not in snapshot)
		// and add before snapshot Ts
		// (forbid remove some new segment register after current snapshot)
		if !intarget && !reserve && stats.ts.Before(snapshotTs) {
			o.sealedDiskSize.Add(-stats.diskSize)
			stats.Remove()
			o.sealed.Remove(segmentID)
		}
		return true
	})

	o.targetVersion.Store(snapshot.targetVersion)
	numRow := o.current.NumRow()
	growingLen := len(o.growing)
	sealedLen := o.sealed.Len()
	o.Unlock()

	o.syncResource()
	log.Ctx(context.TODO()).Info("sync idf distribution finished", zap.Int64("version", snapshot.targetVersion), zap.Int64("numrow", numRow), zap.Int("growing", growingLen), zap.Int("sealed", sealedLen))
	return nil
}

func (o *idfOracle) syncDistributionPartitionLevel(snapshot *snapshot, snapshotTs time.Time, targetMap, reserveMap typeutil.Set[UniqueID], partitionLazyLoad bool) error {
	if !partitionLazyLoad {
		return o.syncDistributionPartitionLevelLoaded(snapshot, snapshotTs, targetMap, reserveMap)
	}

	o.Lock()

	for segmentID, stats := range o.growing {
		if stats.droppedVersion != 0 && stats.droppedVersion <= snapshot.targetVersion {
			if stats.activate {
				o.current.Minus(stats.bm25Stats)
			}
			delete(o.growing, segmentID)
			o.clearPartitionStatsLocked(stats.partitionID)
		}
	}

	o.sealed.Range(func(segmentID int64, stats *sealedBm25Stats) bool {
		reserve := reserveMap.Contain(segmentID)
		intarget := targetMap.Contain(segmentID)

		activate := stats.activate.Load()
		if intarget && !activate {
			stats.activate.Store(true)
		}
		if !intarget && activate {
			stats.activate.Store(false)
		}

		if !intarget && !reserve && stats.ts.Before(snapshotTs) {
			o.sealedDiskSize.Add(-stats.diskSize)
			stats.Remove()
			o.sealed.Remove(segmentID)
			o.clearPartitionStatsLocked(stats.partitionID)
		}
		return true
	})

	o.clearPartitionStatsLocked()
	o.targetVersion.Store(snapshot.targetVersion)
	numRow := o.current.NumRow()
	growingLen := len(o.growing)
	sealedLen := o.sealed.Len()
	o.Unlock()

	o.syncResource()
	log.Ctx(context.TODO()).Info("sync idf distribution finished",
		zap.Int64("version", snapshot.targetVersion),
		zap.Int64("numrow", numRow),
		zap.Int("growing", growingLen),
		zap.Int("sealed", sealedLen),
		zap.Bool("partitionLevel", true))
	return nil
}

func (o *idfOracle) syncDistributionPartitionLevelLoaded(snapshot *snapshot, snapshotTs time.Time, targetMap, reserveMap typeutil.Set[UniqueID]) error {
	targetPartition := make(map[int64]int64)
	sealed, _ := snapshot.Peek()
	for _, item := range sealed {
		for _, segment := range item.Segments {
			if segment.Level == datapb.SegmentLevel_L0 || segment.TargetVersion != snapshot.targetVersion {
				continue
			}
			targetPartition[segment.SegmentID] = segment.PartitionID
		}
	}

	partitionAdds := make(map[int64]bm25Stats)
	partitionRemoves := make(map[int64]bm25Stats)
	mergeDiff := func(diffs map[int64]bm25Stats, partitionID int64, stats bm25Stats) {
		if partitionID == 0 {
			return
		}
		diff, ok := diffs[partitionID]
		if !ok {
			diff = o.emptyStats()
			diffs[partitionID] = diff
		}
		diff.Merge(stats)
	}

	var rangeErr error
	o.sealed.Range(func(segmentID int64, stats *sealedBm25Stats) bool {
		intarget := targetMap.Contain(segmentID)
		activate := stats.activate.Load()
		if intarget && !activate {
			bm25Stats, err := stats.FetchStats()
			if err != nil {
				rangeErr = errors.Wrapf(err, "fetch bm25 stats for segment %d", segmentID)
				return false
			}
			partitionID := stats.partitionID
			if id, ok := targetPartition[segmentID]; ok {
				partitionID = id
			}
			mergeDiff(partitionAdds, partitionID, bm25Stats)
		}
		if !intarget && activate {
			bm25Stats, err := stats.FetchStats()
			if err != nil {
				rangeErr = errors.Wrapf(err, "fetch bm25 stats for segment %d", segmentID)
				return false
			}
			mergeDiff(partitionRemoves, stats.partitionID, bm25Stats)
		}
		return true
	})
	if rangeErr != nil {
		return rangeErr
	}

	o.Lock()

	for segmentID, stats := range o.growing {
		if stats.droppedVersion != 0 && stats.droppedVersion <= snapshot.targetVersion {
			if stats.activate {
				o.current.Minus(stats.bm25Stats)
				o.minusPartitionStatsLocked(stats.partitionID, snapshot.targetVersion, stats.bm25Stats)
			}
			delete(o.growing, segmentID)
		}
	}

	for partitionID, stats := range partitionAdds {
		o.mergePartitionStatsLocked(partitionID, snapshot.targetVersion, stats)
	}
	for partitionID, stats := range partitionRemoves {
		o.minusPartitionStatsLocked(partitionID, snapshot.targetVersion, stats)
	}

	o.sealed.Range(func(segmentID int64, stats *sealedBm25Stats) bool {
		reserve := reserveMap.Contain(segmentID)
		intarget := targetMap.Contain(segmentID)

		activate := stats.activate.Load()
		if intarget && !activate {
			stats.activate.Store(true)
		}
		if !intarget && activate {
			stats.activate.Store(false)
		}

		if !intarget && !reserve && stats.ts.Before(snapshotTs) {
			o.sealedDiskSize.Add(-stats.diskSize)
			stats.Remove()
			o.sealed.Remove(segmentID)
		}
		return true
	})

	for _, stats := range o.partitionStats {
		stats.targetVersion = snapshot.targetVersion
	}
	o.targetVersion.Store(snapshot.targetVersion)
	numRow := int64(0)
	for _, stats := range o.partitionStats {
		numRow += stats.stats.NumRow()
	}
	growingLen := len(o.growing)
	sealedLen := o.sealed.Len()
	o.Unlock()

	o.syncResource()
	log.Ctx(context.TODO()).Info("sync idf distribution finished",
		zap.Int64("version", snapshot.targetVersion),
		zap.Int64("numrow", numRow),
		zap.Int("growing", growingLen),
		zap.Int("sealed", sealedLen),
		zap.Bool("partitionLevel", true),
		zap.Bool("partitionLazyLoad", false))
	return nil
}

func (o *idfOracle) BuildIDF(ctx context.Context, fieldID int64, tfs *schemapb.SparseFloatArray, partitions ...int64) ([][]byte, float64, error) {
	partitionLevel, partitionLazyLoad, err := idfPartitionLoadConfig()
	if err != nil {
		return nil, 0, err
	}
	if partitionLevel {
		partitionStats, err := o.getPartitionStats(ctx, partitionLazyLoad, partitions...)
		if err != nil {
			return nil, 0, err
		}
		stats, err := partitionStats.GetStats(fieldID)
		if err != nil {
			return nil, 0, err
		}
		return buildIDFWithStats(stats, tfs), stats.GetAvgdl(), nil
	}

	o.RLock()
	defer o.RUnlock()

	stats, err := o.current.GetStats(fieldID)
	if err != nil {
		return nil, 0, err
	}
	return buildIDFWithStats(stats, tfs), stats.GetAvgdl(), nil
}

func buildIDFWithStats(stats *storage.BM25Stats, tfs *schemapb.SparseFloatArray) [][]byte {
	idfBytes := make([][]byte, len(tfs.GetContents()))
	for i, tf := range tfs.GetContents() {
		idf := stats.BuildIDF(tf)
		idfBytes[i] = idf
	}
	return idfBytes
}

func (o *idfOracle) getPartitionStats(ctx context.Context, partitionLazyLoad bool, partitions ...int64) (bm25Stats, error) {
	snapshot, _ := o.next.GetSnapshot()
	if snapshot == nil {
		o.RLock()
		stats := o.current.Clone()
		o.RUnlock()
		return stats, nil
	}

	targetVersion := snapshot.targetVersion
	if !partitionLazyLoad {
		return o.getLoadedPartitionStats(snapshot, targetVersion, partitions...)
	}

	if len(partitions) != 1 {
		return o.buildPartitionStats(ctx, snapshot, targetVersion, partitions...)
	}

	partitionID := partitions[0]
	o.RLock()
	cached, ok := o.partitionStats[partitionID]
	if ok && cached.targetVersion == targetVersion {
		stats := cached.stats
		o.RUnlock()
		return stats, nil
	}
	o.RUnlock()

	_, err, _ := o.sf.Do(fmt.Sprintf("load_partition_bm25_%d_%d", partitionID, targetVersion), func() (any, error) {
		stats, err := o.buildPartitionStats(ctx, snapshot, targetVersion, partitionID)
		if err != nil {
			return nil, err
		}

		o.Lock()
		o.partitionStats[partitionID] = &partitionBm25Stats{
			targetVersion: targetVersion,
			stats:         stats,
		}
		o.Unlock()
		o.syncResource()
		return nil, nil
	})
	if err != nil {
		return nil, err
	}

	o.RLock()
	cached, ok = o.partitionStats[partitionID]
	if ok && cached.targetVersion == targetVersion {
		stats := cached.stats
		o.RUnlock()
		return stats, nil
	}
	o.RUnlock()

	return nil, errors.Newf("partition bm25 stats missing after load, partitionID=%d, targetVersion=%d", partitionID, targetVersion)
}

func (o *idfOracle) getLoadedPartitionStats(snapshot *snapshot, targetVersion int64, partitions ...int64) (bm25Stats, error) {
	aggregate := o.emptyStats()
	sealed, growing := snapshot.Peek(partitions...)

	readable := func(entry SegmentEntry) bool {
		return entry.Level != datapb.SegmentLevel_L0 && (entry.TargetVersion == targetVersion || entry.TargetVersion == initialTargetVersion)
	}

	partitionIDs := typeutil.NewSet[int64]()
	for _, item := range sealed {
		for _, entry := range item.Segments {
			if readable(entry) {
				partitionIDs.Insert(entry.PartitionID)
			}
		}
	}
	for _, entry := range growing {
		if readable(entry) {
			partitionIDs.Insert(entry.PartitionID)
		}
	}

	o.RLock()
	defer o.RUnlock()
	var missing []int64
	for partitionID := range partitionIDs {
		stats, ok := o.partitionStats[partitionID]
		if !ok || stats.targetVersion != targetVersion {
			missing = append(missing, partitionID)
			continue
		}
		aggregate.Merge(stats.stats)
	}
	if len(missing) > 0 {
		return nil, errors.Newf("partition bm25 stats missing after sync, partitions=%v, targetVersion=%d", missing, targetVersion)
	}
	return aggregate, nil
}

func (o *idfOracle) buildPartitionStats(ctx context.Context, snapshot *snapshot, targetVersion int64, partitions ...int64) (bm25Stats, error) {
	aggregate := o.emptyStats()
	sealed, growing := snapshot.Peek(partitions...)

	readable := func(entry SegmentEntry) bool {
		return entry.Level != datapb.SegmentLevel_L0 && (entry.TargetVersion == targetVersion || entry.TargetVersion == initialTargetVersion)
	}

	group, ctx := errgroup.WithContext(ctx)
	group.SetLimit(runtime.GOMAXPROCS(0))
	var mu sync.Mutex

	for _, item := range sealed {
		for _, entry := range item.Segments {
			if !readable(entry) {
				continue
			}

			entry := entry
			group.Go(func() error {
				sealedStats, ok := o.sealed.Get(entry.SegmentID)
				if !ok {
					log.Warn("idf oracle lacks sealed segment bm25 stats",
						zap.Int64("segmentID", entry.SegmentID),
						zap.Int64("partitionID", entry.PartitionID))
					return nil
				}

				stats, err := sealedStats.FetchStats(ctx)
				if err != nil {
					return errors.Wrapf(err, "fetch bm25 stats for segment %d", entry.SegmentID)
				}

				mu.Lock()
				aggregate.Merge(stats)
				mu.Unlock()
				return nil
			})
		}
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}

	o.RLock()
	defer o.RUnlock()
	for _, entry := range growing {
		if !readable(entry) {
			continue
		}
		stats, ok := o.growing[entry.SegmentID]
		if !ok || !stats.activate {
			continue
		}
		aggregate.Merge(stats.bm25Stats)
	}
	return aggregate, nil
}

func (o *idfOracle) emptyStats() bm25Stats {
	o.RLock()
	defer o.RUnlock()
	return o.emptyStatsLocked()
}

func (o *idfOracle) emptyStatsLocked() bm25Stats {
	result := make(bm25Stats, len(o.current))
	for fieldID := range o.current {
		result[fieldID] = storage.NewBM25Stats()
	}
	return result
}

func bm25FieldList(paths map[int64][]string) []int64 {
	fieldList := make([]int64, 0, len(paths))
	for fieldID := range paths {
		fieldList = append(fieldList, fieldID)
	}
	return fieldList
}

func NewIDFOracle(channel string, functions []*schemapb.FunctionSchema) IDFOracle {
	return &idfOracle{
		channel:        channel,
		targetVersion:  atomic.NewInt64(0),
		current:        newBm25Stats(functions),
		growing:        make(map[int64]*growingBm25Stats),
		partitionStats: make(map[int64]*partitionBm25Stats),
		sealed:         typeutil.ConcurrentMap[int64, *sealedBm25Stats]{},
		sealedDiskSize: atomic.NewInt64(0),
		dirPath:        path.Join(pathutil.GetPath(pathutil.BM25Path, paramtable.GetNodeID()), channel),
		syncNotify:     make(chan struct{}, 1),
		closeCh:        make(chan struct{}),
		sf:             conc.Singleflight[any]{},
	}
}
