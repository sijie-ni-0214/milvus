package queryview

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/metastore/kv/txn"
	"github.com/milvus-io/milvus/pkg/v3/kv"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const VersionedLoadInfoPrefix = "coord/load_info"

// CollectionLoadInfoCatalog stores both the current intent and historical
// requirements. The owner serializes publications and reference acquisition
// with deletion; the catalog never decides which version is collectible.
type CollectionLoadInfoCatalog interface {
	List(context.Context) ([]*viewpb.CollectionLoadConfig, error)
	Save(context.Context, *viewpb.CollectionLoadConfig) error
	RemoveVersions(context.Context, int64, []uint64) error
	RemoveCollection(context.Context, int64) error
}

type collectionLoadInfoCatalog struct {
	storage  kv.MetaKv
	pageSize int
}

func NewCollectionLoadInfoCatalog(storage kv.MetaKv) CollectionLoadInfoCatalog {
	return &collectionLoadInfoCatalog{
		storage:  kv.NewReliableWriteMetaKv(storage),
		pageSize: paramtable.Get().MetaStoreCfg.PaginationSize.GetAsInt(),
	}
}

func collectionLoadInfoVersionKey(collectionID int64, version uint64) string {
	return fmt.Sprintf("%s/%d/%d", VersionedLoadInfoPrefix, collectionID, version)
}

// List reads only the independent QueryView records. Migration from legacy load
// metadata and broader crash-recovery integration are deliberately deferred.
func (c *collectionLoadInfoCatalog) List(ctx context.Context) ([]*viewpb.CollectionLoadConfig, error) {
	var records []*viewpb.CollectionLoadConfig
	err := c.storage.WalkWithPrefix(ctx, VersionedLoadInfoPrefix+"/", c.pageSize, func(key, value []byte) error {
		info, err := decodeCollectionLoadConfig(value)
		if err != nil {
			return merr.Wrap(err, "decode collection load requirements")
		}
		expected := collectionLoadInfoVersionKey(info.GetCollectionID(), info.GetVersion())
		if info.GetVersion() == 0 || (string(key) != expected && !strings.HasSuffix(string(key), "/"+expected)) {
			return merr.WrapErrDataIntegrityMsg("collection load info identity does not match key %s", key)
		}
		records = append(records, info)
		return nil
	})
	return records, err
}

// Older QueryView builds wrote a warmup epoch (a TSO) at tag 9. Both that
// integer and today's auto_load boolean use varint encoding. Only the boolean
// writer emits 1 for true. Preserve protobuf's last-value-wins behavior when
// reading duplicate tags, and skip nested messages and unknown fields intact.
func decodeCollectionLoadConfig(data []byte) (*viewpb.CollectionLoadConfig, error) {
	info := new(viewpb.CollectionLoadConfig)
	if err := proto.Unmarshal(data, info); err != nil {
		return nil, err
	}
	if !info.AutoLoad {
		return info, nil
	}
	for len(data) > 0 {
		num, typ, tagSize := protowire.ConsumeTag(data)
		if tagSize < 0 {
			return nil, protowire.ParseError(tagSize)
		}
		data = data[tagSize:]
		var size int
		if num == 9 && typ == protowire.VarintType {
			var value uint64
			value, size = protowire.ConsumeVarint(data)
			info.AutoLoad = value == 1
		} else {
			size = protowire.ConsumeFieldValue(num, typ, data)
		}
		if size < 0 {
			return nil, protowire.ParseError(size)
		}
		data = data[size:]
	}
	return info, nil
}

// Save publishes all requirements, including partition and replica membership,
// in one key. No legacy CollectionLoadInfo, PartitionLoadInfo or Replica is written.
func (c *collectionLoadInfoCatalog) Save(ctx context.Context, info *viewpb.CollectionLoadConfig) error {
	if info.GetVersion() == 0 {
		return merr.WrapErrServiceInternalMsg("cannot persist collection load info without a version")
	}
	value, err := proto.Marshal(info)
	if err != nil {
		return merr.Wrap(err, "encode collection load info")
	}
	return c.storage.Save(ctx, collectionLoadInfoVersionKey(info.CollectionID, info.Version), string(value))
}

func (c *collectionLoadInfoCatalog) RemoveVersions(ctx context.Context, collectionID int64, versions []uint64) error {
	batch := txn.New()
	for _, version := range versions {
		batch.Remove(collectionLoadInfoVersionKey(collectionID, version))
	}
	return txn.Commit(ctx, c.storage, batch)
}

func (c *collectionLoadInfoCatalog) RemoveCollection(ctx context.Context, collectionID int64) error {
	return c.storage.RemoveWithPrefix(ctx, fmt.Sprintf("%s/%d/", VersionedLoadInfoPrefix, collectionID))
}
