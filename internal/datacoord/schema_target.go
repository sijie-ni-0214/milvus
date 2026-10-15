package datacoord

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

func coordinatorPreparationSchema(ctx context.Context, collectionID int64, published *schemapb.CollectionSchema) (*schemapb.CollectionSchema, error) {
	/*
		return qvresource.PreparationSchema(ctx, collectionID, func(context.Context, int64) (*schemapb.CollectionSchema, error) {
			return published, nil
		})
	*/
	return published, nil
}
