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

import (
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDistributionAddBatcherConcurrentAdd(t *testing.T) {
	batcher := newDistributionAddBatcher()
	const requestCount = distributionAddBatchSize * 2

	var mut sync.Mutex
	added := make(map[int64]struct{}, requestCount)
	addFunc := func(entries ...SegmentEntry) error {
		mut.Lock()
		defer mut.Unlock()
		for _, entry := range entries {
			added[entry.SegmentID] = struct{}{}
		}
		return nil
	}

	var wg sync.WaitGroup
	errCh := make(chan error, requestCount)
	for i := 0; i < requestCount; i++ {
		wg.Add(1)
		go func(segmentID int64) {
			defer wg.Done()
			err := batcher.add([]SegmentEntry{{SegmentID: segmentID}}, addFunc)
			errCh <- err
		}(int64(i))
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		require.NoError(t, err)
	}
	require.Len(t, added, requestCount)
	for i := 0; i < requestCount; i++ {
		_, ok := added[int64(i)]
		require.True(t, ok)
	}
}

func TestDistributionAddBatcherReturnsAddError(t *testing.T) {
	batcher := newDistributionAddBatcher()
	expectedErr := errors.New("add failed")

	err := batcher.add([]SegmentEntry{{SegmentID: 1}}, func(...SegmentEntry) error {
		return expectedErr
	})
	require.ErrorIs(t, err, expectedErr)
}
