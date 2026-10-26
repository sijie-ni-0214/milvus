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

// Package datacoord contains core functions in datacoord
package meta

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/require"
)

func TestStartManifestIndexRecovery(t *testing.T) {
	t.Run("wait returns the recovery result", func(t *testing.T) {
		injected := errors.New("injected")
		mocker := mockey.Mock((*Meta).ReloadSegmentIndexesFromManifests).Return(injected).Build()
		defer mocker.UnPatch()

		recovery := StartManifestIndexRecovery(context.Background(), &Meta{})
		require.ErrorIs(t, recovery.Wait(), injected)
		recovery.Abort()
		require.ErrorIs(t, recovery.Wait(), injected)
	})

	t.Run("abort cancels a running recovery and joins it", func(t *testing.T) {
		started := make(chan struct{})
		stopped := false
		mocker := mockey.Mock((*Meta).ReloadSegmentIndexesFromManifests).To(func(_ *Meta, ctx context.Context) error {
			close(started)
			<-ctx.Done()
			stopped = true
			return ctx.Err()
		}).Build()
		defer mocker.UnPatch()

		recovery := StartManifestIndexRecovery(context.Background(), &Meta{})
		<-started
		recovery.Abort()
		require.True(t, stopped)
		require.ErrorIs(t, recovery.Wait(), context.Canceled)
	})
}
