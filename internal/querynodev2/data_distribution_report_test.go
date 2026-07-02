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

package querynodev2

import (
	"testing"
)

func TestDataDistributionReportDeltaTrackerChannelDeltaIsExclusive(t *testing.T) {
	cache := newDataDistributionDeltaTracker()

	cache.markChannelUpsert("ch-1")
	if _, ok := cache.dirtyChannels["ch-1"]; !ok {
		t.Fatalf("expected channel upsert to be dirty")
	}
	if _, ok := cache.removedChannels["ch-1"]; ok {
		t.Fatalf("expected channel upsert to clear removed marker")
	}

	cache.markChannelRemove("ch-1")
	if _, ok := cache.dirtyChannels["ch-1"]; ok {
		t.Fatalf("expected channel remove to clear dirty marker")
	}
	if _, ok := cache.removedChannels["ch-1"]; !ok {
		t.Fatalf("expected channel remove to be recorded")
	}

	cache.markChannelUpsert("ch-1")
	if _, ok := cache.dirtyChannels["ch-1"]; !ok {
		t.Fatalf("expected channel re-upsert to be dirty")
	}
	if _, ok := cache.removedChannels["ch-1"]; ok {
		t.Fatalf("expected channel re-upsert to clear removed marker")
	}
}
