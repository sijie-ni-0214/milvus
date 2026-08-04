// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dml

import (
	"time"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const (
	proxyInsertStagePreExecuteTotal      = "pre_execute_total"
	proxyInsertStagePreGetCollectionID   = "pre_get_collection_id"
	proxyInsertStagePreGetCollectionInfo = "pre_get_collection_info"
	proxyInsertStagePreGenFunctionFields = "pre_gen_function_fields"
	proxyInsertStagePreAllocAutoID       = "pre_alloc_auto_id"
	proxyInsertStagePreCheckPrimaryField = "pre_check_primary_field"
	proxyInsertStagePreValidateFields    = "pre_validate_fields"
	proxyInsertStageExecuteGetVChannels  = "execute_get_vchannels"
	proxyInsertStageExecuteRepack        = "execute_repack"
	proxyInsertStageExecuteWALAppend     = "execute_wal_append"
)

func observeProxyInsertStage(stage string, start time.Time, err error) {
	status := metrics.SuccessLabel
	if err != nil {
		status = metrics.FailLabel
	}
	metrics.ProxyInsertStageLatency.
		WithLabelValues(paramtable.GetStringNodeID(), stage, status).
		Observe(float64(time.Since(start).Microseconds()) / 1000.0)
}
