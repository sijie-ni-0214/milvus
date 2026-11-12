#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Regenerate pkg/util/merr/segcore_codes_gen.go from milvus-common's
# enum ErrorCode (the single source of truth for segcore error codes).
#
# Source resolution (first hit wins):
#   1. $MILVUS_COMMON_HEADER (explicit path to EasyAssert.h).
#   2. The source checkout populated by CMake FetchContent.
#   3. A temporary checkout of the repository/ref declared in CMakeLists.txt.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

header="${MILVUS_COMMON_HEADER:-}"
common_source="$ROOT/cmake_build/thirdparty/milvus-common/milvus-common-src"

if [[ -z "$header" && -f "$common_source/include/common/EasyAssert.h" ]]; then
  header="$common_source/include/common/EasyAssert.h"
fi

if [[ -z "$header" ]]; then
  read -r repository ref < <(awk '
    /^FetchContent_Declare\(/ { dependency = "" }
    $1 == "milvus-common" { dependency = $1 }
    dependency && $1 == "GIT_REPOSITORY" { repository = $2 }
    dependency && $1 == "GIT_TAG" { ref = $2 }
    dependency && $1 == ")" { print repository, ref; exit }
  ' "$ROOT/internal/core/CMakeLists.txt")
  checkout="$(mktemp -d)"
  trap 'rm -rf "$checkout"' EXIT
  git clone --quiet --depth 1 --branch "$ref" "$repository" "$checkout"
  header="$checkout/include/common/EasyAssert.h"
fi

if [[ ! -f "$header" ]]; then
  echo "segcoregen: cannot locate milvus-common EasyAssert.h at $header" >&2
  exit 1
fi

header="$(cd "$(dirname "$header")" && pwd)/$(basename "$header")"
echo "segcoregen: using header $header"

cd "$ROOT/pkg"
go run ./util/merr/internal/segcoregen/main.go -header "$header" -out util/merr/segcore_codes_gen.go
gofmt -w util/merr/segcore_codes_gen.go
echo "segcoregen: wrote pkg/util/merr/segcore_codes_gen.go"
