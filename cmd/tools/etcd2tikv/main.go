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

// etcd2tikv copies the metadata of a Milvus cluster from etcd to TiKV, for a
// switch of metastore.type from etcd to tikv, and verifies the copy.
//
// It copies <etcdRoot>/kv/ and <etcdRoot>/meta/ without <etcdRoot>/meta/session/
// to the same keys under <tikvRoot>. The sessions, the config overlay and the
// Woodpecker metadata stay in etcd. etcd must serve RangeStream (3.7 or later).
//
// Use -revision for a point-in-time rehearsal against a live source. Without
// it, keep catalog writers stopped through copy and verification, since each
// stream reads its own latest revision. Verification differences cause exit 1.
package main

import (
	"context"
	"flag"
	"fmt"
	"maps"
	"os"
	"os/signal"
	"path"
	"slices"
	"strings"
	"syscall"
	"time"

	tikvclient "github.com/tikv/client-go/v2/kv"
	"github.com/tikv/client-go/v2/txnkv"
	"golang.org/x/time/rate"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

var (
	etcdAddrs       = flag.String("etcd", "127.0.0.1:2379", "comma-separated etcd endpoints")
	revision        = flag.Int64("revision", 0, "read all source keys at this etcd revision (0: latest per stream; requires quiescent metadata for a consistent copy)")
	etcdRoot        = flag.String("etcdRoot", "", "etcd.rootPath of the cluster")
	pdAddr          = flag.String("pd", "", "PD endpoint of the TiKV cluster")
	tikvRoot        = flag.String("tikvRoot", "", "tikv.rootPath the cluster will use (default: etcdRoot)")
	mode            = flag.String("mode", "copy", "split: write source ranges only; copy: copy then verify; verify: compare only")
	overwrite       = flag.Bool("overwrite", false, "delete the target keys in TiKV before the copy")
	ranges          = flag.Int("ranges", 32, "key ranges read concurrently")
	writers         = flag.Int("writers", 32, "concurrent TiKV write transactions")
	batch           = flag.Int("batch", 1024, "keys per TiKV write transaction")
	readRate        = flag.Int("rate", 0, "keys read from etcd per second, 0 for no limit")
	preSplit        = flag.Bool("preSplit", false, "pre-split and scatter TiKV regions before copying")
	checkAllocators = flag.Bool("checkAllocators", false, "require the fork's two persisted gid allocator timestamps")
	rangesFile      = flag.String("rangesFile", "", "file to save the key ranges to, or to read them from when it exists")
)

func main() {
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	cfg := &config{
		revision: *revision,
		etcdRoot: strings.TrimRight(*etcdRoot, "/"),
		tikvRoot: strings.TrimRight(*tikvRoot, "/"),
		ranges:   *ranges,
		writers:  *writers,
		batch:    *batch,
	}
	if *tikvRoot == "" {
		cfg.tikvRoot = cfg.etcdRoot
	}
	if !validRoot(cfg.etcdRoot) || !validRoot(cfg.tikvRoot) || *ranges < 1 || *writers < 1 || *batch < 1 || *readRate < 0 || cfg.revision < 0 ||
		(*mode != "copy" && *mode != "verify" && *mode != "split") ||
		(*mode == "split" && *rangesFile == "") || (*mode != "split" && *pdAddr == "") ||
		(*mode != "copy" && (*overwrite || *preSplit)) {
		flag.Usage()
		os.Exit(2)
	}
	if *readRate > 0 {
		cfg.limiter = rate.NewLimiter(rate.Limit(*readRate), min(*readRate, 1024))
	}

	if err := run(ctx, cfg); err != nil {
		mlog.Error(ctx, "etcd2tikv failed", mlog.Err(err))
		os.Exit(1)
	}
}

func run(ctx context.Context, cfg *config) error {
	cli, err := etcd.GetRemoteEtcdClient(strings.Split(*etcdAddrs, ","))
	if err != nil {
		return merr.Wrap(err, "failed to connect to etcd")
	}
	defer cli.Close()
	if err := checkSource(ctx, cli, cfg, *checkAllocators); err != nil {
		return merr.Wrap(err, "source preflight failed")
	}
	if *mode == "split" {
		if _, err := splitAndSave(ctx, cli, cfg, *rangesFile); err != nil {
			return merr.Wrap(err, "failed to create source ranges")
		}
		return nil
	}
	keyRanges, err := loadOrSplit(ctx, cli, cfg, *rangesFile)
	if err != nil {
		return merr.Wrap(err, "failed to split the source keys")
	}
	// This setting is process-wide in client-go. Only the standalone copier
	// changes it, so a sorted single-region batch can use one prewrite RPC.
	tikvclient.TxnCommitBatchSize.Store(2 * maxBatchBytes)
	store, err := txnkv.NewClient(strings.Split(*pdAddr, ","))
	if err != nil {
		return merr.Wrap(err, "failed to connect to TiKV")
	}
	defer store.Close()

	if *mode == "copy" {
		if err := prepareTarget(ctx, store, cfg, *overwrite); err != nil {
			return merr.Wrap(err, "failed to prepare the TiKV keys")
		}
		if *preSplit {
			if err := splitRegions(ctx, store, cfg, keyRanges); err != nil {
				return merr.Wrap(err, "target region preparation failed")
			}
		}
		result := &copyResult{}
		stop := report(ctx, "copied", &result.keys)
		start := time.Now()
		err := copyKeys(ctx, cli, store, cfg, keyRanges, result)
		stop()
		if err != nil {
			return merr.Wrapf(err, "copy failed after %d committed keys", result.keys.Load())
		}
		fmt.Printf("copied %d keys, %d value bytes, in %s\n", result.keys.Load(), result.bytes.Load(), time.Since(start).Round(time.Millisecond))
	}

	result := &verifyResult{}
	stop := report(ctx, "verified", &result.compared)
	start := time.Now()
	err = verifyKeys(ctx, cli, store, cfg, keyRanges, result)
	stop()
	if err != nil {
		return merr.Wrap(err, "verify failed")
	}
	fmt.Printf("verified %d keys in %s: %d missing in TiKV, %d extra in TiKV, %d different\n",
		result.compared.Load(), time.Since(start).Round(time.Millisecond), result.missing.Load(), result.extra.Load(), result.different.Load())
	for _, name := range slices.Sorted(maps.Keys(result.families)) {
		fmt.Printf("  %-48s %d\n", name, result.families[name])
	}
	for _, example := range result.examples {
		fmt.Println(example)
	}
	if result.failed() {
		return merr.WrapErrDataIntegrityMsg("TiKV differs from etcd")
	}
	return nil
}

// report logs the progress of a pass every 10 seconds until the returned stop
// is called.
func report(ctx context.Context, what string, done interface{ Load() int64 }) (stop func()) {
	ctx, cancel := context.WithCancel(ctx)
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		start, last := time.Now(), int64(0)
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				n := done.Load()
				mlog.Info(ctx, "progress", mlog.String("pass", what), mlog.Int64("keys", n),
					mlog.Int64("keysPerSecond", (n-last)/10), mlog.Duration("elapsed", time.Since(start).Round(time.Millisecond)))
				last = n
			}
		}
	}()
	return func() {
		cancel()
		<-finished
	}
}

func validRoot(root string) bool {
	return root != "" && root != "." && root != ".." && !strings.HasPrefix(root, "../") && path.Clean(root) == root
}
