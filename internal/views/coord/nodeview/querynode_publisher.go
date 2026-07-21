package nodeview

import (
	"context"
	"sync"

	"github.com/milvus-io/milvus/internal/coordinator/snmanager"
	qnmanager "github.com/milvus-io/milvus/internal/querynodev2/client/manager"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/views/coord/balancer/api"
	"github.com/milvus-io/milvus/internal/views/coord/balancer/cache"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/syncutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Both sources must synchronously replay and serialize committed updates with
// registration. Callbacks run at the source commit point, never as a pull worker.
// Discovery can fail or be canceled before its initial state is available.
type QueryNodeStatePublisher interface {
	RegisterQueryNodeListener(context.Context, func(int64, *qnmanager.NodeInfo)) (func(), error)
}
type ResourceGroupStatePublisher interface {
	RegisterResourceGroupListener(func(string, []int64)) func()
}

// StreamingQueryNodeSource reports the resource group of every StreamingNode
// that embeds a QueryNode.
type StreamingQueryNodeSource interface {
	ListenNodeChanged() *syncutil.VersionedListener
	GetStreamingQueryNodeIDsByResourceGroup() map[string]typeutil.UniqueSet
}

type nodeFacts struct {
	label    string
	stopping bool
	embedded bool
}

// QueryNodePublisher joins node facts with RG assignments on publication.
// A session label has precedence; overlapping RG bindings choose the smallest
// name. QueryNodes embedded in StreamingNodes take the RG of their
// StreamingNode. Cache subscribers only read results.
type QueryNodePublisher struct {
	mu           sync.Mutex
	facts        map[int64]nodeFacts
	groups       map[string]map[int64]struct{}
	bindings     map[int64]map[string]struct{}
	sqnGroups    map[int64]string
	published    map[int64]*api.NodeInfo
	listeners    map[uint64]cache.NodeListener
	nextListener uint64
	unsubscribe  []func()
}

func newQueryNodePublisher() *QueryNodePublisher {
	return &QueryNodePublisher{facts: make(map[int64]nodeFacts), groups: make(map[string]map[int64]struct{}), bindings: make(map[int64]map[string]struct{}), sqnGroups: make(map[int64]string), published: make(map[int64]*api.NodeInfo), listeners: make(map[uint64]cache.NodeListener)}
}

func NewQueryNodePublisher(ctx context.Context, nodes QueryNodeStatePublisher, groups ResourceGroupStatePublisher) (*QueryNodePublisher, error) {
	p := newQueryNodePublisher()
	p.unsubscribe = append(p.unsubscribe, groups.RegisterResourceGroupListener(p.onGroup))
	if paramtable.Get().QueryCoordCfg.EnableSQNServeSegments.GetAsBool() {
		p.unsubscribe = append(p.unsubscribe, p.watchStreamingQueryNodes(snmanager.StaticStreamingNodeManager))
	}
	stop, err := nodes.RegisterQueryNodeListener(ctx, p.onNode)
	if err != nil {
		p.Close()
		return nil, err
	}
	p.unsubscribe = append(p.unsubscribe, stop)
	return p, nil
}

func (p *QueryNodePublisher) Close() {
	for _, unsubscribe := range p.unsubscribe {
		unsubscribe()
	}
}

func (p *QueryNodePublisher) RegisterNodeListener(listener cache.NodeListener) func() {
	p.mu.Lock()
	p.nextListener++
	id := p.nextListener
	p.listeners[id] = listener
	for nodeID, node := range p.published {
		listener(nodeID, node)
	}
	p.mu.Unlock()
	return func() { p.mu.Lock(); delete(p.listeners, id); p.mu.Unlock() }
}

func (p *QueryNodePublisher) onNode(id int64, info *qnmanager.NodeInfo) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if info == nil {
		delete(p.facts, id)
	} else {
		p.facts[id] = nodeFacts{
			label:    info.ServerLabels[sessionutil.LabelResourceGroup],
			stopping: info.Stopping,
			embedded: isEmbeddedQueryNode(info.ServerLabels),
		}
	}
	p.publishLocked(id)
}

// watchStreamingQueryNodes keeps the RG of embedded QueryNodes in sync with
// their StreamingNodes. The source may block until the StreamingNode balancer
// is ready, so it is never queried under p.mu or from a source callback.
func (p *QueryNodePublisher) watchStreamingQueryNodes(source StreamingQueryNodeSource) func() {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		listener := source.ListenNodeChanged()
		for ctx.Err() == nil {
			p.onStreamingQueryNodes(source.GetStreamingQueryNodeIDsByResourceGroup())
			if listener.Wait(ctx) != nil {
				return
			}
		}
	}()
	return cancel
}

func (p *QueryNodePublisher) onStreamingQueryNodes(nodesByRG map[string]typeutil.UniqueSet) {
	next := make(map[int64]string)
	for rg, nodeIDs := range nodesByRG {
		for _, nodeID := range nodeIDs.Collect() {
			next[nodeID] = rg
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	prev := p.sqnGroups
	p.sqnGroups = next
	for nodeID, rg := range next {
		if prev[nodeID] != rg && p.facts[nodeID].embedded {
			p.publishLocked(nodeID)
		}
	}
	for nodeID := range prev {
		if _, ok := next[nodeID]; !ok && p.facts[nodeID].embedded {
			p.publishLocked(nodeID)
		}
	}
}

func isEmbeddedQueryNode(labels map[string]string) bool {
	return labels[sessionutil.LabelStreamingNodeEmbeddedQueryNode] == "1" ||
		labels[sessionutil.LegacyLabelStreamingNodeEmbeddedQueryNode] == "1"
}

func (p *QueryNodePublisher) onGroup(name string, nodes []int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	affected := make(map[int64]struct{})
	for nodeID := range p.groups[name] {
		affected[nodeID] = struct{}{}
		delete(p.bindings[nodeID], name)
		if len(p.bindings[nodeID]) == 0 {
			delete(p.bindings, nodeID)
		}
	}
	delete(p.groups, name)
	if len(nodes) > 0 {
		p.groups[name] = make(map[int64]struct{}, len(nodes))
	}
	for _, nodeID := range nodes {
		affected[nodeID] = struct{}{}
		p.groups[name][nodeID] = struct{}{}
		if p.bindings[nodeID] == nil {
			p.bindings[nodeID] = make(map[string]struct{})
		}
		p.bindings[nodeID][name] = struct{}{}
	}
	for nodeID := range affected {
		p.publishLocked(nodeID)
	}
}

func (p *QueryNodePublisher) publishLocked(id int64) {
	facts, exists := p.facts[id]
	rg := facts.label
	if facts.embedded {
		rg = p.sqnGroups[id]
	} else if rg == "" {
		for name := range p.bindings[id] {
			if rg == "" || name < rg {
				rg = name
			}
		}
	}
	var next *api.NodeInfo
	if exists && rg != "" {
		next = &api.NodeInfo{NodeID: id, Alive: true, Stopping: facts.stopping, ResourceGroup: rg}
	}
	old := p.published[id]
	if (old == nil && next == nil) || (old != nil && next != nil && *old == *next) {
		return
	}
	if next == nil {
		delete(p.published, id)
	} else {
		p.published[id] = next
	}
	for _, listener := range p.listeners {
		listener(id, next)
	}
}
