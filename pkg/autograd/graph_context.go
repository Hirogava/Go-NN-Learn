package autograd

import (
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type GraphContext struct {
	engine      *Engine
	gradEnabled bool
	released    bool
	mu          sync.RWMutex
}

func NewGraph() *GraphContext {
	return &GraphContext{
		engine:      NewEngine(),
		gradEnabled: true,
		released:    false,
	}
}

func (g *GraphContext) WithGrad() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gradEnabled = true
}

func (g *GraphContext) NoGrad() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gradEnabled = false
}

func (g *GraphContext) GradEnabled() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.gradEnabled
}

func (g *GraphContext) Backward(finalNode *graph.Node) {
	g.mu.Lock()
	if g.released {
		g.mu.Unlock()
		panic("autograd: GraphContext already released after Backward")
	}
	g.mu.Unlock()

	g.engine.Backward(finalNode)
	g.release()
}

func (g *GraphContext) release() {
	g.mu.Lock()
	g.engine.Nodes = nil
	g.released = true
	g.mu.Unlock()
	currentGraph.Lock()
	if currentGraph.ctx == g {
		currentGraph.ctx = nil
	}
	currentGraph.Unlock()
}

func (g *GraphContext) RequireGrad(t *tensor.Tensor) *graph.Node {
	return g.engine.RequireGrad(t)
}

func (g *GraphContext) ZeroGrad() {
	g.engine.ZeroGrad()
}

func (g *GraphContext) Engine() *Engine {
	return g.engine
}

var currentGraph struct {
	sync.RWMutex
	ctx *GraphContext
}

func SetGraph(ctx *GraphContext) {
	currentGraph.Lock()
	defer currentGraph.Unlock()
	currentGraph.ctx = ctx
}

func GetGraph() *GraphContext {
	currentGraph.RLock()
	defer currentGraph.RUnlock()
	return currentGraph.ctx
}

func GradEnabled() bool {
	g := GetGraph()
	return g != nil && g.GradEnabled()
}

func ClearGraph() {
	currentGraph.Lock()
	defer currentGraph.Unlock()
	currentGraph.ctx = nil
}
