package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// LBFGSOptimizerState represents GPU-resident L-BFGS optimizer state
type LBFGSOptimizerState struct {
	// Configuration
	config LBFGSConfig
	
	// GPU-resident state buffers
	sVectors      [][]unsafe.Pointer // Parameter differences s_k = x_{k+1} - x_k
	yVectors      [][]unsafe.Pointer // Gradient differences y_k = g_{k+1} - g_k
	rhoBuffers    []unsafe.Pointer   // Scalar values ρ_k = 1/(y_k^T s_k)
	alphaBuffer   unsafe.Pointer     // Alpha values for two-loop recursion
	oldGradients  []unsafe.Pointer   // Previous gradients for computing y_k
	searchDir     []unsafe.Pointer   // Search direction p_k
	WeightBuffers []unsafe.Pointer   // Current weight tensors
	
	// History tracking
	currentStep  uint64
	historyCount int  // Current number of stored history pairs
	historyIndex int  // Circular buffer index
	
	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer
	bufferSizes   []int
	
	// Command buffer pooling
	commandPool unsafe.Pointer
	usePooling  bool
	
	// Line search state
	prevLoss     float32
	prevGradNorm float32
}

// LBFGSConfig holds configuration for L-BFGS optimizer
type LBFGSConfig struct {
	HistorySize   int     // m parameter (number of corrections to store)
	LineSearchTol float32 // Tolerance for line search
	MaxLineSearch int     // Maximum line search iterations
	C1            float32 // Armijo condition parameter
	C2            float32 // Wolfe condition parameter
	InitialStep   float32 // Initial step size for line search
}

// DefaultLBFGSConfig returns default L-BFGS optimizer configuration
func DefaultLBFGSConfig() LBFGSConfig {
	return LBFGSConfig{
		HistorySize:   10,
		LineSearchTol: 1e-4,
		MaxLineSearch: 20,
		C1:            1e-4,
		C2:            0.9,
		InitialStep:   1.0,
	}
}

// NewLBFGSOptimizer creates a new GPU-resident L-BFGS optimizer
func NewLBFGSOptimizer(
	config LBFGSConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*LBFGSOptimizerState, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}
	
	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}
	
	if len(weightShapes) == 0 {
		return nil, fmt.Errorf("no weight shapes provided")
	}
	
	if config.HistorySize <= 0 {
		return nil, fmt.Errorf("history size must be positive, got %d", config.HistorySize)
	}
	
	numWeights := len(weightShapes)
	
	lbfgs := &LBFGSOptimizerState{
		config:        config,
		sVectors:      make([][]unsafe.Pointer, config.HistorySize),
		yVectors:      make([][]unsafe.Pointer, config.HistorySize),
		rhoBuffers:    make([]unsafe.Pointer, config.HistorySize),
		oldGradients:  make([]unsafe.Pointer, numWeights),
		searchDir:     make([]unsafe.Pointer, numWeights),
		WeightBuffers: make([]unsafe.Pointer, numWeights),
		currentStep:   0,
		historyCount:  0,
		historyIndex:  0,
		memoryManager: memoryManager,
		device:        device,
		bufferSizes:   make([]int, numWeights),
	}
	
	// Calculate buffer sizes
	for i, shape := range weightShapes {
		lbfgs.bufferSizes[i] = calculateTensorSize(shape) * 4 // 4 bytes per float32
	}
	
	// Allocate history vectors (s and y) for circular buffer
	for h := 0; h < config.HistorySize; h++ {
		lbfgs.sVectors[h] = make([]unsafe.Pointer, numWeights)
		lbfgs.yVectors[h] = make([]unsafe.Pointer, numWeights)
		
		// Allocate buffers for each weight tensor
		for i, size := range lbfgs.bufferSizes {
			// Allocate s vector
			sBuffer := lbfgs.memoryManager.AllocateBuffer(size)
			if sBuffer == nil {
				lbfgs.cleanup()
				return nil, fmt.Errorf("failed to allocate s vector buffer for history %d, weight %d", h, i)
			}
			lbfgs.sVectors[h][i] = sBuffer
			
			// Allocate y vector
			yBuffer := lbfgs.memoryManager.AllocateBuffer(size)
			if yBuffer == nil {
				lbfgs.cleanup()
				return nil, fmt.Errorf("failed to allocate y vector buffer for history %d, weight %d", h, i)
			}
			lbfgs.yVectors[h][i] = yBuffer
			
			// Initialize to zero
			if err := cgo_bridge.ZeroMetalBuffer(lbfgs.device, sBuffer, size); err != nil {
				lbfgs.cleanup()
				return nil, fmt.Errorf("failed to zero s buffer: %v", err)
			}
			if err := cgo_bridge.ZeroMetalBuffer(lbfgs.device, yBuffer, size); err != nil {
				lbfgs.cleanup()
				return nil, fmt.Errorf("failed to zero y buffer: %v", err)
			}
		}
		
		// Allocate rho scalar buffer (single float32)
		rhoBuffer := lbfgs.memoryManager.AllocateBuffer(4)
		if rhoBuffer == nil {
			lbfgs.cleanup()
			return nil, fmt.Errorf("failed to allocate rho buffer for history %d", h)
		}
		lbfgs.rhoBuffers[h] = rhoBuffer
	}
	
	// Allocate alpha buffer for two-loop recursion
	alphaSize := config.HistorySize * 4 // m float32 values
	lbfgs.alphaBuffer = lbfgs.memoryManager.AllocateBuffer(alphaSize)
	if lbfgs.alphaBuffer == nil {
		lbfgs.cleanup()
		return nil, fmt.Errorf("failed to allocate alpha buffer")
	}
	
	// Allocate old gradients and search direction buffers
	for i, size := range lbfgs.bufferSizes {
		// Old gradients
		oldGradBuffer := lbfgs.memoryManager.AllocateBuffer(size)
		if oldGradBuffer == nil {
			lbfgs.cleanup()
			return nil, fmt.Errorf("failed to allocate old gradient buffer for weight %d", i)
		}
		lbfgs.oldGradients[i] = oldGradBuffer
		
		// Search direction
		searchDirBuffer := lbfgs.memoryManager.AllocateBuffer(size)
		if searchDirBuffer == nil {
			lbfgs.cleanup()
			return nil, fmt.Errorf("failed to allocate search direction buffer for weight %d", i)
		}
		lbfgs.searchDir[i] = searchDirBuffer
		
		// Initialize to zero
		if err := cgo_bridge.ZeroMetalBuffer(lbfgs.device, oldGradBuffer, size); err != nil {
			lbfgs.cleanup()
			return nil, fmt.Errorf("failed to zero old gradient buffer: %v", err)
		}
		if err := cgo_bridge.ZeroMetalBuffer(lbfgs.device, searchDirBuffer, size); err != nil {
			lbfgs.cleanup()
			return nil, fmt.Errorf("failed to zero search direction buffer: %v", err)
		}
	}
	
	return lbfgs, nil
}

// cleanup releases all allocated buffers
func (lbfgs *LBFGSOptimizerState) cleanup() {
	// Clean up history vectors
	for h := 0; h < lbfgs.config.HistorySize; h++ {
		if lbfgs.sVectors[h] != nil {
			for i := 0; i < len(lbfgs.WeightBuffers); i++ {
				if lbfgs.sVectors[h][i] != nil {
					lbfgs.memoryManager.ReleaseBuffer(lbfgs.sVectors[h][i])
				}
			}
		}
		if lbfgs.yVectors[h] != nil {
			for i := 0; i < len(lbfgs.WeightBuffers); i++ {
				if lbfgs.yVectors[h][i] != nil {
					lbfgs.memoryManager.ReleaseBuffer(lbfgs.yVectors[h][i])
				}
			}
		}
		if lbfgs.rhoBuffers[h] != nil {
			lbfgs.memoryManager.ReleaseBuffer(lbfgs.rhoBuffers[h])
		}
	}
	
	// Clean up alpha buffer
	if lbfgs.alphaBuffer != nil {
		lbfgs.memoryManager.ReleaseBuffer(lbfgs.alphaBuffer)
	}
	
	// Clean up old gradients and search direction
	for i := 0; i < len(lbfgs.WeightBuffers); i++ {
		if lbfgs.oldGradients[i] != nil {
			lbfgs.memoryManager.ReleaseBuffer(lbfgs.oldGradients[i])
		}
		if lbfgs.searchDir[i] != nil {
			lbfgs.memoryManager.ReleaseBuffer(lbfgs.searchDir[i])
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
func (lbfgs *LBFGSOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(lbfgs.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(lbfgs.WeightBuffers), len(weightBuffers))
	}
	
	copy(lbfgs.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single L-BFGS optimization step
func (lbfgs *LBFGSOptimizerState) Step(gradientBuffers []unsafe.Pointer, currentLoss float32) error {
	if len(gradientBuffers) != len(lbfgs.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(lbfgs.WeightBuffers))
	}
	
	lbfgs.currentStep++
	
	// Execute L-BFGS step using CGO bridge
	stepSize, err := cgo_bridge.ExecuteLBFGSStepMPSGraph(
		lbfgs.device,
		lbfgs.WeightBuffers,
		gradientBuffers,
		lbfgs.oldGradients,
		lbfgs.searchDir,
		lbfgs.sVectors,
		lbfgs.yVectors,
		lbfgs.rhoBuffers,
		lbfgs.alphaBuffer,
		len(lbfgs.WeightBuffers),
		lbfgs.bufferSizes,
		lbfgs.config.HistorySize,
		lbfgs.historyCount,
		lbfgs.historyIndex,
		lbfgs.config.InitialStep,
		lbfgs.config.C1,
		lbfgs.config.C2,
		lbfgs.config.MaxLineSearch,
		currentLoss,
		lbfgs.prevLoss,
		lbfgs.commandPool,
		lbfgs.usePooling,
	)
	
	if err != nil {
		return fmt.Errorf("L-BFGS step failed: %v", err)
	}
	
	// Note: y_k and rho_k computation is now handled entirely in the C code
	// to avoid race conditions and ensure proper buffer management
	
	// Update history tracking
	lbfgs.historyIndex = (lbfgs.historyIndex + 1) % lbfgs.config.HistorySize
	if lbfgs.historyCount < lbfgs.config.HistorySize {
		lbfgs.historyCount++
	}
	
	lbfgs.prevLoss = currentLoss
	
	// Log progress periodically
	if lbfgs.currentStep%10 == 0 {
		fmt.Printf("L-BFGS step %d: loss=%.6f, step_size=%.6f, history=%d/%d\n",
			lbfgs.currentStep, currentLoss, stepSize, lbfgs.historyCount, lbfgs.config.HistorySize)
	}
	
	return nil
}

// SetCommandPool sets the command buffer pool for Metal operations
func (lbfgs *LBFGSOptimizerState) SetCommandPool(pool unsafe.Pointer) {
	lbfgs.commandPool = pool
	lbfgs.usePooling = (pool != nil)
}

// GetStep returns the current optimization step count
func (lbfgs *LBFGSOptimizerState) GetStep() uint64 {
	return lbfgs.currentStep
}

// GetStats returns optimizer statistics
func (lbfgs *LBFGSOptimizerState) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"step":         lbfgs.currentStep,
		"history_size": lbfgs.config.HistorySize,
		"history_used": lbfgs.historyCount,
		"prev_loss":    lbfgs.prevLoss,
	}
}

// Cleanup releases all GPU buffers
func (lbfgs *LBFGSOptimizerState) Cleanup() {
	lbfgs.cleanup()
}

// UpdateLearningRate is not used in L-BFGS (uses line search instead)
func (lbfgs *LBFGSOptimizerState) UpdateLearningRate(newLR float32) error {
	return fmt.Errorf("L-BFGS does not use a fixed learning rate; it uses line search")
}