package tensor

import (
	"fmt"
	"sync/atomic"
)

type DType int

const (
	Float32 DType = iota
	Float16
	Int32
)

func (d DType) String() string {
	switch d {
	case Float32:
		return "Float32"
	case Float16:
		return "Float16"
	case Int32:
		return "Int32"
	default:
		return "Unknown"
	}
}

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

func (d DeviceType) String() string {
	switch d {
	case CPU:
		return "CPU"
	case GPU:
		return "GPU"
	default:
		return "Unknown"
	}
}

type Operation interface {
	Forward(...*Tensor) *Tensor
	Backward(gradOut *Tensor) []*Tensor
}

type Tensor struct {
	Shape        []int
	Strides      []int
	DType        DType
	Device       DeviceType
	Data         interface{}
	NumElems     int
	requiresGrad bool
	grad         *Tensor
	creator      Operation
	
	// GPU memory management fields
	gpuBuffer    interface{} // *metal_bridge.Buffer for GPU tensors
	refCount     int32       // Reference count for GPU tensor lifetime management
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s, elements=%d)", 
		t.Shape, t.DType, t.Device, t.NumElems)
}

func (t *Tensor) RequiresGrad() bool {
	return t.requiresGrad
}

func (t *Tensor) SetRequiresGrad(requires bool) {
	t.requiresGrad = requires
}

func (t *Tensor) Grad() *Tensor {
	return t.grad
}

// ZeroGrad clears the gradient for this tensor
func (t *Tensor) ZeroGrad() {
	t.grad = nil
}

// Backward performs backpropagation from this tensor
func (t *Tensor) Backward() error {
	// This should be called on a scalar loss tensor
	if t.NumElems != 1 {
		return fmt.Errorf("backward() can only be called on scalar tensors, got shape %v", t.Shape)
	}
	
	// Initialize gradient with ones for the loss tensor
	var err error
	t.grad, err = Ones(t.Shape, t.DType, t.Device)
	if err != nil {
		return fmt.Errorf("failed to initialize gradient: %v", err)
	}
	
	// Perform reverse-mode automatic differentiation
	visited := make(map[*Tensor]bool)
	return t.backwardImpl(visited)
}

// backwardImpl recursively computes gradients using topological ordering
func (t *Tensor) backwardImpl(visited map[*Tensor]bool) error {
	if visited[t] {
		return nil // Already visited this tensor
	}
	visited[t] = true
	
	// If this tensor has a creator (was produced by an operation), compute gradients
	if t.creator != nil && t.requiresGrad {
		// Get gradients for inputs from the operation's backward method
		inputGrads := t.creator.Backward(t.grad)
		
		// Get the input tensors from the operation
		// We need to access inputs through reflection or store them in operation
		// For now, we'll implement a simpler approach
		inputs := t.getCreatorInputs()
		
		if len(inputGrads) != len(inputs) {
			return fmt.Errorf("operation returned %d gradients but has %d inputs", len(inputGrads), len(inputs))
		}
		
		// Accumulate gradients for each input tensor
		for i, input := range inputs {
			if input != nil && input.requiresGrad && inputGrads[i] != nil {
				err := input.accumulateGradient(inputGrads[i])
				if err != nil {
					return fmt.Errorf("failed to accumulate gradient for input %d: %v", i, err)
				}
				
				// Recursively compute gradients for this input
				err = input.backwardImpl(visited)
				if err != nil {
					return err
				}
			}
		}
	}
	
	return nil
}

// accumulateGradient adds the given gradient to this tensor's gradient
func (t *Tensor) accumulateGradient(grad *Tensor) error {
	if t.grad == nil {
		// First gradient - just assign it
		var err error
		t.grad, err = grad.Clone()
		if err != nil {
			return fmt.Errorf("failed to clone gradient: %v", err)
		}
	} else {
		// Accumulate by adding to existing gradient
		var result *Tensor
		var err error
		
		if t.grad.Device == GPU || grad.Device == GPU {
			result, err = AddMPS(t.grad, grad)
		} else {
			result, err = Add(t.grad, grad)
		}
		
		if err != nil {
			return fmt.Errorf("failed to accumulate gradient: %v", err)
		}
		
		t.grad = result
	}
	
	return nil
}

// getCreatorInputs extracts input tensors from the operation that created this tensor
// This is a simplified implementation - in practice, operations should store their inputs
func (t *Tensor) getCreatorInputs() []*Tensor {
	if t.creator == nil {
		return nil
	}
	
	// For now, we'll use type assertions to get inputs from specific operation types
	// In a more robust implementation, operations would store their inputs
	switch op := t.creator.(type) {
	case *AddOp:
		return op.inputs
	case *SubOp:
		return op.inputs
	case *MulOp:
		return op.inputs
	case *MatMulOp:
		return op.inputs
	case *ReLUOp:
		return op.inputs
	case *SigmoidOp:
		return op.inputs
	default:
		return nil
	}
}

func calculateStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func calculateNumElements(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	
	elements := 1
	for _, dim := range shape {
		elements *= dim
	}
	return elements
}

func validateShape(shape []int) error {
	for i, dim := range shape {
		if dim <= 0 {
			return fmt.Errorf("invalid shape: dimension %d has size %d, must be positive", i, dim)
		}
	}
	return nil
}

func getSizeForDType(dtype DType) int {
	switch dtype {
	case Float32:
		return 4
	case Float16:
		return 2
	case Int32:
		return 4
	default:
		return 4
	}
}

// GPU Memory Management Methods

// Retain increments the reference count for GPU tensors
func (t *Tensor) Retain() {
	if t.Device == GPU && t.gpuBuffer != nil {
		atomic.AddInt32(&t.refCount, 1)
		// Also retain the underlying buffer if it supports it
		if buffer, ok := t.gpuBuffer.(interface{ Retain() }); ok {
			buffer.Retain()
		}
	}
}

// Release decrements the reference count and releases GPU buffer when count reaches zero
func (t *Tensor) Release() {
	if t.Device == GPU && t.gpuBuffer != nil {
		newCount := atomic.AddInt32(&t.refCount, -1)
		
		// Also release the underlying buffer if it supports it
		if buffer, ok := t.gpuBuffer.(interface{ Release() }); ok {
			buffer.Release()
		}
		
		if newCount <= 0 {
			// Clear reference to buffer when ref count reaches zero
			t.gpuBuffer = nil
			atomic.StoreInt32(&t.refCount, 0)
		}
	}
}

// RefCount returns the current reference count for GPU tensors
func (t *Tensor) RefCount() int32 {
	if t.Device == GPU {
		return atomic.LoadInt32(&t.refCount)
	}
	return 0
}

// SetGPUBuffer sets the GPU buffer for this tensor and initializes reference counting
func (t *Tensor) SetGPUBuffer(buffer interface{}) {
	if t.Device == GPU {
		t.gpuBuffer = buffer
		atomic.StoreInt32(&t.refCount, 1)
	}
}

// GetGPUBuffer returns the GPU buffer for this tensor
func (t *Tensor) GetGPUBuffer() interface{} {
	if t.Device == GPU {
		return t.gpuBuffer
	}
	return nil
}