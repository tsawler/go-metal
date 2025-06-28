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
	GPU           // Temporary GPU tensors - copy results back to CPU
	PersistentGPU // Persistent GPU tensors - keep results on GPU across operations
)

func (d DeviceType) String() string {
	switch d {
	case CPU:
		return "CPU"
	case GPU:
		return "GPU"
	case PersistentGPU:
		return "PersistentGPU"
	default:
		return "Unknown"
	}
}

type Operation interface {
	Forward(...*Tensor) (*Tensor, error)
	Backward(gradOut *Tensor) ([]*Tensor, error)
	GetInputs() []*Tensor
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
	
	// View management fields
	isView       bool        // Whether this tensor is a view of another tensor
	baseData     interface{} // Points to the underlying data buffer (for views)
	baseTensor   *Tensor     // Reference to base tensor (for view lifecycle management)
	offset       int         // Offset into the base data (for slicing views)
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s, elements=%d)", 
		t.Shape, t.DType, t.Device, t.NumElems)
}

// getDataBuffer returns the actual data buffer, handling views
func (t *Tensor) getDataBuffer() interface{} {
	if t.isView && t.baseData != nil {
		return t.baseData
	}
	return t.Data
}

// getLinearIndex converts multi-dimensional indices to linear index using strides
func (t *Tensor) getLinearIndex(indices []int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("indices length %d doesn't match tensor dimensions %d", len(indices), len(t.Shape)))
	}
	
	index := t.offset // Start with view offset
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for dimension %d (size %d)", idx, i, t.Shape[i]))
		}
		index += idx * t.Strides[i]
	}
	return index
}


// materializeView creates a materialized copy of the view data if needed
// This is used for operations that require contiguous data layout
func (t *Tensor) materializeView() interface{} {
	if !t.isView {
		return t.Data
	}
	
	// Create a new contiguous data buffer
	switch t.DType {
	case Float32:
		newData := make([]float32, t.NumElems)
		for i := 0; i < t.NumElems; i++ {
			indices := getIndicesFromLinear(i, t.Shape)
			val, err := t.At(indices...)
			if err != nil {
				panic(fmt.Sprintf("materializeView failed: %v", err))
			}
			newData[i] = val.(float32)
		}
		return newData
	case Int32:
		newData := make([]int32, t.NumElems)
		for i := 0; i < t.NumElems; i++ {
			indices := getIndicesFromLinear(i, t.Shape)
			val, err := t.At(indices...)
			if err != nil {
				panic(fmt.Sprintf("materializeView failed: %v", err))
			}
			newData[i] = val.(int32)
		}
		return newData
	default:
		panic(fmt.Sprintf("unsupported dtype: %s", t.DType))
	}
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
		inputGrads, err := t.creator.Backward(t.grad)
		if err != nil {
			return fmt.Errorf("backward pass failed: %v", err)
		}
		
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
func (t *Tensor) getCreatorInputs() []*Tensor {
	if t.creator == nil {
		return nil
	}
	
	return t.creator.GetInputs()
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
		return 1  // Scalar has 1 element
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
	if (t.Device == GPU || t.Device == PersistentGPU) && t.gpuBuffer != nil {
		atomic.AddInt32(&t.refCount, 1)
		// Also retain the underlying buffer if it supports it
		if buffer, ok := t.gpuBuffer.(interface{ Retain() }); ok {
			buffer.Retain()
		}
	}
}

// Release decrements the reference count and releases GPU buffer when count reaches zero
func (t *Tensor) Release() {
	if (t.Device == GPU || t.Device == PersistentGPU) && t.gpuBuffer != nil {
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
	if t.Device == GPU || t.Device == PersistentGPU {
		return atomic.LoadInt32(&t.refCount)
	}
	return 0
}

// SetGPUBuffer sets the GPU buffer for this tensor and initializes reference counting
func (t *Tensor) SetGPUBuffer(buffer interface{}) {
	if t.Device == GPU || t.Device == PersistentGPU {
		t.gpuBuffer = buffer
		atomic.StoreInt32(&t.refCount, 1)
	}
}

// ToPersistentGPU converts a CPU tensor to persistent GPU tensor that stays on GPU
func (t *Tensor) ToPersistentGPU() (*Tensor, error) {
	return t.ToDevice(PersistentGPU)
}

// IsOnGPU returns true if tensor is on any GPU device
func (t *Tensor) IsOnGPU() bool {
	return t.Device == GPU || t.Device == PersistentGPU
}

// IsPersistent returns true if tensor stays on GPU across operations
func (t *Tensor) IsPersistent() bool {
	return t.Device == PersistentGPU
}

// GetGPUBuffer returns the GPU buffer for this tensor
func (t *Tensor) GetGPUBuffer() interface{} {
	if t.Device == GPU || t.Device == PersistentGPU {
		return t.gpuBuffer
	}
	return nil
}

// Transpose returns a transposed tensor
func (t *Tensor) Transpose(dim0, dim1 int) (*Tensor, error) {
	return Transpose(t, dim0, dim1)
}

// SetData sets the data for this tensor (public version of setData)
func (t *Tensor) SetData(data interface{}) error {
	return t.setData(data)
}

// SetGrad sets the gradient for this tensor
func (t *Tensor) SetGrad(grad *Tensor) {
	t.grad = grad
}

