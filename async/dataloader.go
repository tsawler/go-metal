package async

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/tsawler/go-metal/memory"
)

// GPUBatch represents a batch of data ready for GPU training
type GPUBatch struct {
	InputTensor  *memory.Tensor    // GPU-resident input data
	LabelTensor  *memory.Tensor    // GPU-resident label data
	WeightTensors []*memory.Tensor // GPU-resident weight tensors
	BatchID      uint64            // Unique identifier for this batch
	Generation   uint64            // For cleanup tracking
}

// Release releases all tensors in the batch
func (gb *GPUBatch) Release() {
	if gb.InputTensor != nil {
		gb.InputTensor.Release()
		gb.InputTensor = nil
	}
	if gb.LabelTensor != nil {
		gb.LabelTensor.Release()
		gb.LabelTensor = nil
	}
	for _, tensor := range gb.WeightTensors {
		if tensor != nil {
			tensor.Release()
		}
	}
	gb.WeightTensors = nil
}

// DataSource represents a source of training data
type DataSource interface {
	// GetBatch returns the next batch of data
	// inputData: raw input data, inputShape: tensor shape
	// labelData: raw label data, labelShape: tensor shape
	GetBatch(batchSize int) (inputData []float32, inputShape []int, labelData []float32, labelShape []int, err error)
	
	// Size returns the total number of samples available
	Size() int
	
	// Reset resets the data source to the beginning
	Reset() error
}

// AsyncDataLoader manages background data loading with GPU transfer pipeline
type AsyncDataLoader struct {
	dataSource    DataSource
	batchSize     int
	prefetchDepth int           // Number of batches to prefetch
	workers       int           // Number of background workers
	
	// Channels for pipeline
	batchChannel  chan *GPUBatch  // Output channel for ready batches
	errorChannel  chan error      // Error reporting
	
	// Control
	ctx           context.Context
	cancel        context.CancelFunc
	wg            sync.WaitGroup
	
	// State
	batchCounter  uint64
	generation    uint64
	isRunning     bool
	mutex         sync.RWMutex
	
	// Memory management
	memoryManager *memory.MemoryManager
	
	// Staging buffers for CPU→GPU transfers
	stagingPool   *StagingBufferPool
}

// AsyncDataLoaderConfig holds configuration for the data loader
type AsyncDataLoaderConfig struct {
	BatchSize     int           // Size of each batch
	PrefetchDepth int           // Number of batches to prefetch (default: 3)
	Workers       int           // Number of background workers (default: 2)
	MemoryManager *memory.MemoryManager
}

// NewAsyncDataLoader creates a new asynchronous data loader
func NewAsyncDataLoader(dataSource DataSource, config AsyncDataLoaderConfig) (*AsyncDataLoader, error) {
	if dataSource == nil {
		return nil, fmt.Errorf("data source cannot be nil")
	}
	
	if config.BatchSize <= 0 {
		return nil, fmt.Errorf("batch size must be positive, got %d", config.BatchSize)
	}
	
	// Set defaults
	if config.PrefetchDepth <= 0 {
		config.PrefetchDepth = 3
	}
	if config.Workers <= 0 {
		config.Workers = 2
	}
	if config.MemoryManager == nil {
		return nil, fmt.Errorf("memory manager is required")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create staging buffer pool
	stagingPool, err := NewStagingBufferPool(config.MemoryManager, 10) // Pool of 10 staging buffers
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create staging buffer pool: %v", err)
	}
	
	loader := &AsyncDataLoader{
		dataSource:    dataSource,
		batchSize:     config.BatchSize,
		prefetchDepth: config.PrefetchDepth,
		workers:       config.Workers,
		
		batchChannel:  make(chan *GPUBatch, config.PrefetchDepth),
		errorChannel:  make(chan error, config.Workers),
		
		ctx:           ctx,
		cancel:        cancel,
		
		batchCounter:  0,
		generation:    1,
		isRunning:     false,
		
		memoryManager: config.MemoryManager,
		stagingPool:   stagingPool,
	}
	
	return loader, nil
}

// Start begins the async data loading pipeline
func (adl *AsyncDataLoader) Start() error {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()
	
	if adl.isRunning {
		return fmt.Errorf("data loader is already running")
	}
	
	// Start worker goroutines
	for i := 0; i < adl.workers; i++ {
		adl.wg.Add(1)
		go adl.worker(i)
	}
	
	adl.isRunning = true
	return nil
}

// Stop stops the async data loading pipeline
func (adl *AsyncDataLoader) Stop() error {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()
	
	if !adl.isRunning {
		return nil
	}
	
	// Cancel context to stop workers
	adl.cancel()
	
	// Wait for all workers to finish
	adl.wg.Wait()
	
	// Drain remaining batches and release them
	close(adl.batchChannel)
	for batch := range adl.batchChannel {
		batch.Release()
	}
	
	// Cleanup staging pool
	adl.stagingPool.Cleanup()
	
	adl.isRunning = false
	return nil
}

// GetBatch returns the next ready batch (blocks until available)
func (adl *AsyncDataLoader) GetBatch() (*GPUBatch, error) {
	select {
	case batch := <-adl.batchChannel:
		if batch == nil {
			return nil, fmt.Errorf("data loader has been stopped")
		}
		return batch, nil
	case err := <-adl.errorChannel:
		return nil, fmt.Errorf("data loader error: %v", err)
	case <-adl.ctx.Done():
		return nil, fmt.Errorf("data loader has been cancelled")
	}
}

// TryGetBatch returns the next ready batch (non-blocking)
func (adl *AsyncDataLoader) TryGetBatch() (*GPUBatch, error) {
	select {
	case batch := <-adl.batchChannel:
		if batch == nil {
			return nil, fmt.Errorf("data loader has been stopped")
		}
		return batch, nil
	case err := <-adl.errorChannel:
		return nil, fmt.Errorf("data loader error: %v", err)
	default:
		return nil, nil // No batch ready
	}
}

// worker runs in background to load and prepare batches
func (adl *AsyncDataLoader) worker(workerID int) {
	defer adl.wg.Done()
	
	for {
		select {
		case <-adl.ctx.Done():
			return
		default:
			batch, err := adl.prepareBatch()
			if err != nil {
				select {
				case adl.errorChannel <- fmt.Errorf("worker %d: %v", workerID, err):
				case <-adl.ctx.Done():
				}
				return
			}
			
			// Try to send batch (with timeout to avoid blocking forever)
			select {
			case adl.batchChannel <- batch:
				// Successfully sent batch
			case <-adl.ctx.Done():
				// Context cancelled, cleanup and exit
				batch.Release()
				return
			case <-time.After(time.Second):
				// Timeout sending batch (channel full), release and continue
				batch.Release()
			}
		}
	}
}

// prepareBatch loads data from source and transfers to GPU
func (adl *AsyncDataLoader) prepareBatch() (*GPUBatch, error) {
	// Get raw data from source
	inputData, inputShape, labelData, labelShape, err := adl.dataSource.GetBatch(adl.batchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to get batch from data source: %v", err)
	}
	
	// Create GPU tensors
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	
	labelTensor, err := memory.NewTensor(labelShape, memory.Float32, memory.GPU)
	if err != nil {
		inputTensor.Release()
		return nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	
	// Transfer data to GPU using staging buffers
	err = adl.stagingPool.TransferToGPU(inputData, inputTensor)
	if err != nil {
		inputTensor.Release()
		labelTensor.Release()
		return nil, fmt.Errorf("failed to transfer input data to GPU: %v", err)
	}
	
	// Convert int32 labels to one-hot float32 format and transfer
	oneHotLabels := make([]float32, labelShape[0]*2) // 2 classes
	for i, label := range labelData {
		if label >= 0 && label < 2 {
			oneHotLabels[i*2+int(label)] = 1.0
		}
	}
	
	err = adl.stagingPool.TransferToGPU(oneHotLabels, labelTensor)
	if err != nil {
		inputTensor.Release()
		labelTensor.Release()
		return nil, fmt.Errorf("failed to transfer label data to GPU: %v", err)
	}
	
	// Create weight tensors (for hybrid approach: FC weights + bias)
	weightTensors, err := adl.createWeightTensors()
	if err != nil {
		inputTensor.Release()
		labelTensor.Release()
		return nil, fmt.Errorf("failed to create weight tensors: %v", err)
	}
	
	adl.mutex.Lock()
	batchID := adl.batchCounter
	adl.batchCounter++
	adl.mutex.Unlock()
	
	batch := &GPUBatch{
		InputTensor:   inputTensor,
		LabelTensor:   labelTensor,
		WeightTensors: weightTensors,
		BatchID:       batchID,
		Generation:    adl.generation,
	}
	
	return batch, nil
}

// createWeightTensors creates weight tensors for the hybrid approach
func (adl *AsyncDataLoader) createWeightTensors() ([]*memory.Tensor, error) {
	// Create weight tensors for hybrid approach (only FC layer - conv is built-in)
	shapes := [][]int{
		{8, 2}, // FC layer weights (8 inputs, 2 outputs)
		{2},    // FC layer bias
	}
	
	weights := make([]*memory.Tensor, len(shapes))
	
	for i, shape := range shapes {
		tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup previously created tensors
			for j := 0; j < i; j++ {
				weights[j].Release()
			}
			return nil, fmt.Errorf("failed to create weight tensor %d: %v", i, err)
		}
		weights[i] = tensor
	}
	
	return weights, nil
}

// Stats returns statistics about the data loader
func (adl *AsyncDataLoader) Stats() AsyncDataLoaderStats {
	adl.mutex.RLock()
	defer adl.mutex.RUnlock()
	
	return AsyncDataLoaderStats{
		IsRunning:       adl.isRunning,
		BatchesProduced: adl.batchCounter,
		QueuedBatches:   len(adl.batchChannel),
		QueueCapacity:   cap(adl.batchChannel),
		Workers:         adl.workers,
		Generation:      adl.generation,
	}
}

// AsyncDataLoaderStats provides statistics about the data loader
type AsyncDataLoaderStats struct {
	IsRunning       bool
	BatchesProduced uint64
	QueuedBatches   int
	QueueCapacity   int
	Workers         int
	Generation      uint64
}