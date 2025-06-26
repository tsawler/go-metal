package training

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/tsawler/go-metal/tensor"
)

// Dataset interface defines methods that all datasets must implement
type Dataset interface {
	Len() int                                                           // Total number of samples
	Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error) // Returns a single sample (CPU Tensor initially)
}

// DataLoader provides batching, shuffling, and efficient data loading
type DataLoader struct {
	dataset    Dataset
	batchSize  int
	shuffle    bool
	numWorkers int
	device     tensor.DeviceType
	indices    []int
	position   int
	mutex      sync.Mutex
}

// NewDataLoader creates a new DataLoader
func NewDataLoader(dataset Dataset, batchSize int, shuffle bool, numWorkers int, device tensor.DeviceType) *DataLoader {
	if numWorkers <= 0 {
		numWorkers = 1
	}
	
	datasetLen := dataset.Len()
	indices := make([]int, datasetLen)
	for i := range indices {
		indices[i] = i
	}
	
	return &DataLoader{
		dataset:    dataset,
		batchSize:  batchSize,
		shuffle:    shuffle,
		numWorkers: numWorkers,
		device:     device,
		indices:    indices,
		position:   0,
	}
}

// Batch represents a batch of data and labels
type Batch struct {
	Data   *tensor.Tensor
	Labels *tensor.Tensor
}

// Len returns the number of batches in an epoch
func (dl *DataLoader) Len() int {
	return (dl.dataset.Len() + dl.batchSize - 1) / dl.batchSize
}

// Reset resets the data loader for a new epoch
func (dl *DataLoader) Reset() {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	dl.position = 0
	
	if dl.shuffle {
		// Shuffle indices for new epoch
		for i := len(dl.indices) - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		}
	}
}

// Next returns the next batch or nil if epoch is complete
func (dl *DataLoader) Next() (*Batch, error) {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	if dl.position >= len(dl.indices) {
		return nil, nil // End of epoch
	}
	
	// Calculate batch end position
	batchEnd := dl.position + dl.batchSize
	if batchEnd > len(dl.indices) {
		batchEnd = len(dl.indices)
	}
	
	actualBatchSize := batchEnd - dl.position
	batchIndices := dl.indices[dl.position:batchEnd]
	dl.position = batchEnd
	
	// Load batch data
	batch, err := dl.loadBatch(batchIndices, actualBatchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to load batch: %v", err)
	}
	
	return batch, nil
}

// HasNext returns true if there are more batches in the current epoch
func (dl *DataLoader) HasNext() bool {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	return dl.position < len(dl.indices)
}

// loadBatch loads a batch of samples and combines them into batched tensors
func (dl *DataLoader) loadBatch(indices []int, batchSize int) (*Batch, error) {
	if len(indices) == 0 {
		return nil, fmt.Errorf("empty batch indices")
	}
	
	// Load first sample to determine shapes and types
	firstData, firstLabel, err := dl.dataset.Get(indices[0])
	if err != nil {
		return nil, fmt.Errorf("failed to load sample %d: %v", indices[0], err)
	}
	
	// Determine batch shapes
	dataShape := append([]int{batchSize}, firstData.Shape...)
	labelShape := append([]int{batchSize}, firstLabel.Shape...)
	
	// Create batch tensors
	batchData, err := tensor.Zeros(dataShape, firstData.DType, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch data tensor: %v", err)
	}
	
	batchLabels, err := tensor.Zeros(labelShape, firstLabel.DType, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch labels tensor: %v", err)
	}
	
	// Load and copy all samples into batch tensors
	for i, idx := range indices {
		data, label, err := dl.dataset.Get(idx)
		if err != nil {
			return nil, fmt.Errorf("failed to load sample %d: %v", idx, err)
		}
		
		// Copy data into batch tensor
		err = dl.copyInto(batchData, data, i)
		if err != nil {
			return nil, fmt.Errorf("failed to copy data for sample %d: %v", i, err)
		}
		
		// Copy label into batch tensor
		err = dl.copyInto(batchLabels, label, i)
		if err != nil {
			return nil, fmt.Errorf("failed to copy label for sample %d: %v", i, err)
		}
	}
	
	// Transfer to target device if needed
	if dl.device == tensor.GPU {
		batchData, err = batchData.ToGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer batch data to GPU: %v", err)
		}
		
		batchLabels, err = batchLabels.ToGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer batch labels to GPU: %v", err)
		}
	} else if dl.device == tensor.PersistentGPU {
		batchData, err = batchData.ToPersistentGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer batch data to persistent GPU: %v", err)
		}
		
		batchLabels, err = batchLabels.ToPersistentGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer batch labels to persistent GPU: %v", err)
		}
	}
	
	return &Batch{
		Data:   batchData,
		Labels: batchLabels,
	}, nil
}

// copyInto copies a sample tensor into a specific position in the batch tensor
func (dl *DataLoader) copyInto(batchTensor, sampleTensor *tensor.Tensor, batchIndex int) error {
	if batchTensor.DType != sampleTensor.DType {
		return fmt.Errorf("dtype mismatch: batch %s, sample %s", batchTensor.DType, sampleTensor.DType)
	}
	
	// Calculate the offset for this batch index
	sampleSize := sampleTensor.NumElems
	offset := batchIndex * sampleSize
	
	switch batchTensor.DType {
	case tensor.Float32:
		batchData := batchTensor.Data.([]float32)
		sampleData := sampleTensor.Data.([]float32)
		
		if len(sampleData) != sampleSize {
			return fmt.Errorf("sample data size mismatch: expected %d, got %d", sampleSize, len(sampleData))
		}
		
		copy(batchData[offset:offset+sampleSize], sampleData)
		
	case tensor.Int32:
		batchData := batchTensor.Data.([]int32)
		sampleData := sampleTensor.Data.([]int32)
		
		if len(sampleData) != sampleSize {
			return fmt.Errorf("sample data size mismatch: expected %d, got %d", sampleSize, len(sampleData))
		}
		
		copy(batchData[offset:offset+sampleSize], sampleData)
		
	default:
		return fmt.Errorf("unsupported dtype for batch copying: %s", batchTensor.DType)
	}
	
	return nil
}

// Iterator returns a channel-based iterator for easy use in training loops
func (dl *DataLoader) Iterator() <-chan *Batch {
	batchChan := make(chan *Batch, 1)
	
	go func() {
		defer close(batchChan)
		
		dl.Reset()
		
		for dl.HasNext() {
			batch, err := dl.Next()
			if err != nil {
				// In a production system, you might want to handle errors differently
				fmt.Printf("DataLoader error: %v\n", err)
				return
			}
			
			if batch == nil {
				break
			}
			
			batchChan <- batch
		}
	}()
	
	return batchChan
}

// SimpleDataset provides a basic implementation of Dataset for testing and simple use cases
type SimpleDataset struct {
	data   []*tensor.Tensor
	labels []*tensor.Tensor
}

// NewSimpleDataset creates a new SimpleDataset
func NewSimpleDataset(data, labels []*tensor.Tensor) (*SimpleDataset, error) {
	if len(data) != len(labels) {
		return nil, fmt.Errorf("data and labels must have the same length: got %d and %d", len(data), len(labels))
	}
	
	return &SimpleDataset{
		data:   data,
		labels: labels,
	}, nil
}

// Len returns the number of samples in the dataset
func (ds *SimpleDataset) Len() int {
	return len(ds.data)
}

// Get returns a sample at the given index
func (ds *SimpleDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error) {
	if idx < 0 || idx >= len(ds.data) {
		return nil, nil, fmt.Errorf("index %d out of range [0, %d)", idx, len(ds.data))
	}
	
	return ds.data[idx], ds.labels[idx], nil
}

// RandomDataset generates random data for testing purposes
type RandomDataset struct {
	size       int
	dataShape  []int
	labelShape []int
	dataType   tensor.DType
	labelType  tensor.DType
	numClasses int
}

// NewRandomDataset creates a new RandomDataset
func NewRandomDataset(size int, dataShape []int, labelShape []int, dataType, labelType tensor.DType, numClasses int) *RandomDataset {
	return &RandomDataset{
		size:       size,
		dataShape:  dataShape,
		labelShape: labelShape,
		dataType:   dataType,
		labelType:  labelType,
		numClasses: numClasses,
	}
}

// Len returns the size of the dataset
func (rd *RandomDataset) Len() int {
	return rd.size
}

// Get generates a random sample
func (rd *RandomDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error) {
	if idx < 0 || idx >= rd.size {
		return nil, nil, fmt.Errorf("index %d out of range [0, %d)", idx, rd.size)
	}
	
	// Generate random data
	
	switch rd.dataType {
	case tensor.Float32:
		dataSize := 1
		for _, dim := range rd.dataShape {
			dataSize *= dim
		}
		
		randomData := make([]float32, dataSize)
		for i := range randomData {
			randomData[i] = rand.Float32()*2.0 - 1.0 // Range [-1, 1]
		}
		
		data, err = tensor.NewTensor(rd.dataShape, rd.dataType, tensor.CPU, randomData)
		
	case tensor.Int32:
		dataSize := 1
		for _, dim := range rd.dataShape {
			dataSize *= dim
		}
		
		randomData := make([]int32, dataSize)
		for i := range randomData {
			randomData[i] = int32(rand.Intn(256)) // Range [0, 255]
		}
		
		data, err = tensor.NewTensor(rd.dataShape, rd.dataType, tensor.CPU, randomData)
		
	default:
		return nil, nil, fmt.Errorf("unsupported data type: %s", rd.dataType)
	}
	
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create data tensor: %v", err)
	}
	
	// Generate random label
	
	switch rd.labelType {
	case tensor.Int32:
		if len(rd.labelShape) == 1 && rd.labelShape[0] == 1 {
			// Classification label - single class index
			randomLabel := []int32{int32(rand.Intn(rd.numClasses))}
			label, err = tensor.NewTensor(rd.labelShape, rd.labelType, tensor.CPU, randomLabel)
		} else {
			// Multi-dimensional label
			labelSize := 1
			for _, dim := range rd.labelShape {
				labelSize *= dim
			}
			
			randomLabel := make([]int32, labelSize)
			for i := range randomLabel {
				randomLabel[i] = int32(rand.Intn(rd.numClasses))
			}
			
			label, err = tensor.NewTensor(rd.labelShape, rd.labelType, tensor.CPU, randomLabel)
		}
		
	case tensor.Float32:
		labelSize := 1
		for _, dim := range rd.labelShape {
			labelSize *= dim
		}
		
		randomLabel := make([]float32, labelSize)
		for i := range randomLabel {
			randomLabel[i] = rand.Float32()
		}
		
		label, err = tensor.NewTensor(rd.labelShape, rd.labelType, tensor.CPU, randomLabel)
		
	default:
		return nil, nil, fmt.Errorf("unsupported label type: %s", rd.labelType)
	}
	
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	
	return data, label, nil
}