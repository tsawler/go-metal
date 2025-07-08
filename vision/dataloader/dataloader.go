package dataloader

import (
	"math/rand"
	"os"
	"sync"

	"github.com/tsawler/go-metal/vision/preprocessing"
)

// Dataset interface defines the contract for datasets
type Dataset interface {
	Len() int
	GetItem(index int) (imagePath string, label int, err error)
}

// DataLoader handles memory-efficient batch data loading with smart caching
type DataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
	indices   []int
	position  int
	mu        sync.Mutex

	// Buffer reuse for memory efficiency
	imageDataBuffer []float32
	labelDataBuffer []int32

	// Cache manager - can be shared between DataLoaders
	cacheManager *CacheManager
	ownedCache   bool // Whether this DataLoader owns the cache

	// Image processor
	processor *preprocessing.ImageProcessor
	imageSize int
}

// Config holds configuration for DataLoader
type Config struct {
	BatchSize    int
	Shuffle      bool
	MaxCacheSize int // Maximum number of images to cache
	ImageSize    int
	NumWorkers   int // Number of parallel workers for preprocessing
	CacheManager *CacheManager // Optional shared cache manager
}

// NewDataLoader creates a new data loader
func NewDataLoader(dataset Dataset, config Config) *DataLoader {
	if config.MaxCacheSize == 0 {
		config.MaxCacheSize = 1000 // Default cache size
	}

	indices := make([]int, dataset.Len())
	for i := range indices {
		indices[i] = i
	}

	if config.Shuffle {
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// Use provided cache manager or create a new one
	var cacheManager *CacheManager
	var ownedCache bool
	if config.CacheManager != nil {
		cacheManager = config.CacheManager
		ownedCache = false
	} else {
		// Create a new cache manager for this DataLoader
		imageSize := config.ImageSize
		itemSize := 3 * imageSize * imageSize // CHW format
		cacheManager = NewCacheManager(config.MaxCacheSize, itemSize)
		ownedCache = true
	}

	return &DataLoader{
		dataset:      dataset,
		batchSize:    config.BatchSize,
		shuffle:      config.Shuffle,
		indices:      indices,
		position:     0,
		cacheManager: cacheManager,
		ownedCache:   ownedCache,
		processor:    preprocessing.NewImageProcessor(config.ImageSize),
		imageSize:    config.ImageSize,
	}
}

// Reset resets the data loader to the beginning
func (dl *DataLoader) Reset() {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	dl.position = 0
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
}

// NextBatch loads the next batch of images
func (dl *DataLoader) NextBatch() (imageData []float32, labelData []int32, actualBatchSize int, err error) {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	remaining := len(dl.indices) - dl.position
	if remaining <= 0 {
		return nil, nil, 0, nil // No more data
	}

	batchSize := dl.batchSize
	if remaining < batchSize {
		batchSize = remaining
	}

	// Calculate required buffer sizes
	channels := 3
	requiredImageSize := batchSize * channels * dl.imageSize * dl.imageSize
	requiredLabelSize := batchSize

	// Resize buffers only if needed
	if len(dl.imageDataBuffer) < requiredImageSize {
		dl.imageDataBuffer = make([]float32, requiredImageSize)
	}
	if len(dl.labelDataBuffer) < requiredLabelSize {
		dl.labelDataBuffer = make([]int32, requiredLabelSize)
	}

	imageData = dl.imageDataBuffer[:requiredImageSize]
	labelData = dl.labelDataBuffer[:requiredLabelSize]

	actualBatchSize = 0
	for i := 0; i < batchSize; i++ {
		if dl.position >= len(dl.indices) {
			break
		}

		idx := dl.indices[dl.position]
		imagePath, label, err := dl.dataset.GetItem(idx)
		if err != nil {
			dl.position++
			continue
		}

		// Load and preprocess image with caching
		imgData, err := dl.loadImageWithCache(imagePath)
		if err != nil {
			dl.position++
			continue
		}

		// Copy image data to batch
		pixelsPerImage := channels * dl.imageSize * dl.imageSize
		copy(imageData[actualBatchSize*pixelsPerImage:(actualBatchSize+1)*pixelsPerImage], imgData)
		labelData[actualBatchSize] = int32(label)

		actualBatchSize++
		dl.position++
	}

	return imageData, labelData, actualBatchSize, nil
}

// loadImageWithCache loads an image with caching support
func (dl *DataLoader) loadImageWithCache(imagePath string) ([]float32, error) {
	// Check cache first
	if cachedData, exists := dl.cacheManager.Get(imagePath); exists {
		return cachedData, nil
	}

	// Load and preprocess image
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	processedImg, err := dl.processor.DecodeAndPreprocess(file)
	if err != nil {
		return nil, err
	}

	// Cache the result
	dl.cacheManager.Put(imagePath, processedImg.Data)
	return processedImg.Data, nil
}

// Stats returns cache statistics
func (dl *DataLoader) Stats() string {
	return dl.cacheManager.Stats().String()
}

// Progress returns the current progress through the dataset
func (dl *DataLoader) Progress() (current, total int) {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.position, len(dl.indices)
}

// ClearCache clears the image cache
func (dl *DataLoader) ClearCache() {
	if dl.ownedCache {
		dl.cacheManager.Clear()
	}
	// If cache is shared, don't clear it
}

// GetCacheManager returns the cache manager for sharing between DataLoaders
func (dl *DataLoader) GetCacheManager() *CacheManager {
	return dl.cacheManager
}