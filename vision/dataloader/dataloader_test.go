package dataloader

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// MockDataset implements the Dataset interface for testing
type MockDataset struct {
	items []MockItem
}

type MockItem struct {
	imagePath string
	label     int
}

func (md *MockDataset) Len() int {
	return len(md.items)
}

func (md *MockDataset) GetItem(index int) (imagePath string, label int, err error) {
	if index < 0 || index >= len(md.items) {
		return "", 0, fmt.Errorf("index %d out of range [0, %d)", index, len(md.items))
	}
	item := md.items[index]
	return item.imagePath, item.label, nil
}

// NewMockDataset creates a mock dataset with the specified number of items
func NewMockDataset(numItems int) *MockDataset {
	items := make([]MockItem, numItems)
	for i := 0; i < numItems; i++ {
		items[i] = MockItem{
			imagePath: fmt.Sprintf("image_%d.jpg", i),
			label:     i % 3, // 3 classes: 0, 1, 2
		}
	}
	return &MockDataset{items: items}
}

// TestNewDataLoader tests DataLoader creation with various configurations
func TestNewDataLoader(t *testing.T) {
	dataset := NewMockDataset(100)

	t.Run("DefaultConfig", func(t *testing.T) {
		config := Config{
			BatchSize: 32,
			Shuffle:   true,
			ImageSize: 224,
		}

		dl := NewDataLoader(dataset, config)

		if dl.dataset != dataset {
			t.Error("Dataset not set correctly")
		}
		if dl.batchSize != 32 {
			t.Errorf("Expected batch size 32, got %d", dl.batchSize)
		}
		if !dl.shuffle {
			t.Error("Expected shuffle to be true")
		}
		if len(dl.indices) != 100 {
			t.Errorf("Expected 100 indices, got %d", len(dl.indices))
		}
		if dl.position != 0 {
			t.Errorf("Expected initial position 0, got %d", dl.position)
		}
		if dl.ownedCache != true {
			t.Error("Expected owned cache to be true when no shared cache provided")
		}
		if dl.imageSize != 224 {
			t.Errorf("Expected image size 224, got %d", dl.imageSize)
		}
	})

	t.Run("WithSharedCache", func(t *testing.T) {
		sharedCache := NewCacheManager(500, 3*224*224)
		config := Config{
			BatchSize:    16,
			Shuffle:      false,
			ImageSize:    224,
			CacheManager: sharedCache,
		}

		dl := NewDataLoader(dataset, config)

		if dl.cacheManager != sharedCache {
			t.Error("Shared cache not set correctly")
		}
		if dl.ownedCache != false {
			t.Error("Expected owned cache to be false when shared cache provided")
		}
		if dl.shuffle {
			t.Error("Expected shuffle to be false")
		}
	})

	t.Run("ShuffleValidation", func(t *testing.T) {
		config := Config{
			BatchSize: 10,
			Shuffle:   true,
			ImageSize: 224,
		}

		// Create multiple DataLoaders and check that indices are different
		dl1 := NewDataLoader(dataset, config)
		dl2 := NewDataLoader(dataset, config)

		// The probability of getting the same shuffle is extremely low
		same := true
		for i := 0; i < min(10, len(dl1.indices)); i++ {
			if dl1.indices[i] != dl2.indices[i] {
				same = false
				break
			}
		}
		
		// This test might occasionally fail due to randomness, but probability is very low
		if same && len(dl1.indices) > 1 {
			t.Log("Warning: Indices are the same (low probability event)")
		}
	})
}

// TestDataLoaderReset tests the Reset functionality
func TestDataLoaderReset(t *testing.T) {
	dataset := NewMockDataset(50)
	config := Config{
		BatchSize: 10,
		Shuffle:   true,
		ImageSize: 224,
	}

	dl := NewDataLoader(dataset, config)

	// Save original indices
	originalIndices := make([]int, len(dl.indices))
	copy(originalIndices, dl.indices)

	// Advance position
	dl.position = 25

	// Reset without shuffle
	dl.shuffle = false
	dl.Reset()

	if dl.position != 0 {
		t.Errorf("Expected position 0 after reset, got %d", dl.position)
	}

	// With shuffle disabled, indices should remain the same
	for i, idx := range dl.indices {
		if idx != originalIndices[i] {
			t.Error("Indices changed when shuffle was disabled")
			break
		}
	}

	// Reset with shuffle
	dl.shuffle = true
	dl.Reset()

	if dl.position != 0 {
		t.Errorf("Expected position 0 after reset, got %d", dl.position)
	}

	// Check that indices might be different (though not guaranteed)
	// This is a probabilistic test
	different := false
	for i, idx := range dl.indices {
		if idx != originalIndices[i] {
			different = true
			break
		}
	}
	
	if !different && len(dl.indices) > 1 {
		t.Log("Warning: Indices are the same after shuffle reset (low probability)")
	}
}

// TestDataLoaderProgress tests progress tracking
func TestDataLoaderProgress(t *testing.T) {
	dataset := NewMockDataset(100)
	config := Config{
		BatchSize: 10,
		Shuffle:   false,
		ImageSize: 224,
	}

	dl := NewDataLoader(dataset, config)

	// Initial progress
	current, total := dl.Progress()
	if current != 0 || total != 100 {
		t.Errorf("Expected progress (0, 100), got (%d, %d)", current, total)
	}

	// Advance position manually
	dl.position = 50
	current, total = dl.Progress()
	if current != 50 || total != 100 {
		t.Errorf("Expected progress (50, 100), got (%d, %d)", current, total)
	}
}

// TestDataLoaderClearCache tests cache clearing functionality
func TestDataLoaderClearCache(t *testing.T) {
	dataset := NewMockDataset(10)
	
	t.Run("OwnedCache", func(t *testing.T) {
		config := Config{
			BatchSize:    5,
			Shuffle:      false,
			ImageSize:    224,
			MaxCacheSize: 10,
		}

		dl := NewDataLoader(dataset, config)
		
		// Add some data to cache
		dl.cacheManager.Put("test_key", []float32{1, 2, 3})
		
		stats := dl.cacheManager.Stats()
		if stats.Size != 1 {
			t.Errorf("Expected cache size 1, got %d", stats.Size)
		}

		// Clear cache
		dl.ClearCache()
		
		stats = dl.cacheManager.Stats()
		if stats.Size != 0 {
			t.Errorf("Expected cache size 0 after clear, got %d", stats.Size)
		}
	})

	t.Run("SharedCache", func(t *testing.T) {
		sharedCache := NewCacheManager(10, 100)
		sharedCache.Put("shared_key", []float32{4, 5, 6})
		
		config := Config{
			BatchSize:    5,
			Shuffle:      false,
			ImageSize:    224,
			CacheManager: sharedCache,
		}

		dl := NewDataLoader(dataset, config)
		
		// Add more data to shared cache
		dl.cacheManager.Put("test_key", []float32{1, 2, 3})
		
		stats := dl.cacheManager.Stats()
		if stats.Size != 2 {
			t.Errorf("Expected cache size 2, got %d", stats.Size)
		}

		// Clear cache - should NOT clear shared cache
		dl.ClearCache()
		
		stats = dl.cacheManager.Stats()
		if stats.Size != 2 {
			t.Errorf("Expected cache size 2 after clear (shared cache), got %d", stats.Size)
		}
	})
}

// TestDataLoaderGetCacheManager tests cache manager retrieval
func TestDataLoaderGetCacheManager(t *testing.T) {
	dataset := NewMockDataset(10)
	config := Config{
		BatchSize: 5,
		Shuffle:   false,
		ImageSize: 224,
	}

	dl := NewDataLoader(dataset, config)
	cacheManager := dl.GetCacheManager()

	if cacheManager == nil {
		t.Error("Expected non-nil cache manager")
	}

	if cacheManager != dl.cacheManager {
		t.Error("Returned cache manager does not match internal cache manager")
	}
}

// TestDataLoaderConcurrency tests thread safety
func TestDataLoaderConcurrency(t *testing.T) {
	dataset := NewMockDataset(100)
	config := Config{
		BatchSize: 10,
		Shuffle:   false,
		ImageSize: 224,
	}

	dl := NewDataLoader(dataset, config)

	// Run multiple goroutines accessing DataLoader concurrently
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- true }()
			
			// Test concurrent access to various methods
			dl.Progress()
			dl.Reset()
			dl.Stats()
			dl.GetCacheManager()
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for concurrent operations")
		}
	}
}

// TestDataLoaderBufferReuse tests that buffers are reused properly
func TestDataLoaderBufferReuse(t *testing.T) {
	dataset := NewMockDataset(20)
	config := Config{
		BatchSize: 5,
		Shuffle:   false,
		ImageSize: 64, // Smaller for faster tests
	}

	// Create a mock image file for testing
	tempDir := t.TempDir()
	mockImagePath := filepath.Join(tempDir, "test.jpg")
	if err := createMockJPEG(mockImagePath, 64, 64); err != nil {
		t.Fatalf("Failed to create mock JPEG: %v", err)
	}

	// Update dataset to use real image path
	for i := range dataset.items {
		dataset.items[i].imagePath = mockImagePath
	}

	dl := NewDataLoader(dataset, config)

	// First batch
	imageData1, labelData1, batchSize1, err := dl.NextBatch()
	if err != nil {
		t.Fatalf("Error getting first batch: %v", err)
	}

	if batchSize1 != 5 {
		t.Errorf("Expected batch size 5, got %d", batchSize1)
	}

	// Save buffer pointers
	imagePtr1 := &dl.imageDataBuffer[0]
	labelPtr1 := &dl.labelDataBuffer[0]

	// Second batch
	imageData2, labelData2, batchSize2, err := dl.NextBatch()
	if err != nil {
		t.Fatalf("Error getting second batch: %v", err)
	}

	if batchSize2 != 5 {
		t.Errorf("Expected batch size 5, got %d", batchSize2)
	}

	// Check that same buffers are reused
	imagePtr2 := &dl.imageDataBuffer[0]
	labelPtr2 := &dl.labelDataBuffer[0]

	if imagePtr1 != imagePtr2 {
		t.Error("Image buffer was not reused")
	}

	if labelPtr1 != labelPtr2 {
		t.Error("Label buffer was not reused")
	}

	// Verify data is different (different positions in dataset)
	if len(imageData1) != len(imageData2) {
		t.Error("Image data length should be the same")
	}

	if len(labelData1) != len(labelData2) {
		t.Error("Label data length should be the same")
	}
}

// TestDataLoaderEdgeCases tests various edge cases
func TestDataLoaderEdgeCases(t *testing.T) {
	t.Run("EmptyDataset", func(t *testing.T) {
		dataset := NewMockDataset(0)
		config := Config{
			BatchSize: 10,
			Shuffle:   false,
			ImageSize: 224,
		}

		dl := NewDataLoader(dataset, config)

		imageData, labelData, batchSize, err := dl.NextBatch()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		if imageData != nil || labelData != nil || batchSize != 0 {
			t.Error("Expected nil data and zero batch size for empty dataset")
		}
	})

	t.Run("BatchSizeLargerThanDataset", func(t *testing.T) {
		dataset := NewMockDataset(5)
		config := Config{
			BatchSize: 10,
			Shuffle:   false,
			ImageSize: 64,
		}

		// Create mock image
		tempDir := t.TempDir()
		mockImagePath := filepath.Join(tempDir, "test.jpg")
		if err := createMockJPEG(mockImagePath, 64, 64); err != nil {
			t.Fatalf("Failed to create mock JPEG: %v", err)
		}

		for i := range dataset.items {
			dataset.items[i].imagePath = mockImagePath
		}

		dl := NewDataLoader(dataset, config)

		imageData, labelData, batchSize, err := dl.NextBatch()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		_ = imageData // Suppress unused variable warning

		if batchSize != 5 {
			t.Errorf("Expected batch size 5 (dataset size), got %d", batchSize)
		}

		if len(labelData) < batchSize {
			t.Error("Label data length should accommodate actual batch size")
		}
	})

	t.Run("MultipleEpochs", func(t *testing.T) {
		dataset := NewMockDataset(7)
		config := Config{
			BatchSize: 3,
			Shuffle:   false,
			ImageSize: 64,
		}

		// Create mock image
		tempDir := t.TempDir()
		mockImagePath := filepath.Join(tempDir, "test.jpg")
		if err := createMockJPEG(mockImagePath, 64, 64); err != nil {
			t.Fatalf("Failed to create mock JPEG: %v", err)
		}

		for i := range dataset.items {
			dataset.items[i].imagePath = mockImagePath
		}

		dl := NewDataLoader(dataset, config)

		// First epoch: 3 + 3 + 1 = 7 samples
		_, _, batchSize1, _ := dl.NextBatch()
		if batchSize1 != 3 {
			t.Errorf("Expected first batch size 3, got %d", batchSize1)
		}

		_, _, batchSize2, _ := dl.NextBatch()
		if batchSize2 != 3 {
			t.Errorf("Expected second batch size 3, got %d", batchSize2)
		}

		_, _, batchSize3, _ := dl.NextBatch()
		if batchSize3 != 1 {
			t.Errorf("Expected third batch size 1, got %d", batchSize3)
		}

		// Should be no more data
		_, _, batchSize4, _ := dl.NextBatch()
		if batchSize4 != 0 {
			t.Errorf("Expected no more data, got batch size %d", batchSize4)
		}

		// Reset and try again
		dl.Reset()
		_, _, batchSize5, _ := dl.NextBatch()
		if batchSize5 != 3 {
			t.Errorf("Expected first batch size 3 after reset, got %d", batchSize5)
		}
	})
}

// TestDataLoaderStats tests statistics functionality
func TestDataLoaderStats(t *testing.T) {
	dataset := NewMockDataset(10)
	config := Config{
		BatchSize:    5,
		Shuffle:      false,
		ImageSize:    224,
		MaxCacheSize: 5,
	}

	dl := NewDataLoader(dataset, config)

	// Initially, stats should show empty cache
	stats := dl.Stats()
	if !containsString(stats, "0/5") {
		t.Errorf("Expected stats to show 0/5 items, got: %s", stats)
	}

	// Add some items to cache manually
	dl.cacheManager.Put("item1", make([]float32, 100))
	dl.cacheManager.Put("item2", make([]float32, 100))

	stats = dl.Stats()
	if !containsString(stats, "2/5") {
		t.Errorf("Expected stats to show 2/5 items, got: %s", stats)
	}
}

// Helper function to check if a string contains a substring
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || 
		   len(s) > len(substr) && s[len(s)-len(substr):] == substr ||
		   (len(s) > len(substr) && 
		    func() bool {
		        for i := 0; i <= len(s)-len(substr); i++ {
		            if s[i:i+len(substr)] == substr {
		                return true
		            }
		        }
		        return false
		    }())
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Helper function to create a mock JPEG file for testing
func createMockJPEG(path string, width, height int) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	// Create a simple test image
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	
	// Fill with a simple pattern
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Create a gradient pattern
			r := uint8((x * 255) / width)
			g := uint8((y * 255) / height)
			b := uint8(128)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	
	// Encode as JPEG
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	if err != nil {
		return err
	}
	
	// Write to file
	return os.WriteFile(path, buf.Bytes(), 0644)
}

// BenchmarkDataLoaderNextBatch benchmarks the NextBatch operation
func BenchmarkDataLoaderNextBatch(b *testing.B) {
	dataset := NewMockDataset(1000)
	config := Config{
		BatchSize:    32,
		Shuffle:      false,
		ImageSize:    224,
		MaxCacheSize: 100,
	}

	dl := NewDataLoader(dataset, config)

	// Create a mock image file
	tempDir := b.TempDir()
	mockImagePath := filepath.Join(tempDir, "test.jpg")
	if err := createMockJPEG(mockImagePath, 224, 224); err != nil {
		b.Fatalf("Failed to create mock JPEG: %v", err)
	}

	// Update dataset to use real image path
	for i := range dataset.items {
		dataset.items[i].imagePath = mockImagePath
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset if we reach the end
		if dl.position >= len(dl.indices) {
			dl.Reset()
		}
		
		_, _, _, err := dl.NextBatch()
		if err != nil {
			b.Fatalf("Error in NextBatch: %v", err)
		}
	}
}

// BenchmarkDataLoaderWithCache benchmarks performance with caching
func BenchmarkDataLoaderWithCache(b *testing.B) {
	dataset := NewMockDataset(100)
	config := Config{
		BatchSize:    16,
		Shuffle:      false,
		ImageSize:    224,
		MaxCacheSize: 200, // Large cache
	}

	dl := NewDataLoader(dataset, config)

	// Create a mock image file
	tempDir := b.TempDir()
	mockImagePath := filepath.Join(tempDir, "test.jpg")
	if err := createMockJPEG(mockImagePath, 224, 224); err != nil {
		b.Fatalf("Failed to create mock JPEG: %v", err)
	}

	// Update dataset to use real image path
	for i := range dataset.items {
		dataset.items[i].imagePath = mockImagePath
	}

	// Warm up cache
	for i := 0; i < 3; i++ {
		dl.Reset()
		for dl.position < len(dl.indices) {
			_, _, _, err := dl.NextBatch()
			if err != nil {
				b.Fatalf("Error warming up cache: %v", err)
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dl.Reset()
		_, _, _, err := dl.NextBatch()
		if err != nil {
			b.Fatalf("Error in NextBatch: %v", err)
		}
	}
}