package preprocessing

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// createMockJPEGImage creates a simple colored JPEG image for testing
func createMockJPEGImage(width, height int, baseColor color.RGBA) ([]byte, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	
	// Fill with gradient based on base color
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Create a simple gradient pattern
			factor := float64(x+y) / float64(width+height)
			r := uint8(float64(baseColor.R) * factor)
			g := uint8(float64(baseColor.G) * factor) 
			b := uint8(float64(baseColor.B) * factor)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	return buf.Bytes(), err
}

// createTestJPEGFile creates a JPEG file for testing
func createTestJPEGFile(path string, width, height int, baseColor color.RGBA) error {
	data, err := createMockJPEGImage(width, height, baseColor)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// TestNewImageProcessor tests ImageProcessor creation
func TestNewImageProcessor(t *testing.T) {
	targetSize := 224
	processor := NewImageProcessor(targetSize)
	
	if processor == nil {
		t.Error("Expected non-nil processor")
	}
	
	if processor.targetSize != targetSize {
		t.Errorf("Expected target size %d, got %d", targetSize, processor.targetSize)
	}
	
	// Initial buffers should be nil
	if processor.tempImageBuffer != nil {
		t.Error("Expected nil tempImageBuffer initially")
	}
	
	if processor.processBuffer != nil {
		t.Error("Expected nil processBuffer initially")
	}
}

// TestImageProcessorDecodeAndPreprocess tests image decoding and preprocessing
func TestImageProcessorDecodeAndPreprocess(t *testing.T) {
	processor := NewImageProcessor(64)
	
	t.Run("ValidJPEGImage", func(t *testing.T) {
		// Create a test image
		baseColor := color.RGBA{255, 128, 64, 255}
		jpegData, err := createMockJPEGImage(100, 100, baseColor)
		if err != nil {
			t.Fatalf("Failed to create mock image: %v", err)
		}
		
		reader := bytes.NewReader(jpegData)
		result, err := processor.DecodeAndPreprocess(reader)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Verify result structure
		if result == nil {
			t.Fatal("Expected non-nil result")
		}
		
		if result.Width != 64 {
			t.Errorf("Expected width 64, got %d", result.Width)
		}
		
		if result.Height != 64 {
			t.Errorf("Expected height 64, got %d", result.Height)
		}
		
		if result.Channels != 3 {
			t.Errorf("Expected 3 channels, got %d", result.Channels)
		}
		
		expectedDataLen := 3 * 64 * 64
		if len(result.Data) != expectedDataLen {
			t.Errorf("Expected data length %d, got %d", expectedDataLen, len(result.Data))
		}
		
		// Verify data is normalized [0, 1]
		for i, val := range result.Data {
			if val < 0 || val > 1 {
				t.Errorf("Value at index %d (%f) not in range [0, 1]", i, val)
			}
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Errorf("Invalid value at index %d: %f", i, val)
			}
		}
		
		// Verify CHW format: check that channels are separated
		// R channel: indices 0 to 64*64-1
		// G channel: indices 64*64 to 2*64*64-1  
		// B channel: indices 2*64*64 to 3*64*64-1
		pixelIdx := 32*64 + 32 // Middle pixel
		rVal := result.Data[0*64*64 + pixelIdx]
		gVal := result.Data[1*64*64 + pixelIdx]
		bVal := result.Data[2*64*64 + pixelIdx]
		
		// Values should be reasonable (not zero due to gradient pattern)
		if rVal == 0 && gVal == 0 && bVal == 0 {
			t.Error("Expected non-zero color values in middle of gradient image")
		}
	})
	
	t.Run("BufferReuse", func(t *testing.T) {
		// Process two images to test buffer reuse
		baseColor1 := color.RGBA{255, 0, 0, 255}
		jpegData1, err := createMockJPEGImage(50, 50, baseColor1)
		if err != nil {
			t.Fatalf("Failed to create first mock image: %v", err)
		}
		
		// First processing should create buffers
		reader1 := bytes.NewReader(jpegData1)
		result1, err := processor.DecodeAndPreprocess(reader1)
		if err != nil {
			t.Fatalf("Unexpected error on first processing: %v", err)
		}
		
		// Buffers should now exist
		if processor.tempImageBuffer == nil {
			t.Error("Expected tempImageBuffer to be created")
		}
		if processor.processBuffer == nil {
			t.Error("Expected processBuffer to be created")
		}
		
		// Second processing should reuse buffers
		baseColor2 := color.RGBA{0, 255, 0, 255}
		jpegData2, err := createMockJPEGImage(80, 80, baseColor2)
		if err != nil {
			t.Fatalf("Failed to create second mock image: %v", err)
		}
		
		reader2 := bytes.NewReader(jpegData2)
		result2, err := processor.DecodeAndPreprocess(reader2)
		if err != nil {
			t.Fatalf("Unexpected error on second processing: %v", err)
		}
		
		// Results should be different (different source colors)
		if len(result1.Data) != len(result2.Data) {
			t.Error("Results should have same data length")
		}
		
		// At least some values should be different
		different := false
		for i := range result1.Data {
			if math.Abs(float64(result1.Data[i] - result2.Data[i])) > 0.01 {
				different = true
				break
			}
		}
		if !different {
			t.Error("Expected different results from different source images")
		}
	})
	
	t.Run("DifferentTargetSizes", func(t *testing.T) {
		// Test with different target sizes
		sizes := []int{32, 64, 128, 256}
		baseColor := color.RGBA{128, 128, 128, 255}
		jpegData, err := createMockJPEGImage(200, 200, baseColor)
		if err != nil {
			t.Fatalf("Failed to create mock image: %v", err)
		}
		
		for _, size := range sizes {
			proc := NewImageProcessor(size)
			reader := bytes.NewReader(jpegData)
			result, err := proc.DecodeAndPreprocess(reader)
			if err != nil {
				t.Errorf("Error processing with size %d: %v", size, err)
				continue
			}
			
			if result.Width != size || result.Height != size {
				t.Errorf("Expected size %d×%d, got %d×%d", size, size, result.Width, result.Height)
			}
			
			expectedLen := 3 * size * size
			if len(result.Data) != expectedLen {
				t.Errorf("Expected data length %d for size %d, got %d", expectedLen, size, len(result.Data))
			}
		}
	})
	
	t.Run("InvalidJPEGData", func(t *testing.T) {
		// Test with invalid JPEG data
		invalidData := []byte("not a jpeg image")
		reader := bytes.NewReader(invalidData)
		
		_, err := processor.DecodeAndPreprocess(reader)
		if err == nil {
			t.Error("Expected error for invalid JPEG data")
		}
		
		if !strings.Contains(err.Error(), "failed to decode JPEG") {
			t.Errorf("Expected JPEG decode error, got: %v", err)
		}
	})
	
	t.Run("EmptyReader", func(t *testing.T) {
		// Test with empty reader
		reader := bytes.NewReader([]byte{})
		
		_, err := processor.DecodeAndPreprocess(reader)
		if err == nil {
			t.Error("Expected error for empty reader")
		}
	})
}

// TestImageProcessorConcurrency tests thread safety
func TestImageProcessorConcurrency(t *testing.T) {
	processor := NewImageProcessor(64)
	
	// Create test image data
	baseColor := color.RGBA{100, 150, 200, 255}
	jpegData, err := createMockJPEGImage(100, 100, baseColor)
	if err != nil {
		t.Fatalf("Failed to create mock image: %v", err)
	}
	
	numGoroutines := 10
	numProcessingsPerGoroutine := 20
	
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numProcessingsPerGoroutine)
	
	// Start concurrent processing
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numProcessingsPerGoroutine; j++ {
				reader := bytes.NewReader(jpegData)
				result, err := processor.DecodeAndPreprocess(reader)
				
				if err != nil {
					errors <- fmt.Errorf("goroutine %d, iteration %d: %v", id, j, err)
					return
				}
				
				// Basic validation
				if result == nil {
					errors <- fmt.Errorf("goroutine %d, iteration %d: nil result", id, j)
					return
				}
				
				if len(result.Data) != 3*64*64 {
					errors <- fmt.Errorf("goroutine %d, iteration %d: wrong data length", id, j)
					return
				}
			}
		}(i)
	}
	
	// Wait for completion with timeout
	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()
	
	select {
	case <-done:
		// Success
	case <-time.After(10 * time.Second):
		t.Fatal("Timeout waiting for concurrent processing")
	}
	
	// Check for errors
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

// TestPreprocessBatch tests batch processing functionality
func TestPreprocessBatch(t *testing.T) {
	// Create temporary directory with test images
	tempDir := t.TempDir()
	
	// Create test images
	testColors := []color.RGBA{
		{255, 0, 0, 255},   // Red
		{0, 255, 0, 255},   // Green
		{0, 0, 255, 255},   // Blue
		{255, 255, 0, 255}, // Yellow
		{255, 0, 255, 255}, // Magenta
	}
	
	imagePaths := make([]string, len(testColors))
	for i, color := range testColors {
		path := filepath.Join(tempDir, fmt.Sprintf("test_image_%d.jpg", i))
		err := createTestJPEGFile(path, 100, 100, color)
		if err != nil {
			t.Fatalf("Failed to create test image %d: %v", i, err)
		}
		imagePaths[i] = path
	}
	
	t.Run("ValidBatch", func(t *testing.T) {
		results, err := PreprocessBatch(imagePaths, 64, 2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if len(results) != len(imagePaths) {
			t.Errorf("Expected %d results, got %d", len(imagePaths), len(results))
		}
		
		// Verify each result
		for i, result := range results {
			if result == nil {
				t.Errorf("Result %d is nil", i)
				continue
			}
			
			if result.Width != 64 || result.Height != 64 {
				t.Errorf("Result %d: expected size 64×64, got %d×%d", i, result.Width, result.Height)
			}
			
			if result.Channels != 3 {
				t.Errorf("Result %d: expected 3 channels, got %d", i, result.Channels)
			}
			
			if len(result.Data) != 3*64*64 {
				t.Errorf("Result %d: expected data length %d, got %d", i, 3*64*64, len(result.Data))
			}
		}
		
		// Verify results are different (different colored source images)
		if len(results) >= 2 {
			same := true
			for i := range results[0].Data {
				if math.Abs(float64(results[0].Data[i] - results[1].Data[i])) > 0.01 {
					same = false
					break
				}
			}
			if same {
				t.Error("Expected different results from different colored images")
			}
		}
	})
	
	t.Run("DifferentWorkerCounts", func(t *testing.T) {
		workerCounts := []int{1, 2, 4, 8}
		
		for _, workers := range workerCounts {
			results, err := PreprocessBatch(imagePaths, 32, workers)
			if err != nil {
				t.Errorf("Error with %d workers: %v", workers, err)
				continue
			}
			
			if len(results) != len(imagePaths) {
				t.Errorf("With %d workers: expected %d results, got %d", workers, len(imagePaths), len(results))
			}
		}
	})
	
	t.Run("ZeroWorkers", func(t *testing.T) {
		// Should default to 1 worker
		results, err := PreprocessBatch(imagePaths[:2], 32, 0)
		if err != nil {
			t.Fatalf("Unexpected error with 0 workers: %v", err)
		}
		
		if len(results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results))
		}
	})
	
	t.Run("NonexistentFiles", func(t *testing.T) {
		invalidPaths := []string{
			"/nonexistent/path1.jpg",
			"/nonexistent/path2.jpg",
		}
		
		_, err := PreprocessBatch(invalidPaths, 32, 1)
		if err == nil {
			t.Error("Expected error for nonexistent files")
		}
		
		if !strings.Contains(err.Error(), "failed to process image") {
			t.Errorf("Expected processing error, got: %v", err)
		}
	})
	
	t.Run("MixedValidInvalidFiles", func(t *testing.T) {
		mixedPaths := []string{
			imagePaths[0], // Valid
			"/nonexistent/path.jpg", // Invalid
		}
		
		_, err := PreprocessBatch(mixedPaths, 32, 1)
		if err == nil {
			t.Error("Expected error when some files are invalid")
		}
	})
	
	t.Run("EmptyBatch", func(t *testing.T) {
		results, err := PreprocessBatch([]string{}, 32, 1)
		if err != nil {
			t.Errorf("Unexpected error for empty batch: %v", err)
		}
		
		if len(results) != 0 {
			t.Errorf("Expected 0 results for empty batch, got %d", len(results))
		}
	})
}

// TestProcessedImageStructure tests the ProcessedImage struct
func TestProcessedImageStructure(t *testing.T) {
	// Create a ProcessedImage directly
	data := make([]float32, 3*32*32)
	for i := range data {
		data[i] = float32(i) / float32(len(data)) // Normalized test data
	}
	
	img := &ProcessedImage{
		Data:     data,
		Width:    32,
		Height:   32,
		Channels: 3,
	}
	
	// Test basic properties
	if img.Width != 32 {
		t.Errorf("Expected width 32, got %d", img.Width)
	}
	
	if img.Height != 32 {
		t.Errorf("Expected height 32, got %d", img.Height)
	}
	
	if img.Channels != 3 {
		t.Errorf("Expected 3 channels, got %d", img.Channels)
	}
	
	expectedLen := img.Width * img.Height * img.Channels
	if len(img.Data) != expectedLen {
		t.Errorf("Expected data length %d, got %d", expectedLen, len(img.Data))
	}
	
	// Test that data is accessible
	if len(img.Data) > 0 && img.Data[0] != 0 {
		// First element should be 0 (normalized from 0/len)
		if img.Data[0] != 0 {
			t.Errorf("Expected first element to be 0, got %f", img.Data[0])
		}
	}
}

// TestImageProcessorEdgeCases tests various edge cases
func TestImageProcessorEdgeCases(t *testing.T) {
	t.Run("VerySmallImage", func(t *testing.T) {
		processor := NewImageProcessor(64)
		
		// Create 1x1 image
		baseColor := color.RGBA{255, 128, 64, 255}
		jpegData, err := createMockJPEGImage(1, 1, baseColor)
		if err != nil {
			t.Fatalf("Failed to create 1x1 image: %v", err)
		}
		
		reader := bytes.NewReader(jpegData)
		result, err := processor.DecodeAndPreprocess(reader)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Should still produce 64x64 output
		if result.Width != 64 || result.Height != 64 {
			t.Errorf("Expected 64×64 output, got %d×%d", result.Width, result.Height)
		}
	})
	
	t.Run("VeryLargeImage", func(t *testing.T) {
		processor := NewImageProcessor(32)
		
		// Create large image
		baseColor := color.RGBA{100, 200, 150, 255}
		jpegData, err := createMockJPEGImage(1000, 1000, baseColor)
		if err != nil {
			t.Fatalf("Failed to create large image: %v", err)
		}
		
		reader := bytes.NewReader(jpegData)
		result, err := processor.DecodeAndPreprocess(reader)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Should produce 32x32 output
		if result.Width != 32 || result.Height != 32 {
			t.Errorf("Expected 32×32 output, got %d×%d", result.Width, result.Height)
		}
	})
	
	t.Run("SquareVsRectangular", func(t *testing.T) {
		processor := NewImageProcessor(64)
		baseColor := color.RGBA{255, 255, 255, 255}
		
		// Test square image
		squareData, err := createMockJPEGImage(100, 100, baseColor)
		if err != nil {
			t.Fatalf("Failed to create square image: %v", err)
		}
		
		// Test rectangular image
		rectData, err := createMockJPEGImage(200, 100, baseColor)
		if err != nil {
			t.Fatalf("Failed to create rectangular image: %v", err)
		}
		
		// Process both
		squareResult, err := processor.DecodeAndPreprocess(bytes.NewReader(squareData))
		if err != nil {
			t.Fatalf("Error processing square image: %v", err)
		}
		
		rectResult, err := processor.DecodeAndPreprocess(bytes.NewReader(rectData))
		if err != nil {
			t.Fatalf("Error processing rectangular image: %v", err)
		}
		
		// Both should produce same output dimensions
		if squareResult.Width != rectResult.Width || squareResult.Height != rectResult.Height {
			t.Error("Square and rectangular images should produce same output dimensions")
		}
		
		// Both should be 64x64
		if squareResult.Width != 64 || rectResult.Width != 64 {
			t.Error("Expected both results to be 64x64")
		}
	})
}

// BenchmarkImageProcessorDecodeAndPreprocess benchmarks single image processing
func BenchmarkImageProcessorDecodeAndPreprocess(b *testing.B) {
	processor := NewImageProcessor(224)
	
	// Create test image data
	baseColor := color.RGBA{128, 128, 128, 255}
	jpegData, err := createMockJPEGImage(300, 300, baseColor)
	if err != nil {
		b.Fatalf("Failed to create test image: %v", err)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := bytes.NewReader(jpegData)
		_, err := processor.DecodeAndPreprocess(reader)
		if err != nil {
			b.Fatalf("Processing error: %v", err)
		}
	}
}

// BenchmarkPreprocessBatch benchmarks batch processing
func BenchmarkPreprocessBatch(b *testing.B) {
	// Create temporary test files
	tempDir := b.TempDir()
	numImages := 10
	imagePaths := make([]string, numImages)
	
	baseColor := color.RGBA{128, 128, 128, 255}
	for i := 0; i < numImages; i++ {
		path := filepath.Join(tempDir, fmt.Sprintf("image_%d.jpg", i))
		err := createTestJPEGFile(path, 200, 200, baseColor)
		if err != nil {
			b.Fatalf("Failed to create test image %d: %v", i, err)
		}
		imagePaths[i] = path
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := PreprocessBatch(imagePaths, 224, 4)
		if err != nil {
			b.Fatalf("Batch processing error: %v", err)
		}
	}
}

// BenchmarkImageProcessorBufferReuse benchmarks buffer reuse efficiency
func BenchmarkImageProcessorBufferReuse(b *testing.B) {
	processor := NewImageProcessor(128)
	
	// Create test images of different sizes to test buffer reuse
	testImages := make([][]byte, 3)
	sizes := []int{100, 150, 200}
	baseColor := color.RGBA{100, 150, 200, 255}
	
	for i, size := range sizes {
		data, err := createMockJPEGImage(size, size, baseColor)
		if err != nil {
			b.Fatalf("Failed to create test image %d: %v", i, err)
		}
		testImages[i] = data
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Cycle through different image sizes to test buffer adaptation
		imageIdx := i % len(testImages)
		reader := bytes.NewReader(testImages[imageIdx])
		_, err := processor.DecodeAndPreprocess(reader)
		if err != nil {
			b.Fatalf("Processing error: %v", err)
		}
	}
}