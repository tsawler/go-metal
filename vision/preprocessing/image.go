package preprocessing

import (
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"os"
	"sync"
)

// ImageProcessor provides high-performance image preprocessing with buffer reuse
type ImageProcessor struct {
	mu              sync.Mutex
	tempImageBuffer *image.RGBA
	processBuffer   []float32
	targetSize      int
}

// NewImageProcessor creates a new image processor with the specified target size
func NewImageProcessor(targetSize int) *ImageProcessor {
	return &ImageProcessor{
		targetSize: targetSize,
	}
}

// ProcessedImage represents a preprocessed image ready for neural network input
type ProcessedImage struct {
	Data     []float32
	Width    int
	Height   int
	Channels int
}

// DecodeAndPreprocess decodes a JPEG image and preprocesses it for neural network input
// Returns data in CHW format (channels, height, width) normalized to [0, 1]
func (p *ImageProcessor) DecodeAndPreprocess(reader io.Reader) (*ProcessedImage, error) {
	// Decode JPEG
	img, err := jpeg.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode JPEG: %w", err)
	}

	// Convert to RGBA if needed
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	p.mu.Lock()
	defer p.mu.Unlock()

	// Reuse image buffer
	if p.tempImageBuffer == nil || p.tempImageBuffer.Bounds().Dx() != p.targetSize || p.tempImageBuffer.Bounds().Dy() != p.targetSize {
		p.tempImageBuffer = image.NewRGBA(image.Rect(0, 0, p.targetSize, p.targetSize))
	}
	targetImg := p.tempImageBuffer

	// Simple center crop/resize
	scaleX := float64(width) / float64(p.targetSize)
	scaleY := float64(height) / float64(p.targetSize)

	for y := 0; y < p.targetSize; y++ {
		for x := 0; x < p.targetSize; x++ {
			srcX := int(float64(x) * scaleX)
			srcY := int(float64(y) * scaleY)

			if srcX >= width {
				srcX = width - 1
			}
			if srcY >= height {
				srcY = height - 1
			}

			targetImg.Set(x, y, img.At(srcX, srcY))
		}
	}

	// Reuse data buffer
	requiredSize := 3 * p.targetSize * p.targetSize
	if len(p.processBuffer) < requiredSize {
		p.processBuffer = make([]float32, requiredSize)
	}
	data := p.processBuffer[:requiredSize]

	// Convert to float32 RGB data in CHW format
	for y := 0; y < p.targetSize; y++ {
		for x := 0; x < p.targetSize; x++ {
			r, g, b, _ := targetImg.At(x, y).RGBA()

			// Normalize to [0, 1]
			idx := y*p.targetSize + x
			rVal := float32(r) / 65535.0
			gVal := float32(g) / 65535.0
			bVal := float32(b) / 65535.0

			// Validate for NaN/Inf values
			if rVal != rVal || rVal < 0 || rVal > 1 {
				rVal = 0.0
			}
			if gVal != gVal || gVal < 0 || gVal > 1 {
				gVal = 0.0
			}
			if bVal != bVal || bVal < 0 || bVal > 1 {
				bVal = 0.0
			}

			// Store in CHW format
			data[0*p.targetSize*p.targetSize+idx] = rVal // R channel
			data[1*p.targetSize*p.targetSize+idx] = gVal // G channel
			data[2*p.targetSize*p.targetSize+idx] = bVal // B channel
		}
	}

	// Create a copy since we're returning a slice of the reusable buffer
	result := make([]float32, len(data))
	copy(result, data)

	return &ProcessedImage{
		Data:     result,
		Width:    p.targetSize,
		Height:   p.targetSize,
		Channels: 3,
	}, nil
}

// PreprocessBatch preprocesses multiple images concurrently
func PreprocessBatch(imagePaths []string, targetSize int, maxWorkers int) ([]*ProcessedImage, error) {
	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	results := make([]*ProcessedImage, len(imagePaths))
	errors := make([]error, len(imagePaths))

	// Create worker pool
	type job struct {
		index int
		path  string
	}

	jobs := make(chan job, len(imagePaths))
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			processor := NewImageProcessor(targetSize)

			for j := range jobs {
				file, err := os.Open(j.path)
				if err != nil {
					errors[j.index] = err
					continue
				}

				img, err := processor.DecodeAndPreprocess(file)
				file.Close()

				if err != nil {
					errors[j.index] = err
				} else {
					results[j.index] = img
				}
			}
		}()
	}

	// Submit jobs
	for i, path := range imagePaths {
		jobs <- job{index: i, path: path}
	}
	close(jobs)

	// Wait for completion
	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("failed to process image %d: %w", i, err)
		}
	}

	return results, nil
}