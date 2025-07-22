# preprocessing
--
    import "."


## Usage

#### type ImageProcessor

```go
type ImageProcessor struct {
}
```

ImageProcessor provides high-performance image preprocessing with buffer reuse

#### func  NewImageProcessor

```go
func NewImageProcessor(targetSize int) *ImageProcessor
```
NewImageProcessor creates a new image processor with the specified target size

#### func (*ImageProcessor) DecodeAndPreprocess

```go
func (p *ImageProcessor) DecodeAndPreprocess(reader io.Reader) (*ProcessedImage, error)
```
DecodeAndPreprocess decodes an image (JPEG, PNG, etc.) and preprocesses it for
neural network input Returns data in CHW format (channels, height, width)
normalized to [0, 1]

#### type ProcessedImage

```go
type ProcessedImage struct {
	Data     []float32
	Width    int
	Height   int
	Channels int
}
```

ProcessedImage represents a preprocessed image ready for neural network input

#### func  PreprocessBatch

```go
func PreprocessBatch(imagePaths []string, targetSize int, maxWorkers int) ([]*ProcessedImage, error)
```
PreprocessBatch preprocesses multiple images concurrently
