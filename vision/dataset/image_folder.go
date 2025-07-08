package dataset

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// ImageFolderDataset represents a dataset loaded from a directory structure
// where each subdirectory represents a class
type ImageFolderDataset struct {
	imagePaths []string
	labels     []int
	classNames []string
	classToIdx map[string]int
}

// NewImageFolderDataset creates a dataset from a directory structure
func NewImageFolderDataset(root string, extensions []string) (*ImageFolderDataset, error) {
	if len(extensions) == 0 {
		extensions = []string{".jpg", ".jpeg", ".png", ".bmp"}
	}

	dataset := &ImageFolderDataset{
		classToIdx: make(map[string]int),
	}

	// Find all classes (subdirectories)
	classes, err := filepath.Glob(filepath.Join(root, "*"))
	if err != nil {
		return nil, fmt.Errorf("failed to list classes: %w", err)
	}

	classIdx := 0
	for _, classPath := range classes {
		info, err := os.Stat(classPath)
		if err != nil || !info.IsDir() {
			continue
		}

		className := filepath.Base(classPath)
		dataset.classNames = append(dataset.classNames, className)
		dataset.classToIdx[className] = classIdx

		// Find all images in this class
		for _, ext := range extensions {
			pattern := filepath.Join(classPath, "*"+ext)
			files, err := filepath.Glob(pattern)
			if err != nil {
				continue
			}

			for _, file := range files {
				dataset.imagePaths = append(dataset.imagePaths, file)
				dataset.labels = append(dataset.labels, classIdx)
			}
		}

		classIdx++
	}

	if len(dataset.imagePaths) == 0 {
		return nil, fmt.Errorf("no images found in %s", root)
	}

	return dataset, nil
}

// Len returns the number of items in the dataset
func (d *ImageFolderDataset) Len() int {
	return len(d.imagePaths)
}

// GetItem returns the image path and label at the given index
func (d *ImageFolderDataset) GetItem(index int) (string, int, error) {
	if index < 0 || index >= len(d.imagePaths) {
		return "", 0, fmt.Errorf("index %d out of range [0, %d)", index, len(d.imagePaths))
	}
	return d.imagePaths[index], d.labels[index], nil
}

// NumClasses returns the number of classes
func (d *ImageFolderDataset) NumClasses() int {
	return len(d.classNames)
}

// ClassNames returns the list of class names
func (d *ImageFolderDataset) ClassNames() []string {
	return d.classNames
}

// ClassDistribution returns the distribution of samples per class
func (d *ImageFolderDataset) ClassDistribution() map[string]int {
	dist := make(map[string]int)
	for _, label := range d.labels {
		className := d.classNames[label]
		dist[className]++
	}
	return dist
}

// Split splits the dataset into train and validation sets
func (d *ImageFolderDataset) Split(trainRatio float64, shuffle bool) (*ImageFolderDataset, *ImageFolderDataset) {
	n := len(d.imagePaths)
	trainSize := int(float64(n) * trainRatio)

	// Create indices
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	if shuffle {
		rand.Shuffle(n, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// Create train dataset
	trainDataset := &ImageFolderDataset{
		imagePaths: make([]string, trainSize),
		labels:     make([]int, trainSize),
		classNames: d.classNames,
		classToIdx: d.classToIdx,
	}

	for i := 0; i < trainSize; i++ {
		idx := indices[i]
		trainDataset.imagePaths[i] = d.imagePaths[idx]
		trainDataset.labels[i] = d.labels[idx]
	}

	// Create validation dataset
	valSize := n - trainSize
	valDataset := &ImageFolderDataset{
		imagePaths: make([]string, valSize),
		labels:     make([]int, valSize),
		classNames: d.classNames,
		classToIdx: d.classToIdx,
	}

	for i := 0; i < valSize; i++ {
		idx := indices[trainSize+i]
		valDataset.imagePaths[i] = d.imagePaths[idx]
		valDataset.labels[i] = d.labels[idx]
	}

	return trainDataset, valDataset
}

// Subset creates a subset of the dataset with the specified indices
func (d *ImageFolderDataset) Subset(indices []int) *ImageFolderDataset {
	subset := &ImageFolderDataset{
		imagePaths: make([]string, len(indices)),
		labels:     make([]int, len(indices)),
		classNames: d.classNames,
		classToIdx: d.classToIdx,
	}

	for i, idx := range indices {
		subset.imagePaths[i] = d.imagePaths[idx]
		subset.labels[i] = d.labels[idx]
	}

	return subset
}

// FilterByClass creates a new dataset containing only samples from specified classes
func (d *ImageFolderDataset) FilterByClass(classNames []string) *ImageFolderDataset {
	// Create a set of valid class indices
	validClasses := make(map[int]bool)
	for _, className := range classNames {
		if idx, exists := d.classToIdx[className]; exists {
			validClasses[idx] = true
		}
	}

	// Filter samples
	var filteredPaths []string
	var filteredLabels []int

	for i, label := range d.labels {
		if validClasses[label] {
			filteredPaths = append(filteredPaths, d.imagePaths[i])
			filteredLabels = append(filteredLabels, label)
		}
	}

	return &ImageFolderDataset{
		imagePaths: filteredPaths,
		labels:     filteredLabels,
		classNames: d.classNames,
		classToIdx: d.classToIdx,
	}
}

// String returns a string representation of the dataset
func (d *ImageFolderDataset) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("ImageFolderDataset: %d samples, %d classes\n", len(d.imagePaths), len(d.classNames)))
	sb.WriteString("Class distribution:\n")
	
	dist := d.ClassDistribution()
	for _, className := range d.classNames {
		count := dist[className]
		sb.WriteString(fmt.Sprintf("  %s: %d samples\n", className, count))
	}
	
	return sb.String()
}