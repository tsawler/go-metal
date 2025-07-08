package dataset

import (
	"fmt"
	"path/filepath"
)

// CatsDogsDataset is a specialized dataset for the cats vs dogs classification task
type CatsDogsDataset struct {
	*ImageFolderDataset
}

// NewCatsDogsDataset creates a new cats and dogs dataset from the standard directory structure
func NewCatsDogsDataset(dataDir string, maxSamplesPerClass int) (*CatsDogsDataset, error) {
	// Load cat images
	catDir := filepath.Join(dataDir, "cat")
	catFiles, err := filepath.Glob(filepath.Join(catDir, "*.jpg"))
	if err != nil {
		return nil, fmt.Errorf("failed to load cat images: %w", err)
	}

	// Load dog images  
	dogDir := filepath.Join(dataDir, "dog")
	dogFiles, err := filepath.Glob(filepath.Join(dogDir, "*.jpg"))
	if err != nil {
		return nil, fmt.Errorf("failed to load dog images: %w", err)
	}

	// Limit samples if specified
	if maxSamplesPerClass > 0 {
		if len(catFiles) > maxSamplesPerClass {
			catFiles = catFiles[:maxSamplesPerClass]
		}
		if len(dogFiles) > maxSamplesPerClass {
			dogFiles = dogFiles[:maxSamplesPerClass]
		}
	}

	// Create dataset manually
	dataset := &ImageFolderDataset{
		imagePaths: make([]string, 0, len(catFiles)+len(dogFiles)),
		labels:     make([]int, 0, len(catFiles)+len(dogFiles)),
		classNames: []string{"cat", "dog"},
		classToIdx: map[string]int{"cat": 0, "dog": 1},
	}

	// Add cat images (label = 0)
	for _, path := range catFiles {
		dataset.imagePaths = append(dataset.imagePaths, path)
		dataset.labels = append(dataset.labels, 0)
	}

	// Add dog images (label = 1)
	for _, path := range dogFiles {
		dataset.imagePaths = append(dataset.imagePaths, path)
		dataset.labels = append(dataset.labels, 1)
	}

	if len(dataset.imagePaths) == 0 {
		return nil, fmt.Errorf("no images found in %s", dataDir)
	}

	return &CatsDogsDataset{ImageFolderDataset: dataset}, nil
}

// Summary returns a summary of the dataset
func (d *CatsDogsDataset) Summary() string {
	dist := d.ClassDistribution()
	return fmt.Sprintf("Cats & Dogs Dataset: %d total images (%d cats, %d dogs)",
		d.Len(), dist["cat"], dist["dog"])
}