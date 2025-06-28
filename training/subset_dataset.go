package training

import (
	"fmt"

	"github.com/tsawler/go-metal/tensor"
)

// SubsetDataset allows training on a limited number of samples from an underlying dataset.
type SubsetDataset struct {
	originalDataset Dataset
	limit           int
}

// NewSubsetDataset creates a new SubsetDataset that wraps an existing dataset
// and limits the number of samples it exposes.
func NewSubsetDataset(original Dataset, limit int) (*SubsetDataset, error) {
	if limit < 0 {
		return nil, fmt.Errorf("limit cannot be negative")
	}
	if limit > original.Len() {
		limit = original.Len() // Adjust limit if it's greater than the original dataset's length
	}
	return &SubsetDataset{
		originalDataset: original,
		limit:           limit,
	}, nil
}

// Len returns the number of samples in the subset, which is the minimum
// of the original dataset's length and the specified limit.
func (sd *SubsetDataset) Len() int {
	return sd.limit
}

// Get returns a sample at the given index from the original dataset.
// It assumes the index is within the bounds of the subset.
func (sd *SubsetDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error) {
	if idx < 0 || idx >= sd.limit {
		return nil, nil, fmt.Errorf("index out of bounds for subset: %d (limit: %d)", idx, sd.limit)
	}
	return sd.originalDataset.Get(idx)
}
