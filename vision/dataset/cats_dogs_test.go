package dataset

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testHelper interface for both *testing.T and *testing.B
type testHelper interface {
	TempDir() string
	Fatalf(format string, args ...interface{})
}

// createCatsDogsTestDataset creates a temporary cats vs dogs dataset structure
func createCatsDogsTestDataset(t testHelper, catsCount, dogsCount int) string {
	tempDir := t.TempDir()

	// Create cat directory
	catDir := filepath.Join(tempDir, "cat")
	if err := os.MkdirAll(catDir, 0755); err != nil {
		t.Fatalf("Failed to create cat directory: %v", err)
	}

	// Create cat images
	for i := 0; i < catsCount; i++ {
		imagePath := filepath.Join(catDir, fmt.Sprintf("cat_%d.jpg", i))
		if err := createMockImageFileForCatsDogs(imagePath); err != nil {
			t.Fatalf("Failed to create cat image: %v", err)
		}
	}

	// Create dog directory
	dogDir := filepath.Join(tempDir, "dog")
	if err := os.MkdirAll(dogDir, 0755); err != nil {
		t.Fatalf("Failed to create dog directory: %v", err)
	}

	// Create dog images
	for i := 0; i < dogsCount; i++ {
		imagePath := filepath.Join(dogDir, fmt.Sprintf("dog_%d.jpg", i))
		if err := createMockImageFileForCatsDogs(imagePath); err != nil {
			t.Fatalf("Failed to create dog image: %v", err)
		}
	}

	return tempDir
}

// TestNewCatsDogsDataset tests cats and dogs dataset creation
func TestNewCatsDogsDataset(t *testing.T) {
	t.Run("ValidDataset", func(t *testing.T) {
		catsCount := 10
		dogsCount := 15
		tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

		dataset, err := NewCatsDogsDataset(tempDir, 0) // No limit
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Check total count
		expectedTotal := catsCount + dogsCount
		if dataset.Len() != expectedTotal {
			t.Errorf("Expected %d images, got %d", expectedTotal, dataset.Len())
		}

		// Check classes
		if dataset.NumClasses() != 2 {
			t.Errorf("Expected 2 classes, got %d", dataset.NumClasses())
		}

		classNames := dataset.ClassNames()
		if len(classNames) != 2 {
			t.Fatalf("Expected 2 class names, got %d", len(classNames))
		}

		// Verify class names are "cat" and "dog"
		expectedClasses := map[string]bool{"cat": true, "dog": true}
		for _, className := range classNames {
			if !expectedClasses[className] {
				t.Errorf("Unexpected class name: %s", className)
			}
		}

		// Check class distribution
		dist := dataset.ClassDistribution()
		if dist["cat"] != catsCount {
			t.Errorf("Expected %d cats, got %d", catsCount, dist["cat"])
		}
		if dist["dog"] != dogsCount {
			t.Errorf("Expected %d dogs, got %d", dogsCount, dist["dog"])
		}

		// Verify class-to-index mapping
		if dataset.classToIdx["cat"] != 0 {
			t.Errorf("Expected cat index 0, got %d", dataset.classToIdx["cat"])
		}
		if dataset.classToIdx["dog"] != 1 {
			t.Errorf("Expected dog index 1, got %d", dataset.classToIdx["dog"])
		}
	})

	t.Run("WithMaxSamplesPerClass", func(t *testing.T) {
		catsCount := 20
		dogsCount := 25
		maxSamplesPerClass := 10
		tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

		dataset, err := NewCatsDogsDataset(tempDir, maxSamplesPerClass)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should be limited to maxSamplesPerClass for each class
		expectedTotal := maxSamplesPerClass * 2
		if dataset.Len() != expectedTotal {
			t.Errorf("Expected %d images (limited), got %d", expectedTotal, dataset.Len())
		}

		dist := dataset.ClassDistribution()
		if dist["cat"] != maxSamplesPerClass {
			t.Errorf("Expected %d cats (limited), got %d", maxSamplesPerClass, dist["cat"])
		}
		if dist["dog"] != maxSamplesPerClass {
			t.Errorf("Expected %d dogs (limited), got %d", maxSamplesPerClass, dist["dog"])
		}
	})

	t.Run("WithMaxSamplesLargerThanAvailable", func(t *testing.T) {
		catsCount := 5
		dogsCount := 7
		maxSamplesPerClass := 10 // Larger than available
		tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

		dataset, err := NewCatsDogsDataset(tempDir, maxSamplesPerClass)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should use all available images
		expectedTotal := catsCount + dogsCount
		if dataset.Len() != expectedTotal {
			t.Errorf("Expected %d images (all available), got %d", expectedTotal, dataset.Len())
		}

		dist := dataset.ClassDistribution()
		if dist["cat"] != catsCount {
			t.Errorf("Expected %d cats (all available), got %d", catsCount, dist["cat"])
		}
		if dist["dog"] != dogsCount {
			t.Errorf("Expected %d dogs (all available), got %d", dogsCount, dist["dog"])
		}
	})

	t.Run("EmptyDataset", func(t *testing.T) {
		tempDir := createCatsDogsTestDataset(t, 0, 0) // No images

		_, err := NewCatsDogsDataset(tempDir, 0)
		if err == nil {
			t.Error("Expected error for empty dataset")
		}

		if !strings.Contains(err.Error(), "no images found") {
			t.Errorf("Expected 'no images found' error, got: %v", err)
		}
	})

	t.Run("MissingCatDirectory", func(t *testing.T) {
		tempDir := t.TempDir()

		// Only create dog directory
		dogDir := filepath.Join(tempDir, "dog")
		if err := os.MkdirAll(dogDir, 0755); err != nil {
			t.Fatalf("Failed to create dog directory: %v", err)
		}

		for i := 0; i < 5; i++ {
			imagePath := filepath.Join(dogDir, fmt.Sprintf("dog_%d.jpg", i))
			if err := createMockImageFileForCatsDogs(imagePath); err != nil {
				t.Fatalf("Failed to create dog image: %v", err)
			}
		}

		dataset, err := NewCatsDogsDataset(tempDir, 0)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should still work with only dogs
		if dataset.Len() != 5 {
			t.Errorf("Expected 5 images, got %d", dataset.Len())
		}

		dist := dataset.ClassDistribution()
		if dist["cat"] != 0 {
			t.Errorf("Expected 0 cats, got %d", dist["cat"])
		}
		if dist["dog"] != 5 {
			t.Errorf("Expected 5 dogs, got %d", dist["dog"])
		}
	})

	t.Run("NonexistentDirectory", func(t *testing.T) {
		_, err := NewCatsDogsDataset("/nonexistent/path", 0)
		if err == nil {
			t.Error("Expected error for nonexistent directory")
		}
	})

	t.Run("OnlyDirectoriesNoImages", func(t *testing.T) {
		tempDir := t.TempDir()

		// Create directories but no images
		catDir := filepath.Join(tempDir, "cat")
		dogDir := filepath.Join(tempDir, "dog")
		if err := os.MkdirAll(catDir, 0755); err != nil {
			t.Fatalf("Failed to create cat directory: %v", err)
		}
		if err := os.MkdirAll(dogDir, 0755); err != nil {
			t.Fatalf("Failed to create dog directory: %v", err)
		}

		_, err := NewCatsDogsDataset(tempDir, 0)
		if err == nil {
			t.Error("Expected error when no images found")
		}
	})

	t.Run("MaxSamplesZero", func(t *testing.T) {
		tempDir := createCatsDogsTestDataset(t, 5, 7)

		dataset, err := NewCatsDogsDataset(tempDir, 0)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should use all available images
		if dataset.Len() != 12 {
			t.Errorf("Expected 12 images with maxSamples=0, got %d", dataset.Len())
		}
	})
}

// TestCatsDogsDatasetGetItem tests item retrieval
func TestCatsDogsDatasetGetItem(t *testing.T) {
	catsCount := 3
	dogsCount := 4
	tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

	dataset, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("ValidIndices", func(t *testing.T) {
		catCount := 0
		dogCount := 0

		for i := 0; i < dataset.Len(); i++ {
			imagePath, label, err := dataset.GetItem(i)
			if err != nil {
				t.Errorf("Unexpected error at index %d: %v", i, err)
			}

			if imagePath == "" {
				t.Errorf("Empty image path at index %d", i)
			}

			// Label should be 0 (cat) or 1 (dog)
			if label != 0 && label != 1 {
				t.Errorf("Invalid label %d at index %d", label, i)
			}

			// Count labels
			if label == 0 {
				catCount++
			} else {
				dogCount++
			}

			// Verify the path contains the expected class name
			if label == 0 && !strings.Contains(imagePath, "cat") {
				t.Errorf("Expected cat path at index %d with label 0, got: %s", i, imagePath)
			}
			if label == 1 && !strings.Contains(imagePath, "dog") {
				t.Errorf("Expected dog path at index %d with label 1, got: %s", i, imagePath)
			}

			// Verify the file exists
			if _, err := os.Stat(imagePath); err != nil {
				t.Errorf("Image file doesn't exist: %s", imagePath)
			}
		}

		// Verify counts match expected distribution
		if catCount != catsCount {
			t.Errorf("Expected %d cat items, got %d", catsCount, catCount)
		}
		if dogCount != dogsCount {
			t.Errorf("Expected %d dog items, got %d", dogsCount, dogCount)
		}
	})

	t.Run("InvalidIndices", func(t *testing.T) {
		invalidIndices := []int{-1, dataset.Len(), dataset.Len() + 1}

		for _, idx := range invalidIndices {
			_, _, err := dataset.GetItem(idx)
			if err == nil {
				t.Errorf("Expected error for invalid index %d", idx)
			}
		}
	})
}

// TestCatsDogsDatasetSummary tests the summary method
func TestCatsDogsDatasetSummary(t *testing.T) {
	catsCount := 8
	dogsCount := 12
	tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

	dataset, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	summary := dataset.Summary()

	// Check that summary contains expected information
	expectedSubstrings := []string{
		"Cats & Dogs Dataset",
		fmt.Sprintf("%d total images", catsCount+dogsCount),
		fmt.Sprintf("%d cats", catsCount),
		fmt.Sprintf("%d dogs", dogsCount),
	}

	for _, substr := range expectedSubstrings {
		if !strings.Contains(summary, substr) {
			t.Errorf("Expected summary to contain '%s', got: %s", substr, summary)
		}
	}
}

// TestCatsDogsDatasetInherited tests inherited methods from ImageFolderDataset
func TestCatsDogsDatasetInherited(t *testing.T) {
	catsCount := 6
	dogsCount := 8
	tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

	dataset, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("Split", func(t *testing.T) {
		trainDataset, valDataset := dataset.Split(0.7, false)

		totalSize := catsCount + dogsCount
		expectedTrainSize := int(float64(totalSize) * 0.7)
		expectedValSize := totalSize - expectedTrainSize

		if trainDataset.Len() != expectedTrainSize {
			t.Errorf("Expected train size %d, got %d", expectedTrainSize, trainDataset.Len())
		}

		if valDataset.Len() != expectedValSize {
			t.Errorf("Expected validation size %d, got %d", expectedValSize, valDataset.Len())
		}

		// Both splits should maintain class structure
		if trainDataset.NumClasses() != 2 {
			t.Error("Train dataset should have 2 classes")
		}
		if valDataset.NumClasses() != 2 {
			t.Error("Validation dataset should have 2 classes")
		}
	})

	t.Run("Subset", func(t *testing.T) {
		indices := []int{0, 2, 4, 6}
		subset := dataset.Subset(indices)

		if subset.Len() != len(indices) {
			t.Errorf("Expected subset size %d, got %d", len(indices), subset.Len())
		}

		if subset.NumClasses() != 2 {
			t.Error("Subset should have 2 classes")
		}
	})

	t.Run("FilterByClass", func(t *testing.T) {
		// Filter to only cats
		catsOnly := dataset.FilterByClass([]string{"cat"})

		if catsOnly.Len() != catsCount {
			t.Errorf("Expected %d cats in filtered dataset, got %d", catsCount, catsOnly.Len())
		}

		// Verify all items are cats (label 0)
		for i := 0; i < catsOnly.Len(); i++ {
			_, label, err := catsOnly.GetItem(i)
			if err != nil {
				t.Errorf("Error getting item %d: %v", i, err)
			}
			if label != 0 {
				t.Errorf("Expected cat label (0) at index %d, got %d", i, label)
			}
		}

		// Filter to only dogs
		dogsOnly := dataset.FilterByClass([]string{"dog"})

		if dogsOnly.Len() != dogsCount {
			t.Errorf("Expected %d dogs in filtered dataset, got %d", dogsCount, dogsOnly.Len())
		}

		// Filter to both (should be same as original)
		both := dataset.FilterByClass([]string{"cat", "dog"})
		if both.Len() != dataset.Len() {
			t.Errorf("Expected same size when filtering to both classes, got %d vs %d", both.Len(), dataset.Len())
		}
	})

	t.Run("String", func(t *testing.T) {
		str := dataset.String()

		// Should contain information about the dataset
		expectedSubstrings := []string{
			"ImageFolderDataset",
			fmt.Sprintf("%d samples", dataset.Len()),
			"2 classes",
			"cat:",
			"dog:",
		}

		for _, substr := range expectedSubstrings {
			if !strings.Contains(str, substr) {
				t.Errorf("Expected string representation to contain '%s', got: %s", substr, str)
			}
		}
	})
}

// TestCatsDogsDatasetOrderingInvariance tests that cat/dog ordering is consistent
func TestCatsDogsDatasetOrderingInvariance(t *testing.T) {
	catsCount := 5
	dogsCount := 5
	tempDir := createCatsDogsTestDataset(t, catsCount, dogsCount)

	// Create dataset multiple times and verify ordering is consistent
	dataset1, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create first dataset: %v", err)
	}

	dataset2, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create second dataset: %v", err)
	}

	// Both datasets should have same ordering
	if dataset1.Len() != dataset2.Len() {
		t.Error("Datasets should have same length")
	}

	for i := 0; i < dataset1.Len(); i++ {
		path1, label1, _ := dataset1.GetItem(i)
		path2, label2, _ := dataset2.GetItem(i)

		if path1 != path2 {
			t.Errorf("Path mismatch at index %d: %s vs %s", i, path1, path2)
		}

		if label1 != label2 {
			t.Errorf("Label mismatch at index %d: %d vs %d", i, label1, label2)
		}
	}

	// Verify that cats come before dogs (implementation detail)
	// First catsCount items should be cats, next dogsCount should be dogs
	for i := 0; i < catsCount; i++ {
		_, label, _ := dataset1.GetItem(i)
		if label != 0 {
			t.Errorf("Expected cat (label 0) at index %d, got label %d", i, label)
		}
	}

	for i := catsCount; i < catsCount+dogsCount; i++ {
		_, label, _ := dataset1.GetItem(i)
		if label != 1 {
			t.Errorf("Expected dog (label 1) at index %d, got label %d", i, label)
		}
	}
}

// BenchmarkCatsDogsDatasetCreation benchmarks dataset creation
func BenchmarkCatsDogsDatasetCreation(b *testing.B) {
	catsCount := 1000
	dogsCount := 1000

	// Setup test data once
	tempDir := b.TempDir()
	catDir := filepath.Join(tempDir, "cat")
	dogDir := filepath.Join(tempDir, "dog")

	if err := os.MkdirAll(catDir, 0755); err != nil {
		b.Fatalf("Failed to create cat directory: %v", err)
	}
	if err := os.MkdirAll(dogDir, 0755); err != nil {
		b.Fatalf("Failed to create dog directory: %v", err)
	}

	for i := 0; i < catsCount; i++ {
		imagePath := filepath.Join(catDir, fmt.Sprintf("cat_%d.jpg", i))
		if err := createMockImageFileForCatsDogs(imagePath); err != nil {
			b.Fatalf("Failed to create cat image: %v", err)
		}
	}

	for i := 0; i < dogsCount; i++ {
		imagePath := filepath.Join(dogDir, fmt.Sprintf("dog_%d.jpg", i))
		if err := createMockImageFileForCatsDogs(imagePath); err != nil {
			b.Fatalf("Failed to create dog image: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewCatsDogsDataset(tempDir, 0)
		if err != nil {
			b.Fatalf("Failed to create dataset: %v", err)
		}
	}
}

// BenchmarkCatsDogsDatasetGetItem benchmarks item retrieval
func BenchmarkCatsDogsDatasetGetItem(b *testing.B) {
	tempDir := createCatsDogsTestDataset(b, 1000, 1000)

	dataset, err := NewCatsDogsDataset(tempDir, 0)
	if err != nil {
		b.Fatalf("Failed to create dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % dataset.Len()
		_, _, err := dataset.GetItem(idx)
		if err != nil {
			b.Fatalf("Failed to get item: %v", err)
		}
	}
}

// createMockImageFileForCatsDogs creates a simple file to simulate an image
func createMockImageFileForCatsDogs(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write some mock content
	_, err = file.WriteString("mock image content")
	return err
}