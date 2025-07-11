package dataset

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// createTestDataset creates a temporary directory structure with test images
func createTestDataset(t *testing.T, classes []string, imagesPerClass int) string {
	tempDir := t.TempDir()

	for _, className := range classes {
		classDir := filepath.Join(tempDir, className)
		if err := os.MkdirAll(classDir, 0755); err != nil {
			t.Fatalf("Failed to create class directory %s: %v", classDir, err)
		}

		for i := 0; i < imagesPerClass; i++ {
			imagePath := filepath.Join(classDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				t.Fatalf("Failed to create mock image %s: %v", imagePath, err)
			}
		}
	}

	return tempDir
}

// createMockImageFile creates a simple file to simulate an image
func createMockImageFile(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write some mock content
	_, err = file.WriteString("mock image content")
	return err
}

// TestNewImageFolderDataset tests dataset creation from directory structure
func TestNewImageFolderDataset(t *testing.T) {
	t.Run("ValidDataset", func(t *testing.T) {
		classes := []string{"cat", "dog", "bird"}
		imagesPerClass := 5
		tempDir := createTestDataset(t, classes, imagesPerClass)

		dataset, err := NewImageFolderDataset(tempDir, nil)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expectedTotal := len(classes) * imagesPerClass
		if dataset.Len() != expectedTotal {
			t.Errorf("Expected %d images, got %d", expectedTotal, dataset.Len())
		}

		if dataset.NumClasses() != len(classes) {
			t.Errorf("Expected %d classes, got %d", len(classes), dataset.NumClasses())
		}

		// Check class names (order might differ)
		actualClasses := dataset.ClassNames()
		sort.Strings(classes)
		sort.Strings(actualClasses)

		for i, expected := range classes {
			if actualClasses[i] != expected {
				t.Errorf("Expected class %s, got %s", expected, actualClasses[i])
			}
		}

		// Check class distribution
		dist := dataset.ClassDistribution()
		for _, className := range classes {
			if count, exists := dist[className]; !exists {
				t.Errorf("Class %s missing from distribution", className)
			} else if count != imagesPerClass {
				t.Errorf("Expected %d images for class %s, got %d", imagesPerClass, className, count)
			}
		}
	})

	t.Run("CustomExtensions", func(t *testing.T) {
		tempDir := t.TempDir()
		classDir := filepath.Join(tempDir, "test_class")
		if err := os.MkdirAll(classDir, 0755); err != nil {
			t.Fatalf("Failed to create class directory: %v", err)
		}

		// Create files with different extensions
		extensions := []string{".jpg", ".png", ".bmp", ".gif"}
		for i, ext := range extensions {
			imagePath := filepath.Join(classDir, fmt.Sprintf("image_%d%s", i, ext))
			if err := createMockImageFile(imagePath); err != nil {
				t.Fatalf("Failed to create image: %v", err)
			}
		}

		// Test with only jpg and png
		dataset, err := NewImageFolderDataset(tempDir, []string{".jpg", ".png"})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expectedImages := 2 // Only .jpg and .png
		if dataset.Len() != expectedImages {
			t.Errorf("Expected %d images, got %d", expectedImages, dataset.Len())
		}
	})

	t.Run("EmptyDirectory", func(t *testing.T) {
		tempDir := t.TempDir()

		_, err := NewImageFolderDataset(tempDir, nil)
		if err == nil {
			t.Error("Expected error for empty directory")
		}

		if !strings.Contains(err.Error(), "no images found") {
			t.Errorf("Expected 'no images found' error, got: %v", err)
		}
	})

	t.Run("NonexistentDirectory", func(t *testing.T) {
		_, err := NewImageFolderDataset("/nonexistent/path", nil)
		if err == nil {
			t.Error("Expected error for nonexistent directory")
		}
	})

	t.Run("DirectoryWithNoClasses", func(t *testing.T) {
		tempDir := t.TempDir()

		// Create some files directly in tempDir (not in subdirectories)
		for i := 0; i < 3; i++ {
			imagePath := filepath.Join(tempDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				t.Fatalf("Failed to create image: %v", err)
			}
		}

		_, err := NewImageFolderDataset(tempDir, nil)
		if err == nil {
			t.Error("Expected error when no class subdirectories exist")
		}
	})

	t.Run("ClassWithNoImages", func(t *testing.T) {
		tempDir := t.TempDir()

		// Create class directories but only put images in one
		classes := []string{"empty_class", "full_class"}
		for _, className := range classes {
			classDir := filepath.Join(tempDir, className)
			if err := os.MkdirAll(classDir, 0755); err != nil {
				t.Fatalf("Failed to create class directory: %v", err)
			}
		}

		// Only add images to "full_class"
		fullClassDir := filepath.Join(tempDir, "full_class")
		for i := 0; i < 3; i++ {
			imagePath := filepath.Join(fullClassDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				t.Fatalf("Failed to create image: %v", err)
			}
		}

		dataset, err := NewImageFolderDataset(tempDir, nil)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should only have images from full_class, but both classes should exist
		if dataset.Len() != 3 {
			t.Errorf("Expected 3 images, got %d", dataset.Len())
		}

		if dataset.NumClasses() != 2 {
			t.Errorf("Expected 2 classes, got %d", dataset.NumClasses())
		}

		dist := dataset.ClassDistribution()
		if dist["empty_class"] != 0 {
			t.Errorf("Expected 0 images for empty_class, got %d", dist["empty_class"])
		}
		if dist["full_class"] != 3 {
			t.Errorf("Expected 3 images for full_class, got %d", dist["full_class"])
		}
	})
}

// TestImageFolderDatasetGetItem tests individual item retrieval
func TestImageFolderDatasetGetItem(t *testing.T) {
	classes := []string{"class1", "class2"}
	imagesPerClass := 3
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("ValidIndices", func(t *testing.T) {
		for i := 0; i < dataset.Len(); i++ {
			imagePath, label, err := dataset.GetItem(i)
			if err != nil {
				t.Errorf("Unexpected error at index %d: %v", i, err)
			}

			if imagePath == "" {
				t.Errorf("Empty image path at index %d", i)
			}

			if label < 0 || label >= dataset.NumClasses() {
				t.Errorf("Invalid label %d at index %d", label, i)
			}

			// Verify the file exists
			if _, err := os.Stat(imagePath); err != nil {
				t.Errorf("Image file doesn't exist: %s", imagePath)
			}
		}
	})

	t.Run("InvalidIndices", func(t *testing.T) {
		invalidIndices := []int{-1, dataset.Len(), dataset.Len() + 1}

		for _, idx := range invalidIndices {
			_, _, err := dataset.GetItem(idx)
			if err == nil {
				t.Errorf("Expected error for invalid index %d", idx)
			}

			if !strings.Contains(err.Error(), "out of range") {
				t.Errorf("Expected 'out of range' error for index %d, got: %v", idx, err)
			}
		}
	})
}

// TestImageFolderDatasetSplit tests dataset splitting
func TestImageFolderDatasetSplit(t *testing.T) {
	classes := []string{"cat", "dog"}
	imagesPerClass := 10
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("StandardSplit", func(t *testing.T) {
		trainRatio := 0.7
		trainDataset, valDataset := dataset.Split(trainRatio, false)

		expectedTrainSize := int(float64(dataset.Len()) * trainRatio)
		expectedValSize := dataset.Len() - expectedTrainSize

		if trainDataset.Len() != expectedTrainSize {
			t.Errorf("Expected train size %d, got %d", expectedTrainSize, trainDataset.Len())
		}

		if valDataset.Len() != expectedValSize {
			t.Errorf("Expected validation size %d, got %d", expectedValSize, valDataset.Len())
		}

		// Check that class information is preserved
		if trainDataset.NumClasses() != dataset.NumClasses() {
			t.Error("Train dataset should have same number of classes")
		}

		if valDataset.NumClasses() != dataset.NumClasses() {
			t.Error("Validation dataset should have same number of classes")
		}
	})

	t.Run("SplitWithShuffle", func(t *testing.T) {
		trainRatio := 0.8

		// Split without shuffle
		train1, val1 := dataset.Split(trainRatio, false)

		// Split with shuffle
		train2, val2 := dataset.Split(trainRatio, true)

		// Sizes should be the same
		if train1.Len() != train2.Len() {
			t.Error("Train dataset sizes should be the same regardless of shuffle")
		}

		if val1.Len() != val2.Len() {
			t.Error("Validation dataset sizes should be the same regardless of shuffle")
		}

		// Content might be different due to shuffle (but not guaranteed)
		// This is a probabilistic test - with small datasets, it might fail occasionally
		different := false
		minLen := min(train1.Len(), train2.Len())
		for i := 0; i < minLen && i < 5; i++ { // Check first few items
			path1, _, _ := train1.GetItem(i)
			path2, _, _ := train2.GetItem(i)
			if path1 != path2 {
				different = true
				break
			}
		}

		if !different && dataset.Len() > 2 {
			t.Log("Warning: Shuffled and non-shuffled datasets appear identical (low probability)")
		}
	})

	t.Run("EdgeCases", func(t *testing.T) {
		// Test with very small and very large ratios
		train1, val1 := dataset.Split(0.1, false) // 10% train
		train2, val2 := dataset.Split(0.9, false) // 90% train

		if train1.Len() == 0 && dataset.Len() > 10 {
			t.Error("Train dataset should not be empty with 10% split on reasonable dataset")
		}

		if val2.Len() == 0 && dataset.Len() > 10 {
			t.Error("Validation dataset should not be empty with 90% train split")
		}

		// Total should always equal original
		if train1.Len()+val1.Len() != dataset.Len() {
			t.Error("Split datasets don't sum to original size")
		}

		if train2.Len()+val2.Len() != dataset.Len() {
			t.Error("Split datasets don't sum to original size")
		}
	})
}

// TestImageFolderDatasetSubset tests subset creation
func TestImageFolderDatasetSubset(t *testing.T) {
	classes := []string{"class1", "class2", "class3"}
	imagesPerClass := 5
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("ValidSubset", func(t *testing.T) {
		indices := []int{0, 2, 4, 6, 8}
		subset := dataset.Subset(indices)

		if subset.Len() != len(indices) {
			t.Errorf("Expected subset size %d, got %d", len(indices), subset.Len())
		}

		// Verify items match original dataset at specified indices
		for i, originalIdx := range indices {
			subsetPath, subsetLabel, err := subset.GetItem(i)
			if err != nil {
				t.Errorf("Error getting subset item %d: %v", i, err)
			}

			originalPath, originalLabel, err := dataset.GetItem(originalIdx)
			if err != nil {
				t.Errorf("Error getting original item %d: %v", originalIdx, err)
			}

			if subsetPath != originalPath {
				t.Errorf("Path mismatch at subset index %d", i)
			}

			if subsetLabel != originalLabel {
				t.Errorf("Label mismatch at subset index %d", i)
			}
		}

		// Class information should be preserved
		if subset.NumClasses() != dataset.NumClasses() {
			t.Error("Subset should preserve class information")
		}
	})

	t.Run("EmptySubset", func(t *testing.T) {
		subset := dataset.Subset([]int{})

		if subset.Len() != 0 {
			t.Errorf("Expected empty subset, got size %d", subset.Len())
		}

		if subset.NumClasses() != dataset.NumClasses() {
			t.Error("Empty subset should preserve class information")
		}
	})

	t.Run("DuplicateIndices", func(t *testing.T) {
		indices := []int{0, 1, 0, 2, 1} // Duplicates allowed
		subset := dataset.Subset(indices)

		if subset.Len() != len(indices) {
			t.Errorf("Expected subset size %d, got %d", len(indices), subset.Len())
		}

		// Check that duplicates are preserved
		path0, _, _ := subset.GetItem(0)
		path2, _, _ := subset.GetItem(2)
		if path0 != path2 {
			t.Error("Duplicate indices should reference same item")
		}
	})
}

// TestImageFolderDatasetFilterByClass tests class filtering
func TestImageFolderDatasetFilterByClass(t *testing.T) {
	classes := []string{"cat", "dog", "bird", "fish"}
	imagesPerClass := 3
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("FilterToSubsetOfClasses", func(t *testing.T) {
		filterClasses := []string{"cat", "bird"}
		filtered := dataset.FilterByClass(filterClasses)

		expectedSize := len(filterClasses) * imagesPerClass
		if filtered.Len() != expectedSize {
			t.Errorf("Expected filtered size %d, got %d", expectedSize, filtered.Len())
		}

		// Check that only specified classes are present in the data
		labelCounts := make(map[int]int)
		for i := 0; i < filtered.Len(); i++ {
			_, label, _ := filtered.GetItem(i)
			labelCounts[label]++
		}

		// Should only have labels for cat and bird
		catIdx, catExists := dataset.classToIdx["cat"]
		birdIdx, birdExists := dataset.classToIdx["bird"]

		if !catExists || !birdExists {
			t.Fatal("Cat or bird class not found in original dataset")
		}

		if labelCounts[catIdx] != imagesPerClass {
			t.Errorf("Expected %d cat images, got %d", imagesPerClass, labelCounts[catIdx])
		}

		if labelCounts[birdIdx] != imagesPerClass {
			t.Errorf("Expected %d bird images, got %d", imagesPerClass, labelCounts[birdIdx])
		}

		// Should not have any other labels
		if len(labelCounts) != 2 {
			t.Errorf("Expected 2 different labels, got %d", len(labelCounts))
		}

		// Class metadata should be preserved (but filtered samples only contain specified classes)
		if filtered.NumClasses() != dataset.NumClasses() {
			t.Error("Filtered dataset should preserve original class metadata")
		}
	})

	t.Run("FilterToNonexistentClass", func(t *testing.T) {
		filterClasses := []string{"elephant", "tiger"}
		filtered := dataset.FilterByClass(filterClasses)

		if filtered.Len() != 0 {
			t.Errorf("Expected empty filtered dataset, got size %d", filtered.Len())
		}
	})

	t.Run("FilterToAllClasses", func(t *testing.T) {
		filtered := dataset.FilterByClass(classes)

		if filtered.Len() != dataset.Len() {
			t.Errorf("Expected filtered size %d, got %d", dataset.Len(), filtered.Len())
		}
	})

	t.Run("FilterToEmptyList", func(t *testing.T) {
		filtered := dataset.FilterByClass([]string{})

		if filtered.Len() != 0 {
			t.Errorf("Expected empty filtered dataset, got size %d", filtered.Len())
		}
	})
}

// TestImageFolderDatasetString tests string representation
func TestImageFolderDatasetString(t *testing.T) {
	classes := []string{"cat", "dog"}
	imagesPerClass := 5
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	str := dataset.String()

	// Check that the string contains expected information
	expectedSubstrings := []string{
		"ImageFolderDataset",
		fmt.Sprintf("%d samples", dataset.Len()),
		fmt.Sprintf("%d classes", dataset.NumClasses()),
		"Class distribution",
	}

	for _, substr := range expectedSubstrings {
		if !strings.Contains(str, substr) {
			t.Errorf("Expected string to contain '%s', got: %s", substr, str)
		}
	}

	// Check that class names and counts are mentioned
	dist := dataset.ClassDistribution()
	for className, count := range dist {
		expectedClassLine := fmt.Sprintf("%s: %d", className, count)
		if !strings.Contains(str, expectedClassLine) {
			t.Errorf("Expected string to contain '%s', got: %s", expectedClassLine, str)
		}
	}
}

// TestImageFolderDatasetClassMethods tests class-related methods
func TestImageFolderDatasetClassMethods(t *testing.T) {
	classes := []string{"a", "b", "c"}
	imagesPerClass := 4
	tempDir := createTestDataset(t, classes, imagesPerClass)

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}

	t.Run("NumClasses", func(t *testing.T) {
		if dataset.NumClasses() != len(classes) {
			t.Errorf("Expected %d classes, got %d", len(classes), dataset.NumClasses())
		}
	})

	t.Run("ClassNames", func(t *testing.T) {
		classNames := dataset.ClassNames()
		if len(classNames) != len(classes) {
			t.Errorf("Expected %d class names, got %d", len(classes), len(classNames))
		}

		// Sort both for comparison
		expectedSorted := make([]string, len(classes))
		copy(expectedSorted, classes)
		sort.Strings(expectedSorted)

		actualSorted := make([]string, len(classNames))
		copy(actualSorted, classNames)
		sort.Strings(actualSorted)

		for i, expected := range expectedSorted {
			if actualSorted[i] != expected {
				t.Errorf("Expected class name '%s', got '%s'", expected, actualSorted[i])
			}
		}
	})

	t.Run("ClassDistribution", func(t *testing.T) {
		dist := dataset.ClassDistribution()

		if len(dist) != len(classes) {
			t.Errorf("Expected distribution for %d classes, got %d", len(classes), len(dist))
		}

		totalImages := 0
		for className, count := range dist {
			if count != imagesPerClass {
				t.Errorf("Expected %d images for class %s, got %d", imagesPerClass, className, count)
			}
			totalImages += count
		}

		if totalImages != dataset.Len() {
			t.Errorf("Total images in distribution (%d) doesn't match dataset size (%d)", totalImages, dataset.Len())
		}
	})
}

// Helper function to find minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// BenchmarkImageFolderDatasetCreation benchmarks dataset creation
func BenchmarkImageFolderDatasetCreation(b *testing.B) {
	// Create a temporary dataset structure
	tempDir := b.TempDir()
	classes := []string{"class1", "class2", "class3", "class4", "class5"}
	imagesPerClass := 100

	for _, className := range classes {
		classDir := filepath.Join(tempDir, className)
		if err := os.MkdirAll(classDir, 0755); err != nil {
			b.Fatalf("Failed to create class directory: %v", err)
		}

		for i := 0; i < imagesPerClass; i++ {
			imagePath := filepath.Join(classDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				b.Fatalf("Failed to create mock image: %v", err)
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewImageFolderDataset(tempDir, nil)
		if err != nil {
			b.Fatalf("Failed to create dataset: %v", err)
		}
	}
}

// BenchmarkImageFolderDatasetGetItem benchmarks item retrieval
func BenchmarkImageFolderDatasetGetItem(b *testing.B) {
	classes := []string{"class1", "class2", "class3"}
	imagesPerClass := 1000

	tempDir := b.TempDir()
	for _, className := range classes {
		classDir := filepath.Join(tempDir, className)
		if err := os.MkdirAll(classDir, 0755); err != nil {
			b.Fatalf("Failed to create class directory: %v", err)
		}

		for i := 0; i < imagesPerClass; i++ {
			imagePath := filepath.Join(classDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				b.Fatalf("Failed to create mock image: %v", err)
			}
		}
	}

	dataset, err := NewImageFolderDataset(tempDir, nil)
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

// BenchmarkImageFolderDatasetSplit benchmarks dataset splitting
func BenchmarkImageFolderDatasetSplit(b *testing.B) {
	classes := []string{"class1", "class2", "class3"}
	imagesPerClass := 500

	tempDir := b.TempDir()
	for _, className := range classes {
		classDir := filepath.Join(tempDir, className)
		if err := os.MkdirAll(classDir, 0755); err != nil {
			b.Fatalf("Failed to create class directory: %v", err)
		}

		for i := 0; i < imagesPerClass; i++ {
			imagePath := filepath.Join(classDir, fmt.Sprintf("image_%d.jpg", i))
			if err := createMockImageFile(imagePath); err != nil {
				b.Fatalf("Failed to create mock image: %v", err)
			}
		}
	}

	dataset, err := NewImageFolderDataset(tempDir, nil)
	if err != nil {
		b.Fatalf("Failed to create dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Split(0.8, true)
	}
}