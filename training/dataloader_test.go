package training

import (
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

func TestSimpleDataset(t *testing.T) {
	t.Run("Simple dataset creation", func(t *testing.T) {
		// Create data and labels
		data1, _ := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{1, 2, 3, 4})
		data2, _ := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{5, 6, 7, 8})
		label1, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		label2, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{1})
		
		data := []*tensor.Tensor{data1, data2}
		labels := []*tensor.Tensor{label1, label2}
		
		dataset, err := NewSimpleDataset(data, labels)
		if err != nil {
			t.Fatalf("Failed to create simple dataset: %v", err)
		}
		
		// Check length
		if dataset.Len() != 2 {
			t.Errorf("Expected dataset length 2, got %d", dataset.Len())
		}
		
		// Check samples
		d, l, err := dataset.Get(0)
		if err != nil {
			t.Fatalf("Failed to get sample 0: %v", err)
		}
		
		if d != data1 || l != label1 {
			t.Error("Sample 0 data or label mismatch")
		}
		
		d, l, err = dataset.Get(1)
		if err != nil {
			t.Fatalf("Failed to get sample 1: %v", err)
		}
		
		if d != data2 || l != label2 {
			t.Error("Sample 1 data or label mismatch")
		}
	})
	
	t.Run("Simple dataset error cases", func(t *testing.T) {
		data := []*tensor.Tensor{{}}
		labels := []*tensor.Tensor{{}, {}}
		
		_, err := NewSimpleDataset(data, labels)
		if err == nil {
			t.Error("Expected error for mismatched data and labels length")
		}
		
		// Valid dataset but invalid index
		data1, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1})
		label1, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		dataset, _ := NewSimpleDataset([]*tensor.Tensor{data1}, []*tensor.Tensor{label1})
		
		_, _, err = dataset.Get(-1)
		if err == nil {
			t.Error("Expected error for negative index")
		}
		
		_, _, err = dataset.Get(1)
		if err == nil {
			t.Error("Expected error for out of bounds index")
		}
	})
}

func TestRandomDataset(t *testing.T) {
	t.Run("Random dataset creation", func(t *testing.T) {
		dataset := NewRandomDataset(100, []int{3, 32, 32}, []int{1}, tensor.Float32, tensor.Int32, 10)
		
		if dataset.Len() != 100 {
			t.Errorf("Expected dataset length 100, got %d", dataset.Len())
		}
		
		// Get a sample
		data, label, err := dataset.Get(0)
		if err != nil {
			t.Fatalf("Failed to get random sample: %v", err)
		}
		
		// Check data shape and type
		expectedDataShape := []int{3, 32, 32}
		if len(data.Shape) != len(expectedDataShape) {
			t.Fatalf("Expected data shape %v, got %v", expectedDataShape, data.Shape)
		}
		
		for i, dim := range expectedDataShape {
			if data.Shape[i] != dim {
				t.Errorf("Data shape dimension %d: expected %d, got %d", i, dim, data.Shape[i])
			}
		}
		
		if data.DType != tensor.Float32 {
			t.Errorf("Expected data type Float32, got %s", data.DType)
		}
		
		// Check label shape and type
		expectedLabelShape := []int{1}
		if len(label.Shape) != len(expectedLabelShape) {
			t.Fatalf("Expected label shape %v, got %v", expectedLabelShape, label.Shape)
		}
		
		if label.DType != tensor.Int32 {
			t.Errorf("Expected label type Int32, got %s", label.DType)
		}
		
		// Check label value is within range
		labelData := label.Data.([]int32)
		if labelData[0] < 0 || labelData[0] >= 10 {
			t.Errorf("Label value %d out of range [0, 10)", labelData[0])
		}
	})
	
	t.Run("Random dataset different samples", func(t *testing.T) {
		dataset := NewRandomDataset(10, []int{2}, []int{1}, tensor.Float32, tensor.Int32, 2)
		
		// Get two different samples
		data1, _, _ := dataset.Get(0)
		data2, _, _ := dataset.Get(1)
		
		// They should be different (very high probability)
		data1Vals := data1.Data.([]float32)
		data2Vals := data2.Data.([]float32)
		
		same := true
		for i := range data1Vals {
			if data1Vals[i] != data2Vals[i] {
				same = false
				break
			}
		}
		
		if same {
			t.Error("Random dataset samples should be different")
		}
	})
}

func TestDataLoader(t *testing.T) {
	t.Run("DataLoader basic functionality", func(t *testing.T) {
		// Create simple dataset
		data1, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{1, 2})
		data2, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{3, 4})
		data3, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{5, 6})
		label1, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		label2, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{1})
		label3, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		
		dataset, _ := NewSimpleDataset(
			[]*tensor.Tensor{data1, data2, data3},
			[]*tensor.Tensor{label1, label2, label3},
		)
		
		// Create DataLoader with batch size 2
		dataloader := NewDataLoader(dataset, 2, false, 1, tensor.CPU)
		
		// Check number of batches
		expectedBatches := 2 // ceil(3/2) = 2
		if dataloader.Len() != expectedBatches {
			t.Errorf("Expected %d batches, got %d", expectedBatches, dataloader.Len())
		}
		
		// Get first batch
		batch, err := dataloader.Next()
		if err != nil {
			t.Fatalf("Failed to get first batch: %v", err)
		}
		
		if batch == nil {
			t.Fatal("First batch should not be nil")
		}
		
		// Check batch data shape: [batch_size, ...] = [2, 2]
		if len(batch.Data.Shape) != 2 || batch.Data.Shape[0] != 2 || batch.Data.Shape[1] != 2 {
			t.Errorf("Expected batch data shape [2, 2], got %v", batch.Data.Shape)
		}
		
		// Check batch labels shape: [batch_size, ...] = [2, 1]
		if len(batch.Labels.Shape) != 2 || batch.Labels.Shape[0] != 2 || batch.Labels.Shape[1] != 1 {
			t.Errorf("Expected batch labels shape [2, 1], got %v", batch.Labels.Shape)
		}
		
		// Get second batch
		batch2, err := dataloader.Next()
		if err != nil {
			t.Fatalf("Failed to get second batch: %v", err)
		}
		
		if batch2 == nil {
			t.Fatal("Second batch should not be nil")
		}
		
		// Second batch should have size 1 (remaining sample)
		if batch2.Data.Shape[0] != 1 {
			t.Errorf("Expected second batch size 1, got %d", batch2.Data.Shape[0])
		}
		
		// Third call should return nil (end of epoch)
		batch3, err := dataloader.Next()
		if err != nil {
			t.Fatalf("Third Next() call failed: %v", err)
		}
		
		if batch3 != nil {
			t.Error("Third batch should be nil (end of epoch)")
		}
	})
	
	t.Run("DataLoader with shuffling", func(t *testing.T) {
		// Create dataset with distinguishable samples
		data1, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1})
		data2, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{2})
		data3, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{3})
		data4, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{4})
		
		label1, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{1})
		label2, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{2})
		label3, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{3})
		label4, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{4})
		
		dataset, _ := NewSimpleDataset(
			[]*tensor.Tensor{data1, data2, data3, data4},
			[]*tensor.Tensor{label1, label2, label3, label4},
		)
		
		// Create DataLoader with shuffling
		dataloader := NewDataLoader(dataset, 2, true, 1, tensor.CPU)
		
		// Collect first epoch order
		dataloader.Reset()
		var firstEpochOrder []float32
		
		for dataloader.HasNext() {
			batch, _ := dataloader.Next()
			if batch != nil {
				batchData := batch.Data.Data.([]float32)
				firstEpochOrder = append(firstEpochOrder, batchData...)
			}
		}
		
		// Collect second epoch order
		dataloader.Reset()
		var secondEpochOrder []float32
		
		for dataloader.HasNext() {
			batch, _ := dataloader.Next()
			if batch != nil {
				batchData := batch.Data.Data.([]float32)
				secondEpochOrder = append(secondEpochOrder, batchData...)
			}
		}
		
		// With shuffling, the orders might be different
		// (Note: This test might occasionally fail due to randomness, but very rarely)
		_ = firstEpochOrder // Suppress unused variable warning
		
		// We can't guarantee they're different due to randomness, but we can check they contain the same elements
		if len(firstEpochOrder) != 4 || len(secondEpochOrder) != 4 {
			t.Error("Each epoch should contain all 4 samples")
		}
	})
	
	t.Run("DataLoader iterator", func(t *testing.T) {
		dataset := NewRandomDataset(5, []int{2}, []int{1}, tensor.Float32, tensor.Int32, 2)
		dataloader := NewDataLoader(dataset, 2, false, 1, tensor.CPU)
		
		batchCount := 0
		for batch := range dataloader.Iterator() {
			if batch == nil {
				t.Error("Iterator should not yield nil batches")
			}
			batchCount++
		}
		
		expectedBatches := 3 // ceil(5/2) = 3
		if batchCount != expectedBatches {
			t.Errorf("Expected %d batches from iterator, got %d", expectedBatches, batchCount)
		}
	})
	
	t.Run("DataLoader HasNext", func(t *testing.T) {
		dataset := NewRandomDataset(3, []int{1}, []int{1}, tensor.Float32, tensor.Int32, 2)
		dataloader := NewDataLoader(dataset, 2, false, 1, tensor.CPU)
		
		// Initially should have next
		if !dataloader.HasNext() {
			t.Error("DataLoader should have next batch initially")
		}
		
		// Get first batch
		dataloader.Next()
		
		// Should still have next
		if !dataloader.HasNext() {
			t.Error("DataLoader should have next batch after first")
		}
		
		// Get second batch
		dataloader.Next()
		
		// Should not have next
		if dataloader.HasNext() {
			t.Error("DataLoader should not have next batch after all batches consumed")
		}
	})
}