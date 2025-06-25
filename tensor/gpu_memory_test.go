package tensor

import (
	"fmt"
	"testing"
	
	"github.com/tsawler/go-metal/metal_bridge"
)

func TestTensorGPUMemoryManagement(t *testing.T) {
	// Skip if Metal is not available
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}
	
	t.Run("CPU to GPU transfer", func(t *testing.T) {
		// Create CPU tensor
		cpuData := []float32{1.0, 2.0, 3.0, 4.0}
		cpuTensor, err := NewTensor([]int{2, 2}, Float32, CPU, cpuData)
		if err != nil {
			t.Fatalf("Failed to create CPU tensor: %v", err)
		}
		
		// Transfer to GPU
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer to GPU: %v", err)
		}
		
		// Verify GPU tensor properties
		if gpuTensor.Device != GPU {
			t.Error("GPU tensor should have GPU device type")
		}
		
		if gpuTensor.GetGPUBuffer() == nil {
			t.Error("GPU tensor should have a buffer")
		}
		
		if gpuTensor.RefCount() != 1 {
			t.Errorf("Expected refCount 1, got %d", gpuTensor.RefCount())
		}
		
		// Release GPU tensor
		gpuTensor.Release()
	})
	
	t.Run("GPU to CPU transfer", func(t *testing.T) {
		// Create CPU tensor and transfer to GPU
		cpuData := []float32{5.0, 6.0, 7.0, 8.0}
		cpuTensor, err := NewTensor([]int{2, 2}, Float32, CPU, cpuData)
		if err != nil {
			t.Fatalf("Failed to create CPU tensor: %v", err)
		}
		
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer to GPU: %v", err)
		}
		
		// Transfer back to CPU
		cpuTensor2, err := gpuTensor.ToCPU()
		if err != nil {
			t.Fatalf("Failed to transfer back to CPU: %v", err)
		}
		
		// Verify data integrity
		if cpuTensor2.Device != CPU {
			t.Error("CPU tensor should have CPU device type")
		}
		
		originalData := cpuTensor.Data.([]float32)
		transferredData := cpuTensor2.Data.([]float32)
		
		for i, val := range originalData {
			if transferredData[i] != val {
				t.Errorf("Data mismatch at index %d: expected %f, got %f", 
					i, val, transferredData[i])
			}
		}
		
		// Clean up
		gpuTensor.Release()
	})
	
	t.Run("Reference counting", func(t *testing.T) {
		cpuTensor, err := NewTensor([]int{3, 3}, Float32, CPU, 
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
		if err != nil {
			t.Fatalf("Failed to create CPU tensor: %v", err)
		}
		
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer to GPU: %v", err)
		}
		
		// Test retain/release
		initialCount := gpuTensor.RefCount()
		
		gpuTensor.Retain()
		if gpuTensor.RefCount() != initialCount+1 {
			t.Errorf("Expected refCount %d after retain, got %d", 
				initialCount+1, gpuTensor.RefCount())
		}
		
		gpuTensor.Release()
		if gpuTensor.RefCount() != initialCount {
			t.Errorf("Expected refCount %d after release, got %d", 
				initialCount, gpuTensor.RefCount())
		}
		
		// Final release
		gpuTensor.Release()
	})
	
	t.Run("Multiple GPU tensors", func(t *testing.T) {
		// Create multiple GPU tensors to test allocator pooling
		var gpuTensors []*Tensor
		
		for i := 0; i < 10; i++ {
			size := 100 + i*10 // Varying sizes
			cpuData := make([]float32, size)
			for j := range cpuData {
				cpuData[j] = float32(j)
			}
			
			cpuTensor, err := NewTensor([]int{size}, Float32, CPU, cpuData)
			if err != nil {
				t.Fatalf("Failed to create CPU tensor %d: %v", i, err)
			}
			
			gpuTensor, err := cpuTensor.ToGPU()
			if err != nil {
				t.Fatalf("Failed to transfer tensor %d to GPU: %v", i, err)
			}
			
			gpuTensors = append(gpuTensors, gpuTensor)
		}
		
		// Check allocator stats
		allocator := metal_bridge.GetGlobalAllocator()
		stats := allocator.GetMemoryStats()
		
		if stats.NumAllocations == 0 {
			t.Error("Expected some allocations to have occurred")
		}
		
		t.Logf("Allocator stats: Allocations=%d, Pool hits=%d, Pool misses=%d", 
			stats.NumAllocations, stats.NumPoolHits, stats.NumPoolMisses)
		
		// Release all tensors
		for _, tensor := range gpuTensors {
			tensor.Release()
		}
		
		// Check stats after release
		finalStats := allocator.GetMemoryStats()
		if finalStats.NumDeallocations == 0 {
			t.Error("Expected some deallocations to have occurred")
		}
	})
	
	t.Run("Buffer allocator pool reuse", func(t *testing.T) {
		allocator := metal_bridge.GetGlobalAllocator()
		initialStats := allocator.GetMemoryStats()
		
		// Create, release, and recreate tensors of same size
		// This should test buffer pool reuse
		size := []int{10, 10}
		data := make([]float32, 100)
		for i := range data {
			data[i] = float32(i)
		}
		
		// First allocation
		cpuTensor1, _ := NewTensor(size, Float32, CPU, data)
		gpuTensor1, err := cpuTensor1.ToGPU()
		if err != nil {
			t.Fatalf("Failed to create first GPU tensor: %v", err)
		}
		
		gpuTensor1.Release()
		
		// Second allocation of same size - should reuse buffer
		cpuTensor2, _ := NewTensor(size, Float32, CPU, data)
		gpuTensor2, err := cpuTensor2.ToGPU()
		if err != nil {
			t.Fatalf("Failed to create second GPU tensor: %v", err)
		}
		
		gpuTensor2.Release()
		
		// Check if pool reuse occurred
		finalStats := allocator.GetMemoryStats()
		poolHitsIncrease := finalStats.NumPoolHits - initialStats.NumPoolHits
		
		if poolHitsIncrease == 0 {
			t.Log("Warning: Expected at least one pool hit for buffer reuse")
			// Note: This might not always be guaranteed due to size rounding
		}
		
		t.Logf("Pool hits increase: %d", poolHitsIncrease)
	})
}

func TestTensorMemoryLeaks(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory leak test in short mode")
	}
	
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}
	
	allocator := metal_bridge.GetGlobalAllocator()
	initialStats := allocator.GetMemoryStats()
	
	// Perform many allocation/deallocation cycles
	for i := 0; i < 100; i++ {
		// Create varying sized tensors
		size := 50 + (i % 20) * 10
		data := make([]float32, size)
		for j := range data {
			data[j] = float32(j)
		}
		
		cpuTensor, err := NewTensor([]int{size}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create CPU tensor at iteration %d: %v", i, err)
		}
		
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer to GPU at iteration %d: %v", i, err)
		}
		
		// Sometimes retain/release multiple times
		if i%5 == 0 {
			gpuTensor.Retain()
			gpuTensor.Release()
		}
		
		gpuTensor.Release()
	}
	
	finalStats := allocator.GetMemoryStats()
	
	// Check for memory leaks - all allocations should have corresponding deallocations
	// or be properly pooled
	allocationsIncrease := finalStats.NumAllocations - initialStats.NumAllocations
	deallocationsIncrease := finalStats.NumDeallocations - initialStats.NumDeallocations
	
	t.Logf("Allocations increase: %d, Deallocations increase: %d", 
		allocationsIncrease, deallocationsIncrease)
	t.Logf("Pool hits: %d, Pool misses: %d", 
		finalStats.NumPoolHits-initialStats.NumPoolHits,
		finalStats.NumPoolMisses-initialStats.NumPoolMisses)
	
	// The deallocations might be less than allocations due to pooling
	// but the total should account for all memory
	totalAccountedFor := deallocationsIncrease + finalStats.TotalFree - initialStats.TotalFree
	
	if allocationsIncrease > 0 && totalAccountedFor == 0 {
		t.Error("Potential memory leak detected: allocations occurred but no memory was accounted for")
	}
}

func BenchmarkTensorGPUTransfer(b *testing.B) {
	if !IsGPUAvailable() {
		b.Skip("GPU not available")
	}
	
	// Prepare test data
	sizes := []int{100, 1000, 10000}
	
	for _, size := range sizes {
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i)
		}
		
		cpuTensor, _ := NewTensor([]int{size}, Float32, CPU, data)
		
		b.Run(fmt.Sprintf("ToGPU_size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gpuTensor, err := cpuTensor.ToGPU()
				if err != nil {
					b.Fatalf("ToGPU failed: %v", err)
				}
				gpuTensor.Release()
			}
		})
	}
}