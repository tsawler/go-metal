package main

import (
	"fmt"
	"log"
	"time"

	"github.com/tsawler/go-metal/metal_bridge"
	"github.com/tsawler/go-metal/tensor"
)

func main() {
	fmt.Println("=== Go-Metal Phase 5: GPU Memory Management Demo ===\n")
	
	// Check GPU availability
	if !tensor.IsGPUAvailable() {
		log.Fatal("GPU not available - Metal compute required for this demo")
	}
	
	// Get the global allocator for monitoring
	allocator := metal_bridge.GetGlobalAllocator()
	
	fmt.Println("1. Initial Allocator State")
	printAllocatorStats(allocator)
	
	fmt.Println("\n2. Creating GPU Tensors with Different Sizes")
	
	// Create several GPU tensors of varying sizes to demonstrate pooling
	var gpuTensors []*tensor.Tensor
	sizes := []int{100, 500, 1000, 500, 100, 2000, 1000} // Note: some sizes repeat
	
	for i, size := range sizes {
		// Create CPU tensor with data
		data := make([]float32, size)
		for j := range data {
			data[j] = float32(j * (i + 1))
		}
		
		cpuTensor, err := tensor.NewTensor([]int{size}, tensor.Float32, tensor.CPU, data)
		if err != nil {
			log.Fatalf("Failed to create CPU tensor %d: %v", i, err)
		}
		
		// Transfer to GPU using BufferAllocator
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			log.Fatalf("Failed to transfer tensor %d to GPU: %v", i, err)
		}
		
		gpuTensors = append(gpuTensors, gpuTensor)
		fmt.Printf("  Created GPU tensor %d: size=%d, refCount=%d\n", 
			i+1, size, gpuTensor.RefCount())
	}
	
	fmt.Println("\n3. Allocator State After GPU Tensor Creation")
	printAllocatorStats(allocator)
	
	fmt.Println("\n4. Testing Reference Counting")
	
	// Demonstrate reference counting
	testTensor := gpuTensors[0]
	fmt.Printf("  Initial refCount: %d\n", testTensor.RefCount())
	
	testTensor.Retain()
	fmt.Printf("  After Retain(): %d\n", testTensor.RefCount())
	
	testTensor.Release()
	fmt.Printf("  After Release(): %d\n", testTensor.RefCount())
	
	fmt.Println("\n5. Releasing Half the Tensors (should return buffers to pool)")
	
	// Release some tensors to demonstrate pooling
	for i := 0; i < len(gpuTensors)/2; i++ {
		gpuTensors[i].Release()
		fmt.Printf("  Released tensor %d\n", i+1)
	}
	
	fmt.Println("\n6. Allocator State After Partial Release")
	printAllocatorStats(allocator)
	
	fmt.Println("\n7. Creating New Tensors (should reuse buffers from pool)")
	
	// Create new tensors that should reuse pooled buffers
	var newTensors []*tensor.Tensor
	reuseSizes := []int{100, 500, 1000} // Same sizes as some released tensors
	
	for i, size := range reuseSizes {
		data := make([]float32, size)
		for j := range data {
			data[j] = float32(j * 100)
		}
		
		cpuTensor, _ := tensor.NewTensor([]int{size}, tensor.Float32, tensor.CPU, data)
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			log.Fatalf("Failed to create reuse tensor %d: %v", i, err)
		}
		
		newTensors = append(newTensors, gpuTensor)
		fmt.Printf("  Created reuse tensor %d: size=%d\n", i+1, size)
	}
	
	fmt.Println("\n8. Final Allocator State (notice pool hits)")
	stats := allocator.GetMemoryStats()
	printAllocatorStats(allocator)
	
	// Demonstrate pool efficiency
	if stats.NumPoolHits > 0 {
		efficiency := float64(stats.NumPoolHits) / float64(stats.NumAllocations) * 100
		fmt.Printf("  Pool efficiency: %.1f%% (%d hits out of %d allocations)\n", 
			efficiency, stats.NumPoolHits, stats.NumAllocations)
	}
	
	fmt.Println("\n9. Testing CPU-GPU Data Transfer Integrity")
	
	// Verify data integrity through CPU-GPU transfers
	originalData := []float32{1.5, 2.7, 3.9, 4.1, 5.3}
	cpuTensor, _ := tensor.NewTensor([]int{5}, tensor.Float32, tensor.CPU, originalData)
	
	// CPU -> GPU -> CPU
	gpuTensor, _ := cpuTensor.ToGPU()
	cpuTensor2, _ := gpuTensor.ToCPU()
	
	transferredData := cpuTensor2.Data.([]float32)
	
	dataIntegrityOK := true
	for i, original := range originalData {
		if transferredData[i] != original {
			dataIntegrityOK = false
			break
		}
	}
	
	if dataIntegrityOK {
		fmt.Println("  ✓ Data integrity preserved through CPU-GPU transfers")
	} else {
		fmt.Println("  ✗ Data integrity check failed")
	}
	
	gpuTensor.Release()
	
	fmt.Println("\n10. Memory Stress Test")
	
	initialStats := allocator.GetMemoryStats()
	
	// Perform rapid allocation/deallocation to test memory management
	fmt.Println("  Performing 1000 rapid allocate/release cycles...")
	
	start := time.Now()
	for i := 0; i < 1000; i++ {
		size := 100 + (i%10)*50 // Varying sizes
		data := make([]float32, size)
		
		cpuTensor, _ := tensor.NewTensor([]int{size}, tensor.Float32, tensor.CPU, data)
		gpuTensor, err := cpuTensor.ToGPU()
		if err != nil {
			log.Fatalf("Failed to create stress test tensor %d: %v", i, err)
		}
		
		// Immediately release
		gpuTensor.Release()
	}
	
	duration := time.Since(start)
	finalStats := allocator.GetMemoryStats()
	
	fmt.Printf("  Completed in %v\n", duration)
	fmt.Printf("  Allocations: %d, Deallocations: %d\n", 
		finalStats.NumAllocations-initialStats.NumAllocations,
		finalStats.NumDeallocations-initialStats.NumDeallocations)
	
	poolHitsIncrease := finalStats.NumPoolHits - initialStats.NumPoolHits
	poolMissesIncrease := finalStats.NumPoolMisses - initialStats.NumPoolMisses
	
	fmt.Printf("  Pool hits: %d, Pool misses: %d\n", poolHitsIncrease, poolMissesIncrease)
	
	if poolHitsIncrease > poolMissesIncrease {
		fmt.Println("  ✓ Pool reuse is working effectively")
	}
	
	fmt.Println("\n11. Cleanup - Releasing All Remaining Tensors")
	
	// Release remaining tensors
	remainingTensors := gpuTensors[len(gpuTensors)/2:]
	for i, tensor := range remainingTensors {
		tensor.Release()
		fmt.Printf("  Released remaining tensor %d\n", i+1)
	}
	
	for i, tensor := range newTensors {
		tensor.Release()
		fmt.Printf("  Released new tensor %d\n", i+1)
	}
	
	fmt.Println("\n12. Final Memory State")
	printAllocatorStats(allocator)
	
	fmt.Println("\n=== Phase 5 Demo Complete ===")
	fmt.Println("\nKey Features Demonstrated:")
	fmt.Println("✓ GPU Memory Pooling and Reuse")
	fmt.Println("✓ Reference Counting for Tensor Lifetime Management")
	fmt.Println("✓ Memory Fragmentation Reduction")
	fmt.Println("✓ Efficient CPU-GPU Data Transfers")
	fmt.Println("✓ Memory Leak Prevention")
	fmt.Println("✓ Performance Monitoring and Diagnostics")
}

func printAllocatorStats(allocator *metal_bridge.BufferAllocator) {
	stats := allocator.GetMemoryStats()
	
	fmt.Printf("  Total Allocated: %d bytes\n", stats.TotalAllocated)
	fmt.Printf("  Total Free (pooled): %d bytes\n", stats.TotalFree)
	fmt.Printf("  Active Pools: %d\n", stats.NumPools)
	fmt.Printf("  Allocations: %d\n", stats.NumAllocations)
	fmt.Printf("  Deallocations: %d\n", stats.NumDeallocations)
	fmt.Printf("  Pool Hits: %d\n", stats.NumPoolHits)
	fmt.Printf("  Pool Misses: %d\n", stats.NumPoolMisses)
	
	if stats.NumAllocations > 0 {
		poolEfficiency := float64(stats.NumPoolHits) / float64(stats.NumAllocations) * 100
		fmt.Printf("  Pool Efficiency: %.1f%%\n", poolEfficiency)
	}
}