package metal_bridge

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestMetalObjectLifecycleStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	// Force frequent GC to trigger finalizers
	go func() {
		for {
			runtime.GC()
			time.Sleep(10 * time.Millisecond)
		}
	}()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Create many Metal objects rapidly
			for j := 0; j < 50; j++ {
				engine, err := NewComputeEngine()
				if err != nil {
					t.Errorf("Error creating engine %d-%d: %v", id, j, err)
					continue
				}

				data := []float32{1.0, 2.0, 3.0, 4.0}
				result, err := engine.AddArraysFloat32(data, data)
				if err != nil {
					t.Errorf("Error in computation %d-%d: %v", id, j, err)
					continue
				}
				_ = result

				// Create some buffers that might get released
				for k := 0; k < 5; k++ {
					buf, err := engine.GetDevice().CreateBufferWithBytes(data, ResourceStorageModeShared)
					if err == nil {
						_ = buf.Contents()
					}
				}

				if j%10 == 0 {
					runtime.GC()
				}
			}
		}(i)
	}
	wg.Wait()

	// Final GC to trigger any remaining finalizers
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(100 * time.Millisecond)
	}

	t.Log("Stress test completed without crashes")
}