package tensor

import (
	"testing"
)

// Benchmark tensor creation functions
func BenchmarkZeros(b *testing.B) {
	shapes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
		{100, 100, 100},
	}
	
	for _, shape := range shapes {
		b.Run(formatShape(shape), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := Zeros(shape, Float32, CPU)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkOnes(b *testing.B) {
	shapes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
	}
	
	for _, shape := range shapes {
		b.Run(formatShape(shape), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := Ones(shape, Float32, CPU)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkRandom(b *testing.B) {
	shapes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
	}
	
	for _, shape := range shapes {
		b.Run(formatShape(shape), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := Random(shape, Float32, CPU)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Benchmark element-wise operations
func BenchmarkAdd(b *testing.B) {
	sizes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
		{100, 100, 100},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			b2, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Add(a, b2)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkMul(b *testing.B) {
	sizes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			b2, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Mul(a, b2)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkReLU(b *testing.B) {
	sizes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
		{100, 100, 100},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ReLU(a)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSigmoid(b *testing.B) {
	sizes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Sigmoid(a)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Benchmark matrix operations
func BenchmarkMatMul(b *testing.B) {
	sizes := [][4]int{
		{64, 64, 64, 64},    // Small square matrices
		{128, 128, 128, 128}, // Medium square matrices
		{256, 256, 256, 256}, // Large square matrices
		{100, 200, 200, 150}, // Rectangular matrices
	}
	
	for _, size := range sizes {
		b.Run(formatMatMulSize(size), func(b *testing.B) {
			a, _ := Random([]int{size[0], size[1]}, Float32, CPU)
			b2, _ := Random([]int{size[2], size[3]}, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := MatMul(a, b2)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkTranspose(b *testing.B) {
	sizes := [][]int{
		{100, 100},
		{200, 300},
		{10, 10, 10},
		{5, 20, 30},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Transpose(a, 0, 1)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkReshape(b *testing.B) {
	testCases := []struct {
		original []int
		target   []int
		name     string
	}{
		{[]int{100, 100}, []int{10000}, "100x100_to_10000"},
		{[]int{10000}, []int{100, 100}, "10000_to_100x100"},
		{[]int{2, 3, 4, 5}, []int{120}, "2x3x4x5_to_120"},
		{[]int{120}, []int{2, 3, 4, 5}, "120_to_2x3x4x5"},
	}
	
	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			a, _ := Random(tc.original, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Reshape(a, tc.target)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	sizes := [][]int{
		{100, 100},
		{10, 10, 10},
		{100, 100, 100},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Sum(a, 0, false)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Benchmark utility functions
func BenchmarkClone(b *testing.B) {
	sizes := [][]int{
		{100},
		{100, 100},
		{10, 10, 10},
	}
	
	for _, size := range sizes {
		b.Run(formatShape(size), func(b *testing.B) {
			a, _ := Random(size, Float32, CPU)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := a.Clone()
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAt(b *testing.B) {
	tensor, _ := Random([]int{100, 100}, Float32, CPU)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tensor.At(i%100, (i*7)%100)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSetAt(b *testing.B) {
	tensor, _ := Random([]int{100, 100}, Float32, CPU)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := tensor.SetAt(float32(i), i%100, (i*7)%100)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark data type comparisons
func BenchmarkFloat32VsInt32_Add(b *testing.B) {
	size := []int{100, 100}
	
	b.Run("Float32", func(b *testing.B) {
		a, _ := Random(size, Float32, CPU)
		b2, _ := Random(size, Float32, CPU)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := Add(a, b2)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	
	b.Run("Int32", func(b *testing.B) {
		a, _ := Random(size, Int32, CPU)
		b2, _ := Random(size, Int32, CPU)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := Add(a, b2)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// Memory allocation benchmarks
func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("SmallTensor", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tensor, _ := Zeros([]int{10, 10}, Float32, CPU)
			tensor.Release() // Explicit cleanup
		}
	})
	
	b.Run("LargeTensor", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tensor, _ := Zeros([]int{100, 100}, Float32, CPU)
			tensor.Release() // Explicit cleanup
		}
	})
}

// Complex operation benchmarks
func BenchmarkComplexOperations(b *testing.B) {
	b.Run("ChainedOperations", func(b *testing.B) {
		a, _ := Random([]int{50, 50}, Float32, CPU)
		b2, _ := Random([]int{50, 50}, Float32, CPU)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Simulate a = relu(a + b * 2)
			temp1, _ := Mul(b2, b2)    // b * b
			temp2, _ := Add(a, temp1)  // a + (b * b)
			result, _ := ReLU(temp2)   // relu(a + b * b)
			_ = result
		}
	})
	
	b.Run("MatMulChain", func(b *testing.B) {
		a, _ := Random([]int{50, 50}, Float32, CPU)
		b2, _ := Random([]int{50, 50}, Float32, CPU)
		c, _ := Random([]int{50, 50}, Float32, CPU)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Simulate (a @ b) @ c
			temp, _ := MatMul(a, b2)
			result, _ := MatMul(temp, c)
			_ = result
		}
	})
}

// Helper functions for benchmark naming
func formatShape(shape []int) string {
	result := ""
	for i, dim := range shape {
		if i > 0 {
			result += "x"
		}
		result += string(rune('0' + dim/1000))
		result += string(rune('0' + (dim%1000)/100))
		result += string(rune('0' + (dim%100)/10))
		result += string(rune('0' + dim%10))
	}
	return result
}

func formatMatMulSize(size [4]int) string {
	return formatShape([]int{size[0], size[1]}) + "_x_" + formatShape([]int{size[2], size[3]})
}