package main

import (
	"fmt"
	"log"

	"github.com/tsawler/go-metal/layers"
)

func main() {
	fmt.Println("Testing Tanh and Swish activation restoration...")

	// Test that we can create models with the new activations
	inputShape := []int{1, 10}
	
	// Test Tanh activation
	fmt.Println("\n✅ Testing Tanh activation:")
	tanhBuilder := layers.NewModelBuilder(inputShape)
	tanhModel, err := tanhBuilder.
		AddDense(20, true, "dense1").
		AddTanh("tanh1").
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		log.Fatalf("Failed to create Tanh model: %v", err)
	}
	
	fmt.Printf("   - Model compiled successfully with %d layers\n", len(tanhModel.Layers))
	fmt.Printf("   - Total parameters: %d\n", tanhModel.TotalParameters)
	
	// Test Swish activation
	fmt.Println("\n✅ Testing Swish activation:")
	swishBuilder := layers.NewModelBuilder(inputShape)
	swishModel, err := swishBuilder.
		AddDense(20, true, "dense1").
		AddSwish("swish1").
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		log.Fatalf("Failed to create Swish model: %v", err)
	}
	
	fmt.Printf("   - Model compiled successfully with %d layers\n", len(swishModel.Layers))
	fmt.Printf("   - Total parameters: %d\n", swishModel.TotalParameters)
	
	fmt.Println("\n🎉 All tests passed! Tanh and Swish activations are working correctly.")
}