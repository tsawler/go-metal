package main

import (
	"fmt"
	"os"

	"github.com/tsawler/go-metal/training"
)

func main() {
	fmt.Println("=== Phase 1 Integration Test ===")
	
	// Run the Phase 1 test
	err := training.TestPhase1()
	if err != nil {
		fmt.Printf("Phase 1 test failed: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Println("Phase 1 test completed successfully!")
}