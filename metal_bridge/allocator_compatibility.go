package metal_bridge

// Compatibility stubs for the old allocator functionality
// These are minimal implementations to allow tests to compile

// MemoryStats represents memory allocation statistics
type MemoryStats struct {
	NumAllocations   int
	NumDeallocations int
	NumPoolHits      int
	NumPoolMisses    int
	TotalMemory      int
	UsedMemory       int
	TotalFree        int
}

// BufferAllocator is a stub for compatibility
type BufferAllocator struct {
	// Empty for compatibility
}

// GetGlobalAllocator returns a stub allocator for compatibility
func GetGlobalAllocator() *BufferAllocator {
	return &BufferAllocator{}
}

// GetMemoryStats returns zero stats for compatibility
func (ba *BufferAllocator) GetMemoryStats() MemoryStats {
	return MemoryStats{
		NumAllocations:   0,
		NumDeallocations: 0,
		NumPoolHits:      0,
		NumPoolMisses:    0,
		TotalMemory:      0,
		UsedMemory:       0,
		TotalFree:        0,
	}
}