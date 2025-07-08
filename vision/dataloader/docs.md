# dataloader
--
    import "."


## Usage

#### func  CreateSharedDataLoaders

```go
func CreateSharedDataLoaders(trainDataset, valDataset Dataset, config Config) (*DataLoader, *DataLoader)
```
CreateSharedDataLoaders creates train and validation DataLoaders with a shared
cache

#### type CacheManager

```go
type CacheManager struct {
}
```

CacheManager manages a shared cache for preprocessed data

#### func  NewCacheManager

```go
func NewCacheManager(maxSize int, itemSize int) *CacheManager
```
NewCacheManager creates a new cache manager

#### func (*CacheManager) Clear

```go
func (cm *CacheManager) Clear()
```
Clear clears the cache

#### func (*CacheManager) Get

```go
func (cm *CacheManager) Get(key string) ([]float32, bool)
```
Get retrieves an item from the cache

#### func (*CacheManager) Put

```go
func (cm *CacheManager) Put(key string, data []float32)
```
Put adds an item to the cache

#### func (*CacheManager) ResetStats

```go
func (cm *CacheManager) ResetStats()
```
ResetStats resets the statistics

#### func (*CacheManager) Stats

```go
func (cm *CacheManager) Stats() CacheStats
```
Stats returns cache statistics

#### type CacheStats

```go
type CacheStats struct {
	Size    int
	MaxSize int
	Hits    int64
	Misses  int64
	HitRate float64
}
```

CacheStats holds cache statistics

#### func (CacheStats) String

```go
func (cs CacheStats) String() string
```
String returns a string representation of cache stats

#### type Config

```go
type Config struct {
	BatchSize    int
	Shuffle      bool
	MaxCacheSize int // Maximum number of images to cache
	ImageSize    int
	NumWorkers   int           // Number of parallel workers for preprocessing
	CacheManager *CacheManager // Optional shared cache manager
}
```

Config holds configuration for DataLoader

#### type DataLoader

```go
type DataLoader struct {
}
```

DataLoader handles memory-efficient batch data loading with smart caching

#### func  NewDataLoader

```go
func NewDataLoader(dataset Dataset, config Config) *DataLoader
```
NewDataLoader creates a new data loader

#### func (*DataLoader) ClearCache

```go
func (dl *DataLoader) ClearCache()
```
ClearCache clears the image cache

#### func (*DataLoader) GetCacheManager

```go
func (dl *DataLoader) GetCacheManager() *CacheManager
```
GetCacheManager returns the cache manager for sharing between DataLoaders

#### func (*DataLoader) NextBatch

```go
func (dl *DataLoader) NextBatch() (imageData []float32, labelData []int32, actualBatchSize int, err error)
```
NextBatch loads the next batch of images

#### func (*DataLoader) Progress

```go
func (dl *DataLoader) Progress() (current, total int)
```
Progress returns the current progress through the dataset

#### func (*DataLoader) Reset

```go
func (dl *DataLoader) Reset()
```
Reset resets the data loader to the beginning

#### func (*DataLoader) Stats

```go
func (dl *DataLoader) Stats() string
```
Stats returns cache statistics

#### type Dataset

```go
type Dataset interface {
	Len() int
	GetItem(index int) (imagePath string, label int, err error)
}
```

Dataset interface defines the contract for datasets

#### type SharedCacheManager

```go
type SharedCacheManager struct {
}
```

SharedCacheManager manages a global cache that can be shared across multiple
DataLoaders

#### func  GetGlobalSharedCache

```go
func GetGlobalSharedCache() *SharedCacheManager
```
GetGlobalSharedCache returns the global shared cache manager

#### func (*SharedCacheManager) ClearAllCaches

```go
func (scm *SharedCacheManager) ClearAllCaches()
```
ClearAllCaches clears all managed caches

#### func (*SharedCacheManager) GetOrCreateCache

```go
func (scm *SharedCacheManager) GetOrCreateCache(name string, maxSize int, itemSize int) *CacheManager
```
GetOrCreateCache gets or creates a cache with the given name and parameters

#### func (*SharedCacheManager) RemoveCache

```go
func (scm *SharedCacheManager) RemoveCache(name string)
```
RemoveCache removes a cache by name
