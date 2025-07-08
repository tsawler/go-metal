# dataset
--
    import "."


## Usage

#### type CatsDogsDataset

```go
type CatsDogsDataset struct {
	*ImageFolderDataset
}
```

CatsDogsDataset is a specialized dataset for the cats vs dogs classification
task

#### func  NewCatsDogsDataset

```go
func NewCatsDogsDataset(dataDir string, maxSamplesPerClass int) (*CatsDogsDataset, error)
```
NewCatsDogsDataset creates a new cats and dogs dataset from the standard
directory structure

#### func (*CatsDogsDataset) Summary

```go
func (d *CatsDogsDataset) Summary() string
```
Summary returns a summary of the dataset

#### type ImageFolderDataset

```go
type ImageFolderDataset struct {
}
```

ImageFolderDataset represents a dataset loaded from a directory structure where
each subdirectory represents a class

#### func  NewImageFolderDataset

```go
func NewImageFolderDataset(root string, extensions []string) (*ImageFolderDataset, error)
```
NewImageFolderDataset creates a dataset from a directory structure

#### func (*ImageFolderDataset) ClassDistribution

```go
func (d *ImageFolderDataset) ClassDistribution() map[string]int
```
ClassDistribution returns the distribution of samples per class

#### func (*ImageFolderDataset) ClassNames

```go
func (d *ImageFolderDataset) ClassNames() []string
```
ClassNames returns the list of class names

#### func (*ImageFolderDataset) FilterByClass

```go
func (d *ImageFolderDataset) FilterByClass(classNames []string) *ImageFolderDataset
```
FilterByClass creates a new dataset containing only samples from specified
classes

#### func (*ImageFolderDataset) GetItem

```go
func (d *ImageFolderDataset) GetItem(index int) (string, int, error)
```
GetItem returns the image path and label at the given index

#### func (*ImageFolderDataset) Len

```go
func (d *ImageFolderDataset) Len() int
```
Len returns the number of items in the dataset

#### func (*ImageFolderDataset) NumClasses

```go
func (d *ImageFolderDataset) NumClasses() int
```
NumClasses returns the number of classes

#### func (*ImageFolderDataset) Split

```go
func (d *ImageFolderDataset) Split(trainRatio float64, shuffle bool) (*ImageFolderDataset, *ImageFolderDataset)
```
Split splits the dataset into train and validation sets

#### func (*ImageFolderDataset) String

```go
func (d *ImageFolderDataset) String() string
```
String returns a string representation of the dataset

#### func (*ImageFolderDataset) Subset

```go
func (d *ImageFolderDataset) Subset(indices []int) *ImageFolderDataset
```
Subset creates a subset of the dataset with the specified indices
