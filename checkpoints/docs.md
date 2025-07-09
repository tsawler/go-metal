# checkpoints
--
    import "."


## Usage

```go
var (
	TensorProto_DataType_name = map[int32]string{
		0:  "UNDEFINED",
		1:  "FLOAT",
		2:  "UINT8",
		3:  "INT8",
		4:  "UINT16",
		5:  "INT16",
		6:  "INT32",
		7:  "INT64",
		8:  "STRING",
		9:  "BOOL",
		10: "FLOAT16",
		11: "DOUBLE",
		12: "UINT32",
		13: "UINT64",
		14: "COMPLEX64",
		15: "COMPLEX128",
		16: "BFLOAT16",
	}
	TensorProto_DataType_value = map[string]int32{
		"UNDEFINED":  0,
		"FLOAT":      1,
		"UINT8":      2,
		"INT8":       3,
		"UINT16":     4,
		"INT16":      5,
		"INT32":      6,
		"INT64":      7,
		"STRING":     8,
		"BOOL":       9,
		"FLOAT16":    10,
		"DOUBLE":     11,
		"UINT32":     12,
		"UINT64":     13,
		"COMPLEX64":  14,
		"COMPLEX128": 15,
		"BFLOAT16":   16,
	}
)
```
Enum value maps for TensorProto_DataType.

```go
var (
	AttributeProto_AttributeType_name = map[int32]string{
		0:  "UNDEFINED_ATTR",
		1:  "FLOAT_ATTR",
		2:  "INT_ATTR",
		3:  "STRING_ATTR",
		4:  "TENSOR_ATTR",
		5:  "GRAPH_ATTR",
		11: "SPARSE_TENSOR_ATTR",
		13: "TYPE_PROTO_ATTR",
		6:  "FLOATS_ATTR",
		7:  "INTS_ATTR",
		8:  "STRINGS_ATTR",
		9:  "TENSORS_ATTR",
		10: "GRAPHS_ATTR",
		12: "SPARSE_TENSORS_ATTR",
		14: "TYPE_PROTOS_ATTR",
	}
	AttributeProto_AttributeType_value = map[string]int32{
		"UNDEFINED_ATTR":      0,
		"FLOAT_ATTR":          1,
		"INT_ATTR":            2,
		"STRING_ATTR":         3,
		"TENSOR_ATTR":         4,
		"GRAPH_ATTR":          5,
		"SPARSE_TENSOR_ATTR":  11,
		"TYPE_PROTO_ATTR":     13,
		"FLOATS_ATTR":         6,
		"INTS_ATTR":           7,
		"STRINGS_ATTR":        8,
		"TENSORS_ATTR":        9,
		"GRAPHS_ATTR":         10,
		"SPARSE_TENSORS_ATTR": 12,
		"TYPE_PROTOS_ATTR":    14,
	}
)
```
Enum value maps for AttributeProto_AttributeType.

```go
var File_checkpoints_onnx_proto protoreflect.FileDescriptor
```

#### func  LoadWeightsIntoTensors

```go
func LoadWeightsIntoTensors(weights []WeightTensor, tensors []*memory.Tensor) error
```
LoadWeightsIntoTensors loads weight data back into GPU tensors

#### type AttributeProto

```go
type AttributeProto struct {
	Name          string                       `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	RefAttrName   string                       `protobuf:"bytes,21,opt,name=ref_attr_name,json=refAttrName,proto3" json:"ref_attr_name,omitempty"`
	DocString     string                       `protobuf:"bytes,13,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
	Type          AttributeProto_AttributeType `protobuf:"varint,20,opt,name=type,proto3,enum=checkpoints.AttributeProto_AttributeType" json:"type,omitempty"`
	F             float32                      `protobuf:"fixed32,2,opt,name=f,proto3" json:"f,omitempty"`
	I             int64                        `protobuf:"varint,3,opt,name=i,proto3" json:"i,omitempty"`
	S             []byte                       `protobuf:"bytes,4,opt,name=s,proto3" json:"s,omitempty"`
	T             *TensorProto                 `protobuf:"bytes,5,opt,name=t,proto3" json:"t,omitempty"`
	G             *GraphProto                  `protobuf:"bytes,6,opt,name=g,proto3" json:"g,omitempty"`
	SparseTensor  *SparseTensorProto           `protobuf:"bytes,22,opt,name=sparse_tensor,json=sparseTensor,proto3" json:"sparse_tensor,omitempty"`
	Tp            *TypeProto                   `protobuf:"bytes,14,opt,name=tp,proto3" json:"tp,omitempty"`
	Floats        []float32                    `protobuf:"fixed32,7,rep,packed,name=floats,proto3" json:"floats,omitempty"`
	Ints          []int64                      `protobuf:"varint,8,rep,packed,name=ints,proto3" json:"ints,omitempty"`
	Strings       [][]byte                     `protobuf:"bytes,9,rep,name=strings,proto3" json:"strings,omitempty"`
	Tensors       []*TensorProto               `protobuf:"bytes,10,rep,name=tensors,proto3" json:"tensors,omitempty"`
	Graphs        []*GraphProto                `protobuf:"bytes,11,rep,name=graphs,proto3" json:"graphs,omitempty"`
	SparseTensors []*SparseTensorProto         `protobuf:"bytes,23,rep,name=sparse_tensors,json=sparseTensors,proto3" json:"sparse_tensors,omitempty"`
	TypeProtos    []*TypeProto                 `protobuf:"bytes,15,rep,name=type_protos,json=typeProtos,proto3" json:"type_protos,omitempty"`
}
```

Attribute definition

#### func (*AttributeProto) Descriptor

```go
func (*AttributeProto) Descriptor() ([]byte, []int)
```
Deprecated: Use AttributeProto.ProtoReflect.Descriptor instead.

#### func (*AttributeProto) GetDocString

```go
func (x *AttributeProto) GetDocString() string
```

#### func (*AttributeProto) GetF

```go
func (x *AttributeProto) GetF() float32
```

#### func (*AttributeProto) GetFloats

```go
func (x *AttributeProto) GetFloats() []float32
```

#### func (*AttributeProto) GetG

```go
func (x *AttributeProto) GetG() *GraphProto
```

#### func (*AttributeProto) GetGraphs

```go
func (x *AttributeProto) GetGraphs() []*GraphProto
```

#### func (*AttributeProto) GetI

```go
func (x *AttributeProto) GetI() int64
```

#### func (*AttributeProto) GetInts

```go
func (x *AttributeProto) GetInts() []int64
```

#### func (*AttributeProto) GetName

```go
func (x *AttributeProto) GetName() string
```

#### func (*AttributeProto) GetRefAttrName

```go
func (x *AttributeProto) GetRefAttrName() string
```

#### func (*AttributeProto) GetS

```go
func (x *AttributeProto) GetS() []byte
```

#### func (*AttributeProto) GetSparseTensor

```go
func (x *AttributeProto) GetSparseTensor() *SparseTensorProto
```

#### func (*AttributeProto) GetSparseTensors

```go
func (x *AttributeProto) GetSparseTensors() []*SparseTensorProto
```

#### func (*AttributeProto) GetStrings

```go
func (x *AttributeProto) GetStrings() [][]byte
```

#### func (*AttributeProto) GetT

```go
func (x *AttributeProto) GetT() *TensorProto
```

#### func (*AttributeProto) GetTensors

```go
func (x *AttributeProto) GetTensors() []*TensorProto
```

#### func (*AttributeProto) GetTp

```go
func (x *AttributeProto) GetTp() *TypeProto
```

#### func (*AttributeProto) GetType

```go
func (x *AttributeProto) GetType() AttributeProto_AttributeType
```

#### func (*AttributeProto) GetTypeProtos

```go
func (x *AttributeProto) GetTypeProtos() []*TypeProto
```

#### func (*AttributeProto) ProtoMessage

```go
func (*AttributeProto) ProtoMessage()
```

#### func (*AttributeProto) ProtoReflect

```go
func (x *AttributeProto) ProtoReflect() protoreflect.Message
```

#### func (*AttributeProto) Reset

```go
func (x *AttributeProto) Reset()
```

#### func (*AttributeProto) String

```go
func (x *AttributeProto) String() string
```

#### type AttributeProto_AttributeType

```go
type AttributeProto_AttributeType int32
```

Attribute types

```go
const (
	AttributeProto_AttributeType_UNDEFINED_ATTR      AttributeProto_AttributeType = 0
	AttributeProto_AttributeType_FLOAT_ATTR          AttributeProto_AttributeType = 1
	AttributeProto_AttributeType_INT_ATTR            AttributeProto_AttributeType = 2
	AttributeProto_AttributeType_STRING_ATTR         AttributeProto_AttributeType = 3
	AttributeProto_AttributeType_TENSOR_ATTR         AttributeProto_AttributeType = 4
	AttributeProto_AttributeType_GRAPH_ATTR          AttributeProto_AttributeType = 5
	AttributeProto_AttributeType_SPARSE_TENSOR_ATTR  AttributeProto_AttributeType = 11
	AttributeProto_AttributeType_TYPE_PROTO_ATTR     AttributeProto_AttributeType = 13
	AttributeProto_AttributeType_FLOATS_ATTR         AttributeProto_AttributeType = 6
	AttributeProto_AttributeType_INTS_ATTR           AttributeProto_AttributeType = 7
	AttributeProto_AttributeType_STRINGS_ATTR        AttributeProto_AttributeType = 8
	AttributeProto_AttributeType_TENSORS_ATTR        AttributeProto_AttributeType = 9
	AttributeProto_AttributeType_GRAPHS_ATTR         AttributeProto_AttributeType = 10
	AttributeProto_AttributeType_SPARSE_TENSORS_ATTR AttributeProto_AttributeType = 12
	AttributeProto_AttributeType_TYPE_PROTOS_ATTR    AttributeProto_AttributeType = 14
)
```

#### func (AttributeProto_AttributeType) Descriptor

```go
func (AttributeProto_AttributeType) Descriptor() protoreflect.EnumDescriptor
```

#### func (AttributeProto_AttributeType) Enum

```go
func (x AttributeProto_AttributeType) Enum() *AttributeProto_AttributeType
```

#### func (AttributeProto_AttributeType) EnumDescriptor

```go
func (AttributeProto_AttributeType) EnumDescriptor() ([]byte, []int)
```
Deprecated: Use AttributeProto_AttributeType.Descriptor instead.

#### func (AttributeProto_AttributeType) Number

```go
func (x AttributeProto_AttributeType) Number() protoreflect.EnumNumber
```

#### func (AttributeProto_AttributeType) String

```go
func (x AttributeProto_AttributeType) String() string
```

#### func (AttributeProto_AttributeType) Type

```go
func (AttributeProto_AttributeType) Type() protoreflect.EnumType
```

#### type Checkpoint

```go
type Checkpoint struct {
	// Model architecture and weights
	ModelSpec *layers.ModelSpec `json:"model_spec"`
	Weights   []WeightTensor    `json:"weights"`

	// Training state
	TrainingState TrainingState `json:"training_state"`

	// Optimizer state (if available)
	OptimizerState *OptimizerState `json:"optimizer_state,omitempty"`

	// Metadata
	Metadata CheckpointMetadata `json:"metadata"`
}
```

Checkpoint represents a complete model state including weights, optimizer state,
and training metadata

#### type CheckpointFormat

```go
type CheckpointFormat int
```

CheckpointFormat defines the serialization format

```go
const (
	FormatJSON CheckpointFormat = iota
	FormatONNX
)
```

#### func (CheckpointFormat) String

```go
func (cf CheckpointFormat) String() string
```

#### type CheckpointMetadata

```go
type CheckpointMetadata struct {
	Version     string    `json:"version"`
	Framework   string    `json:"framework"`
	CreatedAt   time.Time `json:"created_at"`
	Description string    `json:"description,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
}
```

CheckpointMetadata contains checkpoint metadata

#### type CheckpointSaver

```go
type CheckpointSaver struct {
}
```

CheckpointSaver handles saving model checkpoints in various formats

#### func  NewCheckpointSaver

```go
func NewCheckpointSaver(format CheckpointFormat) *CheckpointSaver
```
NewCheckpointSaver creates a new checkpoint saver for the specified format

#### func (*CheckpointSaver) LoadCheckpoint

```go
func (cs *CheckpointSaver) LoadCheckpoint(path string) (*Checkpoint, error)
```
LoadCheckpoint loads a model checkpoint

#### func (*CheckpointSaver) SaveCheckpoint

```go
func (cs *CheckpointSaver) SaveCheckpoint(checkpoint *Checkpoint, path string) error
```
SaveCheckpoint saves a complete model checkpoint

#### type FunctionProto

```go
type FunctionProto struct {
	Name        string                `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Input       []string              `protobuf:"bytes,4,rep,name=input,proto3" json:"input,omitempty"`
	Output      []string              `protobuf:"bytes,5,rep,name=output,proto3" json:"output,omitempty"`
	Attribute   []string              `protobuf:"bytes,6,rep,name=attribute,proto3" json:"attribute,omitempty"`
	Node        []*NodeProto          `protobuf:"bytes,7,rep,name=node,proto3" json:"node,omitempty"`
	DocString   string                `protobuf:"bytes,8,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
	OpsetImport []*OperatorSetIdProto `protobuf:"bytes,9,rep,name=opset_import,json=opsetImport,proto3" json:"opset_import,omitempty"`
	Domain      string                `protobuf:"bytes,10,opt,name=domain,proto3" json:"domain,omitempty"`
}
```

Function (minimal)

#### func (*FunctionProto) Descriptor

```go
func (*FunctionProto) Descriptor() ([]byte, []int)
```
Deprecated: Use FunctionProto.ProtoReflect.Descriptor instead.

#### func (*FunctionProto) GetAttribute

```go
func (x *FunctionProto) GetAttribute() []string
```

#### func (*FunctionProto) GetDocString

```go
func (x *FunctionProto) GetDocString() string
```

#### func (*FunctionProto) GetDomain

```go
func (x *FunctionProto) GetDomain() string
```

#### func (*FunctionProto) GetInput

```go
func (x *FunctionProto) GetInput() []string
```

#### func (*FunctionProto) GetName

```go
func (x *FunctionProto) GetName() string
```

#### func (*FunctionProto) GetNode

```go
func (x *FunctionProto) GetNode() []*NodeProto
```

#### func (*FunctionProto) GetOpsetImport

```go
func (x *FunctionProto) GetOpsetImport() []*OperatorSetIdProto
```

#### func (*FunctionProto) GetOutput

```go
func (x *FunctionProto) GetOutput() []string
```

#### func (*FunctionProto) ProtoMessage

```go
func (*FunctionProto) ProtoMessage()
```

#### func (*FunctionProto) ProtoReflect

```go
func (x *FunctionProto) ProtoReflect() protoreflect.Message
```

#### func (*FunctionProto) Reset

```go
func (x *FunctionProto) Reset()
```

#### func (*FunctionProto) String

```go
func (x *FunctionProto) String() string
```

#### type GraphProto

```go
type GraphProto struct {
	Node                   []*NodeProto         `protobuf:"bytes,1,rep,name=node,proto3" json:"node,omitempty"`
	Name                   string               `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	Initializer            []*TensorProto       `protobuf:"bytes,5,rep,name=initializer,proto3" json:"initializer,omitempty"`
	SparseInitializer      []*SparseTensorProto `protobuf:"bytes,15,rep,name=sparse_initializer,json=sparseInitializer,proto3" json:"sparse_initializer,omitempty"`
	DocString              string               `protobuf:"bytes,10,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
	Input                  []*ValueInfoProto    `protobuf:"bytes,11,rep,name=input,proto3" json:"input,omitempty"`
	Output                 []*ValueInfoProto    `protobuf:"bytes,12,rep,name=output,proto3" json:"output,omitempty"`
	ValueInfo              []*ValueInfoProto    `protobuf:"bytes,13,rep,name=value_info,json=valueInfo,proto3" json:"value_info,omitempty"`
	QuantizationAnnotation []*TensorAnnotation  `protobuf:"bytes,14,rep,name=quantization_annotation,json=quantizationAnnotation,proto3" json:"quantization_annotation,omitempty"`
}
```

Graph

#### func (*GraphProto) Descriptor

```go
func (*GraphProto) Descriptor() ([]byte, []int)
```
Deprecated: Use GraphProto.ProtoReflect.Descriptor instead.

#### func (*GraphProto) GetDocString

```go
func (x *GraphProto) GetDocString() string
```

#### func (*GraphProto) GetInitializer

```go
func (x *GraphProto) GetInitializer() []*TensorProto
```

#### func (*GraphProto) GetInput

```go
func (x *GraphProto) GetInput() []*ValueInfoProto
```

#### func (*GraphProto) GetName

```go
func (x *GraphProto) GetName() string
```

#### func (*GraphProto) GetNode

```go
func (x *GraphProto) GetNode() []*NodeProto
```

#### func (*GraphProto) GetOutput

```go
func (x *GraphProto) GetOutput() []*ValueInfoProto
```

#### func (*GraphProto) GetQuantizationAnnotation

```go
func (x *GraphProto) GetQuantizationAnnotation() []*TensorAnnotation
```

#### func (*GraphProto) GetSparseInitializer

```go
func (x *GraphProto) GetSparseInitializer() []*SparseTensorProto
```

#### func (*GraphProto) GetValueInfo

```go
func (x *GraphProto) GetValueInfo() []*ValueInfoProto
```

#### func (*GraphProto) ProtoMessage

```go
func (*GraphProto) ProtoMessage()
```

#### func (*GraphProto) ProtoReflect

```go
func (x *GraphProto) ProtoReflect() protoreflect.Message
```

#### func (*GraphProto) Reset

```go
func (x *GraphProto) Reset()
```

#### func (*GraphProto) String

```go
func (x *GraphProto) String() string
```

#### type ModelProto

```go
type ModelProto struct {
	IrVersion       int64                     `protobuf:"varint,1,opt,name=ir_version,json=irVersion,proto3" json:"ir_version,omitempty"`
	OpsetImport     []*OperatorSetIdProto     `protobuf:"bytes,8,rep,name=opset_import,json=opsetImport,proto3" json:"opset_import,omitempty"`
	ProducerName    string                    `protobuf:"bytes,2,opt,name=producer_name,json=producerName,proto3" json:"producer_name,omitempty"`
	ProducerVersion string                    `protobuf:"bytes,3,opt,name=producer_version,json=producerVersion,proto3" json:"producer_version,omitempty"`
	Domain          string                    `protobuf:"bytes,4,opt,name=domain,proto3" json:"domain,omitempty"`
	ModelVersion    int64                     `protobuf:"varint,5,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
	DocString       string                    `protobuf:"bytes,6,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
	Graph           *GraphProto               `protobuf:"bytes,7,opt,name=graph,proto3" json:"graph,omitempty"`
	MetadataProps   []*StringStringEntryProto `protobuf:"bytes,14,rep,name=metadata_props,json=metadataProps,proto3" json:"metadata_props,omitempty"`
	TrainingInfo    []*TrainingInfoProto      `protobuf:"bytes,20,rep,name=training_info,json=trainingInfo,proto3" json:"training_info,omitempty"`
	Functions       []*FunctionProto          `protobuf:"bytes,25,rep,name=functions,proto3" json:"functions,omitempty"`
}
```

Model

#### func (*ModelProto) Descriptor

```go
func (*ModelProto) Descriptor() ([]byte, []int)
```
Deprecated: Use ModelProto.ProtoReflect.Descriptor instead.

#### func (*ModelProto) GetDocString

```go
func (x *ModelProto) GetDocString() string
```

#### func (*ModelProto) GetDomain

```go
func (x *ModelProto) GetDomain() string
```

#### func (*ModelProto) GetFunctions

```go
func (x *ModelProto) GetFunctions() []*FunctionProto
```

#### func (*ModelProto) GetGraph

```go
func (x *ModelProto) GetGraph() *GraphProto
```

#### func (*ModelProto) GetIrVersion

```go
func (x *ModelProto) GetIrVersion() int64
```

#### func (*ModelProto) GetMetadataProps

```go
func (x *ModelProto) GetMetadataProps() []*StringStringEntryProto
```

#### func (*ModelProto) GetModelVersion

```go
func (x *ModelProto) GetModelVersion() int64
```

#### func (*ModelProto) GetOpsetImport

```go
func (x *ModelProto) GetOpsetImport() []*OperatorSetIdProto
```

#### func (*ModelProto) GetProducerName

```go
func (x *ModelProto) GetProducerName() string
```

#### func (*ModelProto) GetProducerVersion

```go
func (x *ModelProto) GetProducerVersion() string
```

#### func (*ModelProto) GetTrainingInfo

```go
func (x *ModelProto) GetTrainingInfo() []*TrainingInfoProto
```

#### func (*ModelProto) ProtoMessage

```go
func (*ModelProto) ProtoMessage()
```

#### func (*ModelProto) ProtoReflect

```go
func (x *ModelProto) ProtoReflect() protoreflect.Message
```

#### func (*ModelProto) Reset

```go
func (x *ModelProto) Reset()
```

#### func (*ModelProto) String

```go
func (x *ModelProto) String() string
```

#### type NodeProto

```go
type NodeProto struct {
	Input     []string          `protobuf:"bytes,1,rep,name=input,proto3" json:"input,omitempty"`
	Output    []string          `protobuf:"bytes,2,rep,name=output,proto3" json:"output,omitempty"`
	Name      string            `protobuf:"bytes,3,opt,name=name,proto3" json:"name,omitempty"`
	OpType    string            `protobuf:"bytes,4,opt,name=op_type,json=opType,proto3" json:"op_type,omitempty"`
	Domain    string            `protobuf:"bytes,7,opt,name=domain,proto3" json:"domain,omitempty"`
	Attribute []*AttributeProto `protobuf:"bytes,5,rep,name=attribute,proto3" json:"attribute,omitempty"`
	DocString string            `protobuf:"bytes,6,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
}
```

Node - represents a computation operation

#### func (*NodeProto) Descriptor

```go
func (*NodeProto) Descriptor() ([]byte, []int)
```
Deprecated: Use NodeProto.ProtoReflect.Descriptor instead.

#### func (*NodeProto) GetAttribute

```go
func (x *NodeProto) GetAttribute() []*AttributeProto
```

#### func (*NodeProto) GetDocString

```go
func (x *NodeProto) GetDocString() string
```

#### func (*NodeProto) GetDomain

```go
func (x *NodeProto) GetDomain() string
```

#### func (*NodeProto) GetInput

```go
func (x *NodeProto) GetInput() []string
```

#### func (*NodeProto) GetName

```go
func (x *NodeProto) GetName() string
```

#### func (*NodeProto) GetOpType

```go
func (x *NodeProto) GetOpType() string
```

#### func (*NodeProto) GetOutput

```go
func (x *NodeProto) GetOutput() []string
```

#### func (*NodeProto) ProtoMessage

```go
func (*NodeProto) ProtoMessage()
```

#### func (*NodeProto) ProtoReflect

```go
func (x *NodeProto) ProtoReflect() protoreflect.Message
```

#### func (*NodeProto) Reset

```go
func (x *NodeProto) Reset()
```

#### func (*NodeProto) String

```go
func (x *NodeProto) String() string
```

#### type ONNXExporter

```go
type ONNXExporter struct {
}
```

ONNXExporter handles conversion of go-metal models to ONNX format

#### func  NewONNXExporter

```go
func NewONNXExporter() *ONNXExporter
```
NewONNXExporter creates a new ONNX exporter

#### func (*ONNXExporter) ExportToONNX

```go
func (oe *ONNXExporter) ExportToONNX(checkpoint *Checkpoint, path string) error
```
ExportToONNX converts a go-metal checkpoint to ONNX format

#### type ONNXImporter

```go
type ONNXImporter struct{}
```

ONNXImporter handles importing ONNX models to go-metal format

#### func  NewONNXImporter

```go
func NewONNXImporter() *ONNXImporter
```
NewONNXImporter creates a new ONNX importer

#### func (*ONNXImporter) ImportFromONNX

```go
func (oi *ONNXImporter) ImportFromONNX(path string) (*Checkpoint, error)
```
ImportFromONNX converts an ONNX model to go-metal checkpoint format

#### type OperatorSetIdProto

```go
type OperatorSetIdProto struct {
	Domain  string `protobuf:"bytes,1,opt,name=domain,proto3" json:"domain,omitempty"`
	Version int64  `protobuf:"varint,2,opt,name=version,proto3" json:"version,omitempty"`
}
```

Operator Set

#### func (*OperatorSetIdProto) Descriptor

```go
func (*OperatorSetIdProto) Descriptor() ([]byte, []int)
```
Deprecated: Use OperatorSetIdProto.ProtoReflect.Descriptor instead.

#### func (*OperatorSetIdProto) GetDomain

```go
func (x *OperatorSetIdProto) GetDomain() string
```

#### func (*OperatorSetIdProto) GetVersion

```go
func (x *OperatorSetIdProto) GetVersion() int64
```

#### func (*OperatorSetIdProto) ProtoMessage

```go
func (*OperatorSetIdProto) ProtoMessage()
```

#### func (*OperatorSetIdProto) ProtoReflect

```go
func (x *OperatorSetIdProto) ProtoReflect() protoreflect.Message
```

#### func (*OperatorSetIdProto) Reset

```go
func (x *OperatorSetIdProto) Reset()
```

#### func (*OperatorSetIdProto) String

```go
func (x *OperatorSetIdProto) String() string
```

#### type OptimizerState

```go
type OptimizerState struct {
	Type       string                 `json:"type"` // "SGD", "Adam", etc.
	Parameters map[string]interface{} `json:"parameters"`
	StateData  []OptimizerTensor      `json:"state_data"`
}
```

OptimizerState captures optimizer-specific state (momentum, variance, etc.)

#### type OptimizerTensor

```go
type OptimizerTensor struct {
	Name      string    `json:"name"`
	Shape     []int     `json:"shape"`
	Data      []float32 `json:"data"`
	StateType string    `json:"state_type"` // "momentum", "variance", "m", "v", etc.
}
```

OptimizerTensor represents optimizer state tensors (momentum, variance, etc.)

#### type SparseTensorProto

```go
type SparseTensorProto struct {
	Values  *TensorProto `protobuf:"bytes,1,opt,name=values,proto3" json:"values,omitempty"`
	Indices *TensorProto `protobuf:"bytes,2,opt,name=indices,proto3" json:"indices,omitempty"`
	Dims    []int64      `protobuf:"varint,3,rep,packed,name=dims,proto3" json:"dims,omitempty"`
}
```

SparseTensor (minimal for compatibility)

#### func (*SparseTensorProto) Descriptor

```go
func (*SparseTensorProto) Descriptor() ([]byte, []int)
```
Deprecated: Use SparseTensorProto.ProtoReflect.Descriptor instead.

#### func (*SparseTensorProto) GetDims

```go
func (x *SparseTensorProto) GetDims() []int64
```

#### func (*SparseTensorProto) GetIndices

```go
func (x *SparseTensorProto) GetIndices() *TensorProto
```

#### func (*SparseTensorProto) GetValues

```go
func (x *SparseTensorProto) GetValues() *TensorProto
```

#### func (*SparseTensorProto) ProtoMessage

```go
func (*SparseTensorProto) ProtoMessage()
```

#### func (*SparseTensorProto) ProtoReflect

```go
func (x *SparseTensorProto) ProtoReflect() protoreflect.Message
```

#### func (*SparseTensorProto) Reset

```go
func (x *SparseTensorProto) Reset()
```

#### func (*SparseTensorProto) String

```go
func (x *SparseTensorProto) String() string
```

#### type StringStringEntryProto

```go
type StringStringEntryProto struct {
	Key   string `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Value string `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
}
```

StringStringEntry

#### func (*StringStringEntryProto) Descriptor

```go
func (*StringStringEntryProto) Descriptor() ([]byte, []int)
```
Deprecated: Use StringStringEntryProto.ProtoReflect.Descriptor instead.

#### func (*StringStringEntryProto) GetKey

```go
func (x *StringStringEntryProto) GetKey() string
```

#### func (*StringStringEntryProto) GetValue

```go
func (x *StringStringEntryProto) GetValue() string
```

#### func (*StringStringEntryProto) ProtoMessage

```go
func (*StringStringEntryProto) ProtoMessage()
```

#### func (*StringStringEntryProto) ProtoReflect

```go
func (x *StringStringEntryProto) ProtoReflect() protoreflect.Message
```

#### func (*StringStringEntryProto) Reset

```go
func (x *StringStringEntryProto) Reset()
```

#### func (*StringStringEntryProto) String

```go
func (x *StringStringEntryProto) String() string
```

#### type TensorAnnotation

```go
type TensorAnnotation struct {
	TensorName                string                    `protobuf:"bytes,1,opt,name=tensor_name,json=tensorName,proto3" json:"tensor_name,omitempty"`
	QuantParameterTensorNames []*StringStringEntryProto `protobuf:"bytes,2,rep,name=quant_parameter_tensor_names,json=quantParameterTensorNames,proto3" json:"quant_parameter_tensor_names,omitempty"`
}
```

TensorAnnotation

#### func (*TensorAnnotation) Descriptor

```go
func (*TensorAnnotation) Descriptor() ([]byte, []int)
```
Deprecated: Use TensorAnnotation.ProtoReflect.Descriptor instead.

#### func (*TensorAnnotation) GetQuantParameterTensorNames

```go
func (x *TensorAnnotation) GetQuantParameterTensorNames() []*StringStringEntryProto
```

#### func (*TensorAnnotation) GetTensorName

```go
func (x *TensorAnnotation) GetTensorName() string
```

#### func (*TensorAnnotation) ProtoMessage

```go
func (*TensorAnnotation) ProtoMessage()
```

#### func (*TensorAnnotation) ProtoReflect

```go
func (x *TensorAnnotation) ProtoReflect() protoreflect.Message
```

#### func (*TensorAnnotation) Reset

```go
func (x *TensorAnnotation) Reset()
```

#### func (*TensorAnnotation) String

```go
func (x *TensorAnnotation) String() string
```

#### type TensorProto

```go
type TensorProto struct {
	Dims       []int64              `protobuf:"varint,1,rep,packed,name=dims,proto3" json:"dims,omitempty"`
	DataType   TensorProto_DataType `protobuf:"varint,2,opt,name=data_type,json=dataType,proto3,enum=checkpoints.TensorProto_DataType" json:"data_type,omitempty"`
	Segment    *TensorProto_Segment `protobuf:"bytes,3,opt,name=segment,proto3" json:"segment,omitempty"`
	FloatData  []float32            `protobuf:"fixed32,4,rep,packed,name=float_data,json=floatData,proto3" json:"float_data,omitempty"`
	Int32Data  []int32              `protobuf:"varint,5,rep,packed,name=int32_data,json=int32Data,proto3" json:"int32_data,omitempty"`
	StringData [][]byte             `protobuf:"bytes,6,rep,name=string_data,json=stringData,proto3" json:"string_data,omitempty"`
	Int64Data  []int64              `protobuf:"varint,7,rep,packed,name=int64_data,json=int64Data,proto3" json:"int64_data,omitempty"`
	Name       string               `protobuf:"bytes,8,opt,name=name,proto3" json:"name,omitempty"`
	DocString  string               `protobuf:"bytes,12,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
	RawData    []byte               `protobuf:"bytes,9,opt,name=raw_data,json=rawData,proto3" json:"raw_data,omitempty"`
	DoubleData []float64            `protobuf:"fixed64,10,rep,packed,name=double_data,json=doubleData,proto3" json:"double_data,omitempty"`
	Uint64Data []uint64             `protobuf:"varint,11,rep,packed,name=uint64_data,json=uint64Data,proto3" json:"uint64_data,omitempty"`
}
```

Tensor

#### func (*TensorProto) Descriptor

```go
func (*TensorProto) Descriptor() ([]byte, []int)
```
Deprecated: Use TensorProto.ProtoReflect.Descriptor instead.

#### func (*TensorProto) GetDataType

```go
func (x *TensorProto) GetDataType() TensorProto_DataType
```

#### func (*TensorProto) GetDims

```go
func (x *TensorProto) GetDims() []int64
```

#### func (*TensorProto) GetDocString

```go
func (x *TensorProto) GetDocString() string
```

#### func (*TensorProto) GetDoubleData

```go
func (x *TensorProto) GetDoubleData() []float64
```

#### func (*TensorProto) GetFloatData

```go
func (x *TensorProto) GetFloatData() []float32
```

#### func (*TensorProto) GetInt32Data

```go
func (x *TensorProto) GetInt32Data() []int32
```

#### func (*TensorProto) GetInt64Data

```go
func (x *TensorProto) GetInt64Data() []int64
```

#### func (*TensorProto) GetName

```go
func (x *TensorProto) GetName() string
```

#### func (*TensorProto) GetRawData

```go
func (x *TensorProto) GetRawData() []byte
```

#### func (*TensorProto) GetSegment

```go
func (x *TensorProto) GetSegment() *TensorProto_Segment
```

#### func (*TensorProto) GetStringData

```go
func (x *TensorProto) GetStringData() [][]byte
```

#### func (*TensorProto) GetUint64Data

```go
func (x *TensorProto) GetUint64Data() []uint64
```

#### func (*TensorProto) ProtoMessage

```go
func (*TensorProto) ProtoMessage()
```

#### func (*TensorProto) ProtoReflect

```go
func (x *TensorProto) ProtoReflect() protoreflect.Message
```

#### func (*TensorProto) Reset

```go
func (x *TensorProto) Reset()
```

#### func (*TensorProto) String

```go
func (x *TensorProto) String() string
```

#### type TensorProto_DataType

```go
type TensorProto_DataType int32
```

Tensor data types

```go
const (
	TensorProto_DataType_UNDEFINED  TensorProto_DataType = 0
	TensorProto_DataType_FLOAT      TensorProto_DataType = 1 // float32
	TensorProto_DataType_UINT8      TensorProto_DataType = 2 // uint8
	TensorProto_DataType_INT8       TensorProto_DataType = 3 // int8
	TensorProto_DataType_UINT16     TensorProto_DataType = 4 // uint16
	TensorProto_DataType_INT16      TensorProto_DataType = 5 // int16
	TensorProto_DataType_INT32      TensorProto_DataType = 6 // int32
	TensorProto_DataType_INT64      TensorProto_DataType = 7 // int64
	TensorProto_DataType_STRING     TensorProto_DataType = 8 // string
	TensorProto_DataType_BOOL       TensorProto_DataType = 9 // bool
	TensorProto_DataType_FLOAT16    TensorProto_DataType = 10
	TensorProto_DataType_DOUBLE     TensorProto_DataType = 11
	TensorProto_DataType_UINT32     TensorProto_DataType = 12
	TensorProto_DataType_UINT64     TensorProto_DataType = 13
	TensorProto_DataType_COMPLEX64  TensorProto_DataType = 14
	TensorProto_DataType_COMPLEX128 TensorProto_DataType = 15
	TensorProto_DataType_BFLOAT16   TensorProto_DataType = 16
)
```

#### func (TensorProto_DataType) Descriptor

```go
func (TensorProto_DataType) Descriptor() protoreflect.EnumDescriptor
```

#### func (TensorProto_DataType) Enum

```go
func (x TensorProto_DataType) Enum() *TensorProto_DataType
```

#### func (TensorProto_DataType) EnumDescriptor

```go
func (TensorProto_DataType) EnumDescriptor() ([]byte, []int)
```
Deprecated: Use TensorProto_DataType.Descriptor instead.

#### func (TensorProto_DataType) Number

```go
func (x TensorProto_DataType) Number() protoreflect.EnumNumber
```

#### func (TensorProto_DataType) String

```go
func (x TensorProto_DataType) String() string
```

#### func (TensorProto_DataType) Type

```go
func (TensorProto_DataType) Type() protoreflect.EnumType
```

#### type TensorProto_Segment

```go
type TensorProto_Segment struct {
	Begin int64 `protobuf:"varint,1,opt,name=begin,proto3" json:"begin,omitempty"`
	End   int64 `protobuf:"varint,2,opt,name=end,proto3" json:"end,omitempty"`
}
```


#### func (*TensorProto_Segment) Descriptor

```go
func (*TensorProto_Segment) Descriptor() ([]byte, []int)
```
Deprecated: Use TensorProto_Segment.ProtoReflect.Descriptor instead.

#### func (*TensorProto_Segment) GetBegin

```go
func (x *TensorProto_Segment) GetBegin() int64
```

#### func (*TensorProto_Segment) GetEnd

```go
func (x *TensorProto_Segment) GetEnd() int64
```

#### func (*TensorProto_Segment) ProtoMessage

```go
func (*TensorProto_Segment) ProtoMessage()
```

#### func (*TensorProto_Segment) ProtoReflect

```go
func (x *TensorProto_Segment) ProtoReflect() protoreflect.Message
```

#### func (*TensorProto_Segment) Reset

```go
func (x *TensorProto_Segment) Reset()
```

#### func (*TensorProto_Segment) String

```go
func (x *TensorProto_Segment) String() string
```

#### type TensorShapeProto

```go
type TensorShapeProto struct {
	Dim []*TensorShapeProto_Dimension `protobuf:"bytes,1,rep,name=dim,proto3" json:"dim,omitempty"`
}
```

Dimension of a tensor shape

#### func (*TensorShapeProto) Descriptor

```go
func (*TensorShapeProto) Descriptor() ([]byte, []int)
```
Deprecated: Use TensorShapeProto.ProtoReflect.Descriptor instead.

#### func (*TensorShapeProto) GetDim

```go
func (x *TensorShapeProto) GetDim() []*TensorShapeProto_Dimension
```

#### func (*TensorShapeProto) ProtoMessage

```go
func (*TensorShapeProto) ProtoMessage()
```

#### func (*TensorShapeProto) ProtoReflect

```go
func (x *TensorShapeProto) ProtoReflect() protoreflect.Message
```

#### func (*TensorShapeProto) Reset

```go
func (x *TensorShapeProto) Reset()
```

#### func (*TensorShapeProto) String

```go
func (x *TensorShapeProto) String() string
```

#### type TensorShapeProto_Dimension

```go
type TensorShapeProto_Dimension struct {

	// Types that are valid to be assigned to Value:
	//
	//	*TensorShapeProto_Dimension_DimValue
	//	*TensorShapeProto_Dimension_DimParam
	Value      isTensorShapeProto_Dimension_Value `protobuf_oneof:"value"`
	Denotation string                             `protobuf:"bytes,3,opt,name=denotation,proto3" json:"denotation,omitempty"`
}
```


#### func (*TensorShapeProto_Dimension) Descriptor

```go
func (*TensorShapeProto_Dimension) Descriptor() ([]byte, []int)
```
Deprecated: Use TensorShapeProto_Dimension.ProtoReflect.Descriptor instead.

#### func (*TensorShapeProto_Dimension) GetDenotation

```go
func (x *TensorShapeProto_Dimension) GetDenotation() string
```

#### func (*TensorShapeProto_Dimension) GetDimParam

```go
func (x *TensorShapeProto_Dimension) GetDimParam() string
```

#### func (*TensorShapeProto_Dimension) GetDimValue

```go
func (x *TensorShapeProto_Dimension) GetDimValue() int64
```

#### func (*TensorShapeProto_Dimension) GetValue

```go
func (x *TensorShapeProto_Dimension) GetValue() isTensorShapeProto_Dimension_Value
```

#### func (*TensorShapeProto_Dimension) ProtoMessage

```go
func (*TensorShapeProto_Dimension) ProtoMessage()
```

#### func (*TensorShapeProto_Dimension) ProtoReflect

```go
func (x *TensorShapeProto_Dimension) ProtoReflect() protoreflect.Message
```

#### func (*TensorShapeProto_Dimension) Reset

```go
func (x *TensorShapeProto_Dimension) Reset()
```

#### func (*TensorShapeProto_Dimension) String

```go
func (x *TensorShapeProto_Dimension) String() string
```

#### type TensorShapeProto_Dimension_DimParam

```go
type TensorShapeProto_Dimension_DimParam struct {
	DimParam string `protobuf:"bytes,2,opt,name=dim_param,json=dimParam,proto3,oneof"`
}
```


#### type TensorShapeProto_Dimension_DimValue

```go
type TensorShapeProto_Dimension_DimValue struct {
	DimValue int64 `protobuf:"varint,1,opt,name=dim_value,json=dimValue,proto3,oneof"`
}
```


#### type TrainingInfoProto

```go
type TrainingInfoProto struct {
	Initialization *GraphProto `protobuf:"bytes,1,opt,name=initialization,proto3" json:"initialization,omitempty"`
	Algorithm      *GraphProto `protobuf:"bytes,2,opt,name=algorithm,proto3" json:"algorithm,omitempty"`
}
```

TrainingInfo (minimal)

#### func (*TrainingInfoProto) Descriptor

```go
func (*TrainingInfoProto) Descriptor() ([]byte, []int)
```
Deprecated: Use TrainingInfoProto.ProtoReflect.Descriptor instead.

#### func (*TrainingInfoProto) GetAlgorithm

```go
func (x *TrainingInfoProto) GetAlgorithm() *GraphProto
```

#### func (*TrainingInfoProto) GetInitialization

```go
func (x *TrainingInfoProto) GetInitialization() *GraphProto
```

#### func (*TrainingInfoProto) ProtoMessage

```go
func (*TrainingInfoProto) ProtoMessage()
```

#### func (*TrainingInfoProto) ProtoReflect

```go
func (x *TrainingInfoProto) ProtoReflect() protoreflect.Message
```

#### func (*TrainingInfoProto) Reset

```go
func (x *TrainingInfoProto) Reset()
```

#### func (*TrainingInfoProto) String

```go
func (x *TrainingInfoProto) String() string
```

#### type TrainingState

```go
type TrainingState struct {
	Epoch        int     `json:"epoch"`
	Step         int     `json:"step"`
	LearningRate float32 `json:"learning_rate"`
	BestLoss     float32 `json:"best_loss"`
	BestAccuracy float32 `json:"best_accuracy"`
	TotalSteps   int     `json:"total_steps"`
}
```

TrainingState captures the current training progress

#### type TypeProto

```go
type TypeProto struct {

	// Types that are valid to be assigned to Value:
	//
	//	*TypeProto_TensorType
	Value      isTypeProto_Value `protobuf_oneof:"value"`
	Denotation string            `protobuf:"bytes,6,opt,name=denotation,proto3" json:"denotation,omitempty"`
}
```

Type of a tensor

#### func (*TypeProto) Descriptor

```go
func (*TypeProto) Descriptor() ([]byte, []int)
```
Deprecated: Use TypeProto.ProtoReflect.Descriptor instead.

#### func (*TypeProto) GetDenotation

```go
func (x *TypeProto) GetDenotation() string
```

#### func (*TypeProto) GetTensorType

```go
func (x *TypeProto) GetTensorType() *TypeProto_Tensor
```

#### func (*TypeProto) GetValue

```go
func (x *TypeProto) GetValue() isTypeProto_Value
```

#### func (*TypeProto) ProtoMessage

```go
func (*TypeProto) ProtoMessage()
```

#### func (*TypeProto) ProtoReflect

```go
func (x *TypeProto) ProtoReflect() protoreflect.Message
```

#### func (*TypeProto) Reset

```go
func (x *TypeProto) Reset()
```

#### func (*TypeProto) String

```go
func (x *TypeProto) String() string
```

#### type TypeProto_Tensor

```go
type TypeProto_Tensor struct {
	ElemType TensorProto_DataType `protobuf:"varint,1,opt,name=elem_type,json=elemType,proto3,enum=checkpoints.TensorProto_DataType" json:"elem_type,omitempty"`
	Shape    *TensorShapeProto    `protobuf:"bytes,2,opt,name=shape,proto3" json:"shape,omitempty"`
}
```


#### func (*TypeProto_Tensor) Descriptor

```go
func (*TypeProto_Tensor) Descriptor() ([]byte, []int)
```
Deprecated: Use TypeProto_Tensor.ProtoReflect.Descriptor instead.

#### func (*TypeProto_Tensor) GetElemType

```go
func (x *TypeProto_Tensor) GetElemType() TensorProto_DataType
```

#### func (*TypeProto_Tensor) GetShape

```go
func (x *TypeProto_Tensor) GetShape() *TensorShapeProto
```

#### func (*TypeProto_Tensor) ProtoMessage

```go
func (*TypeProto_Tensor) ProtoMessage()
```

#### func (*TypeProto_Tensor) ProtoReflect

```go
func (x *TypeProto_Tensor) ProtoReflect() protoreflect.Message
```

#### func (*TypeProto_Tensor) Reset

```go
func (x *TypeProto_Tensor) Reset()
```

#### func (*TypeProto_Tensor) String

```go
func (x *TypeProto_Tensor) String() string
```

#### type TypeProto_TensorType

```go
type TypeProto_TensorType struct {
	TensorType *TypeProto_Tensor `protobuf:"bytes,1,opt,name=tensor_type,json=tensorType,proto3,oneof"`
}
```


#### type ValueInfoProto

```go
type ValueInfoProto struct {
	Name      string     `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Type      *TypeProto `protobuf:"bytes,2,opt,name=type,proto3" json:"type,omitempty"`
	DocString string     `protobuf:"bytes,3,opt,name=doc_string,json=docString,proto3" json:"doc_string,omitempty"`
}
```

ValueInfo

#### func (*ValueInfoProto) Descriptor

```go
func (*ValueInfoProto) Descriptor() ([]byte, []int)
```
Deprecated: Use ValueInfoProto.ProtoReflect.Descriptor instead.

#### func (*ValueInfoProto) GetDocString

```go
func (x *ValueInfoProto) GetDocString() string
```

#### func (*ValueInfoProto) GetName

```go
func (x *ValueInfoProto) GetName() string
```

#### func (*ValueInfoProto) GetType

```go
func (x *ValueInfoProto) GetType() *TypeProto
```

#### func (*ValueInfoProto) ProtoMessage

```go
func (*ValueInfoProto) ProtoMessage()
```

#### func (*ValueInfoProto) ProtoReflect

```go
func (x *ValueInfoProto) ProtoReflect() protoreflect.Message
```

#### func (*ValueInfoProto) Reset

```go
func (x *ValueInfoProto) Reset()
```

#### func (*ValueInfoProto) String

```go
func (x *ValueInfoProto) String() string
```

#### type Version

```go
type Version struct {
	IrVersion    int64 `protobuf:"varint,1,opt,name=ir_version,json=irVersion,proto3" json:"ir_version,omitempty"`
	ModelVersion int64 `protobuf:"varint,2,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
}
```

Versioning

#### func (*Version) Descriptor

```go
func (*Version) Descriptor() ([]byte, []int)
```
Deprecated: Use Version.ProtoReflect.Descriptor instead.

#### func (*Version) GetIrVersion

```go
func (x *Version) GetIrVersion() int64
```

#### func (*Version) GetModelVersion

```go
func (x *Version) GetModelVersion() int64
```

#### func (*Version) ProtoMessage

```go
func (*Version) ProtoMessage()
```

#### func (*Version) ProtoReflect

```go
func (x *Version) ProtoReflect() protoreflect.Message
```

#### func (*Version) Reset

```go
func (x *Version) Reset()
```

#### func (*Version) String

```go
func (x *Version) String() string
```

#### type WeightTensor

```go
type WeightTensor struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
	Layer string    `json:"layer"`
	Type  string    `json:"type"` // "weight", "bias", "gamma", "beta", etc.
}
```

WeightTensor represents a model parameter tensor with its data

#### func  ExtractWeightsFromTensors

```go
func ExtractWeightsFromTensors(tensors []*memory.Tensor, modelSpec *layers.ModelSpec) ([]WeightTensor, error)
```
ExtractWeightsFromTensors extracts weight data from GPU tensors while
maintaining GPU-resident design
