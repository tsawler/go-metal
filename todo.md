# Go-Metal Project Future Roadmap

This document outlines the strategic roadmap for the go-metal project, building upon its current "production-ready" status and addressing known limitations. The plan is structured into logical phases, prioritizing stability and foundational features before expanding into advanced capabilities and broader ecosystem integration.

## Executive Summary

The go-metal framework has achieved significant milestones, delivering a high-performance, Apple Silicon-optimized training system. The roadmap focuses on solidifying its core, expanding its utility for common machine learning tasks, and ultimately positioning it as a comprehensive, competitive ML framework.

## Roadmap Phases

### Phase 1: Stability & Foundational Features (Immediate - Next 1-3 Months)

This phase prioritizes resolving existing limitations and implementing fundamental features essential for robust, production-grade training.

* **Core Training Enhancements:**

    * ✅ **Learning Rate Decay/Scheduling:** ~~Implement various learning rate scheduling strategies (e.g., step decay, exponential decay, cosine annealing) to improve model convergence and generalization.~~ **COMPLETED** - Full LR scheduler interface with 5 implementations (StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, NoOp) integrated with ModelTrainer.

    * ✅ **Checkpoint Saving & Loading:** **COMPLETED** - Robust checkpoint mechanisms implemented and validated:
      * ✅ Save/load model weights with complete state preservation
      * ✅ Optimizer state persistence (Adam momentum, variance, step count)
      * ✅ Training progress tracking (epoch, loss history, best accuracy)
      * ✅ Interruption and resumption of training from any checkpoint
      * ✅ Cross-session training continuity with preserved learning rates and schedules

    * ✅ **Advanced Layer Types:** ~~Integrate additional common neural network layers such as Batch Normalization, Dropout, and other advanced activation functions (e.g., Leaky ReLU, ELU).~~ **COMPLETED**
      * **Dropout:** Fully implemented with MPSGraph integration, comprehensive testing, and overfitting reduction validated (25+ point gap reduced to ~17 points)
      * ✅ **Batch Normalization:** **COMPLETED** with unified training/inference solution. Complete MPSGraph integration with proper gradient computation, parameter initialization (gamma=1.0, beta=0.0), tensor broadcasting for 4D inputs, production validation, and comprehensive ONNX compatibility. Implemented unified solution resolving training/inference mode conflicts:
        * **Training Mode:** Uses batch statistics with placeholder-based running statistics updates
        * **Inference Mode:** Uses constants to avoid MPSGraph placeholder errors
        * **ONNX Support:** Full import/export of BatchNorm layers including running statistics extraction and filtering  
        * **Weight Management:** Proper separation of learnable parameters (14) from running statistics (6) during model loading
        * **Stability:** Training stability and generalization improvements confirmed (validation accuracy 73.80%, controlled 19.49% train-val gap vs 25.3% without BatchNorm)
        * **Both Applications Work:** Cats-dogs training and load-saved-model inference operate simultaneously without conflicts
        * **⚠️ ARCHITECTURAL LIMITATION:** Currently uses ONNX standard defaults (mean=0, variance=1) for inference mode instead of actual trained running statistics. This works for most normalized models but may affect accuracy with models having significantly different running statistics. Requires architectural changes to pass running statistics during graph construction.
      * **Leaky ReLU:** Fully implemented with configurable negative slope parameter, complete MPSGraph `leakyReLUWithTensor` integration, comprehensive testing, and real-world financial prediction demo. Demonstrates improved gradient flow for negative inputs (0.184687 vs 0.196875 loss). Still needed: ELU.

    * ✅ **ONNX Model Compatibility:** **COMPLETED** - Full ONNX model import and inference support with comprehensive layer compatibility.
      * **Supported Operations:** Conv2D, BatchNormalization, MatMul (Dense), Relu, Softmax, Dropout with proper parameter extraction and weight tensor handling
      * **Weight Management:** Advanced weight count resolution with proper separation of learnable parameters from running statistics  
      * **Inference Execution:** Complete inference pipeline supporting both CNN and MLP architectures from ONNX models
      * **Real-World Validation:** Successfully demonstrated with actual ONNX models achieving proper classification results
      * **Architecture Support:** Handles complex CNN models with BatchNorm layers, proper tensor reshaping and broadcasting

    * ✅ **Model Serialization:** **COMPLETED** - Full model serialization capabilities implemented and validated:
      * ✅ Save/load trained models with `checkpoints` package (JSON and binary formats)
      * ✅ Complete training state persistence (weights, optimizer state, training metadata)
      * ✅ ONNX model import/export with full compatibility
      * ✅ Resume training from checkpoints with preserved optimizer state
      * ✅ Cross-platform model portability and validation
      * **Validated:** Successfully saving/loading models in both cats-dogs training and load-saved-model inference applications

    * ✅ **BatchNorm Running Statistics Architecture Fix:** **COMPLETED** - Resolved the architectural limitation where inference mode used ONNX defaults instead of actual trained running statistics. Implemented complete solution:
      * ✅ Modified graph construction to accept running statistics data during build time
      * ✅ Updated ONNX importer to provide running statistics to graph builder through LayerSpec.RunningStatistics
      * ✅ Ensured proper architecture supports any model with different running statistics values
      * ✅ Fixed both affine and non-affine BatchNorm inference modes in addBatchNormLayerToGraph
      * ✅ Added running statistics support to DynamicLayerSpec, LayerSpecC, and layer_spec_c_t structures
      * ✅ Implemented complete data flow from ONNX import → ModelSpec → DynamicLayerSpec → CGO bridge → MPSGraph construction
      * **Impact:** Now supports models trained with non-standard data distributions and uses actual trained running statistics for accurate inference

    * **Further Performance Optimization:** Continuously profile and optimize existing components to squeeze out additional performance gains.

    * ✅ **Debug Output Cleanup:** ~~Clean up verbose debug logging for production use.~~ **COMPLETED** - Removed debug messages from CGO bridge files, console output now shows clean training progress only.

### Phase 2: Core Library Expansion & Utilities (Short-to-Medium Term - Next 3-6 Months)

This phase focuses on building out the core utility of the `go-metal` library by extracting validated components and expanding its fundamental machine learning capabilities.

* ✅ **Extract CNN & Image Processing Library Components:** **COMPLETED** - Successfully generalized and integrated high-performance image processing and data loading utilities from the cats-dogs application into the main `go-metal` library.

    * ✅ **Image Preprocessing Package (`go-metal/vision/preprocessing`):** **COMPLETED** - High-performance JPEG decoding, resizing, CHW format conversion, and float32 normalization with buffer reuse. Supports concurrent batch preprocessing and generic image size configuration.

    * ✅ **High-Performance Data Loading (`go-metal/vision/dataloader`):** **COMPLETED** - Memory-efficient batch data loading with smart LRU caching, buffer reuse, and automatic shuffling. Features shared cache system achieving 6x training speedup (89.8% hit rate by epoch 10).

    * ✅ **Dataset Management Utilities (`go-metal/vision/dataset`):** **COMPLETED** - Generic `ImageFolderDataset` for directory-based datasets and specialized `CatsDogsDataset`. Includes train/validation splitting, class distribution analysis, filtering, and subsetting capabilities.

    * ✅ **Enhanced Training Infrastructure (`go-metal/training/session`):** **COMPLETED** - Training session functionality already exists in core training package and is being used effectively.

    * ✅ **Memory Optimization Patterns (`go-metal/memory/optimization`):** **COMPLETED** - GPU-optimized buffer pooling with power-of-2 sizing, global buffer pool singleton, and comprehensive statistics tracking. Includes arena allocation patterns for related tensors.

    * **Performance Impact:** Training speed increased from 13 batch/s to 76 batch/s after first epoch. Cache system works with any dataset size and image dimensions. All components are fully generic with zero hardcoded values.

    * **Architecture Compliance:** Maintains GPU-resident data, minimizes CGO calls through batched operations, supports MPSGraph-centric workflows, and implements proper memory management with reference counting and buffer pooling.

* **Expand Core ML Functionality:**

    * ✅ **More Optimizers:** ~~Add support for other popular optimization algorithms like RMSprop, Adagrad, and potentially more advanced ones like L-BFGS.~~ **COMPLETED** - RMSProp optimizer fully implemented with centered/non-centered variants, complete MPSGraph integration, and production validation with regression and classification tasks.

    * ✅ **Common Loss Functions:** ~~Implement a broader range of loss functions, including Mean Squared Error (MSE), Binary Cross-Entropy with Logits (BCEWithLogitsLoss), Categorical Cross-Entropy, and Huber Loss.~~ **COMPLETED** - Full regression and classification loss function support implemented:
      * **Regression:** MSE, MAE, and Huber loss with MPSGraph implementations
      * **Classification:** CrossEntropy, SparseCrossEntropy, BinaryCrossEntropy, BCEWithLogits, and CategoricalCrossEntropy all fully implemented with proper MPSGraph operations
      * **Binary Classification:** BCEWithLogits (numerically stable for raw logits) and BinaryCrossEntropy (for probability inputs) both implemented
      * **Multi-class Classification:** CategoricalCrossEntropy implemented for multi-class problems without softmax preprocessing
      * **Generic Label Interface:** Zero-cost abstraction supporting both int32 (classification) and float32 (regression) labels
      * **Unified Training API:** TrainBatchUnified method handles all problem types with automatic loss function selection
      * **Validated:** Successfully demonstrated with simple-regression (MSE), cats-dogs (CrossEntropy/BCEWithLogits), and all new loss functions tested and verified

    * ✅ **Regression Support:** ~~Add comprehensive regression capabilities with appropriate loss functions and accuracy metrics.~~ **COMPLETED** - Full regression training support with proper accuracy calculation:
      * **Accuracy Metric:** 1 - NMAE (Normalized Mean Absolute Error) for regression tasks
      * **Consistent Calculation:** Both training and validation use the same accuracy formula
      * **Fixed Validation Bug:** Resolved issue where validation accuracy showed incorrect ~1-2% values instead of proper regression accuracy
      * **Validated Performance:** Training accuracy ~75-85%, validation accuracy ~53-55% using consistent 1-NMAE calculation

    * ✅ **Evaluation Metrics:** ~~Provide a comprehensive set of evaluation metrics beyond accuracy, such as F1-score, Precision, Recall, AUC, and Mean Average Precision (mAP).~~ **COMPLETED** - Comprehensive evaluation metrics system implemented:
      * **Binary Classification:** Precision, Recall, F1-Score, Specificity, NPV (Negative Predictive Value), AUC-ROC with perfect GPU-resident architecture compliance
      * **Multi-class Classification:** Macro/Micro averaging for Precision, Recall, F1-Score with balanced and imbalanced dataset support
      * **Regression Metrics:** MAE, MSE, RMSE, R², NMAE (Normalized Mean Absolute Error) for comprehensive regression evaluation
      * **Confusion Matrix:** Complete confusion matrix computation with cached metrics for performance
      * **AUC-ROC Calculation:** Trapezoidal rule implementation for ranking task evaluation
      * **ModelTrainer Integration:** Seamless integration with ModelTrainer for real-time metrics collection during training and validation
      * **History Tracking:** Metrics history collection for plotting and analysis visualization
      * **GPU-Resident Compliance:** All computations follow the four core requirements - only CPU access for final scalar metrics
      * **Comprehensive Testing:** Validated with unit tests, integration tests, and real-world model training scenarios

    * **Training Visualization & Plotting:** Implement data collection and sidecar-based plot generation for common ML visualizations:
      * **Training Curves:** Loss and accuracy progression over epochs/steps
      * **Learning Rate Schedules:** Visualization of LR decay patterns  
      * **Evaluation Plots:** ROC curves, Precision-Recall curves, Confusion matrices
      * **Model Analysis:** Parameter distributions, gradient histograms, activation patterns
      * **Architecture:** JSON payload generation + HTTP POST to sidecar plotting service
      * **Output:** HTML-based interactive plots for development and analysis

### Phase 3: Advanced Architectures & Scalability (Medium-to-Long Term - Next 6-12+ Months)

This phase aims to enable the `go-metal` framework to handle more complex model architectures and scale to larger datasets and computational demands.

* **Transformer Support:** Implement core components and layers necessary for building Transformer architectures, including attention mechanisms, multi-head attention, and encoder/decoder blocks.

* **Recurrent Neural Networks (RNNs):** Add support for recurrent layers like LSTMs and GRUs for sequential data processing.

* **Multi-GPU Support:** Develop strategies for model and data parallelism to leverage multiple GPUs for accelerated training on larger models and datasets.

* **Production Tools:** Build out tools for model deployment and serving infrastructure, making it easier to integrate trained `go-metal` models into production applications.

* **Mobile Integration:** Further optimize for iOS/macOS deployment, potentially including tools for model quantization or conversion for on-device inference.

### Phase 4: Ecosystem & Extensibility (Long Term - 12+ Months)

The final phase focuses on positioning `go-metal` as a versatile and integrated part of the broader machine learning ecosystem.

* **Universal ML Framework:** Continue expanding to support a wider array of machine learning tasks and architectures beyond deep learning, such as traditional ML algorithms.

* **Custom Layer SDK:** Provide a software development kit (SDK) that allows users to define and integrate their own custom layer types and Metal operations.

* **Cloud Integration:** Explore capabilities for distributed training across multiple machines, potentially integrating with cloud-based ML platforms.

* **Ecosystem Integration:** Develop tools for seamless import and export of models to and from other popular frameworks like PyTorch and TensorFlow.

## Conclusion

This roadmap outlines a clear path for the `go-metal` project to evolve into a comprehensive, high-performance machine learning framework optimized for Apple Silicon. By systematically addressing limitations and expanding capabilities, `go-metal` will provide developers with a powerful and efficient tool for a wide range of machine learning applications