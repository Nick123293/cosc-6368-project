# Project: Communication-Efficient Distributed Neural Network Training Using Gradient Compression

## 1. Project objective

I am conducting a graduate-level artificial intelligence research project investigating whether gradient compression can reduce end-to-end neural network training time in synchronous data-parallel, multi-GPU training while maintaining comparable model convergence, classification accuracy, and generalization.

For the initial implementation, I want to train a multilayer perceptron (MLP), consisting of fully connected layers, to classify handwritten letters and digits using the EMNIST dataset. I want to compare conventional FP32 gradient synchronization against compressed gradient synchronization using the quantization methods described in:

**QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding**, Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic, NeurIPS 2017.

- Paper: https://arxiv.org/abs/1610.02132
- Published version: https://proceedings.neurips.cc/paper_files/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html
- PyTorch distributed documentation: https://docs.pytorch.org/docs/stable/distributed.html

Read the paper before implementing its quantization algorithms. Pay particular attention to Algorithm 1, Section 3.1 (stochastic quantization and encoding), and Section 4 (practical QSGD variants).

The primary research question is:

**Under what conditions does the reduction in gradient communication time achieved through lossy compression outweigh the computational overhead of compression and decompression, as well as any additional training iterations required because of gradient distortion?**

This implementation will serve as the foundation of a larger research project. Prioritize correctness, reproducibility, readability, extensibility, and rigorous performance evaluation.

---

## 2. Software design philosophy

Before implementing each new component or functionality, evaluate the following questions:

1. How can this functionality be implemented such that new features can be added without substantially modifying existing code?
2. How can this implementation be easily understood and modified by both a human developer and an AI coding agent?
3. Can this functionality be separated from the rest of the implementation through a well-defined interface?
4. Does this component introduce unnecessary coupling between the neural network architecture, dataset, training procedure, distributed communication, compression algorithm, or performance instrumentation?
5. How can the correctness of this component be independently tested?

Document the resulting design decisions and their rationale concisely before implementing each major component.

Follow these software engineering principles:

- Separation of concerns.
- Composition over inheritance where appropriate.
- Explicit, well-defined interfaces between components.
- Type annotations and dataclasses for structured configuration and results.
- Descriptive function, class, and variable names.
- Small functions with clearly defined responsibilities.
- Docstrings describing inputs, outputs, tensor dimensions, devices, data types, and important assumptions.
- Minimal mutable global state.
- Minimal hardcoded configuration values.
- Avoid unnecessary dependencies and excessive software abstraction.
- Use PyTorch vectorized tensor operations rather than Python loops over individual gradient elements.
- Do not prematurely optimize at the expense of correctness or maintainability.

Adding a new compression algorithm should not require modifying the MLP, dataset-loading logic, training loop, or notebook analysis logic.

Likewise, replacing the MLP with a CNN or transformer should not require reimplementing the gradient compression and synchronization infrastructure.

---

## 3. Project structure

Initially, implement the core functionality in a single importable Python script, organized into logically independent classes and functions.

Use the following project structure:

    gradient-compression/
    ├── distributed_training.py
    ├── experiments.ipynb
    ├── requirements.txt
    ├── README.md
    ├── tests/
    │   ├── test_compression.py
    │   ├── test_distributed.py
    │   └── test_metrics.py
    └── results/

The Python script should contain the following logically independent components:

- Experiment configuration
- Distributed process management
- Model creation
- Dataset preparation and partitioning
- Gradient compression and decompression
- Gradient synchronization
- Training and evaluation
- Performance instrumentation
- Gradient error measurement
- Experiment execution
- Result serialization

Organize the script such that these components can eventually be moved into separate Python modules without changing their public interfaces.

Do not place the primary training logic inside the Jupyter notebook. The notebook should primarily handle configuration, experiment launching, hyperparameter sweeps, result loading, and visualization.

---

## 4. Experiment configuration

Create an `ExperimentConfig` dataclass containing the configurable parameters for an experiment.

At minimum, support the following categories of parameters.

### Distributed configuration

- Number of GPUs
- GPU device IDs
- Distributed backend, defaulting to NCCL for CUDA
- Process-group initialization timeout
- Number of data-loader workers
- Gradient communication bucket size

### Model configuration

- Model architecture
- Input dimensions
- Hidden-layer dimensions
- Number of hidden layers
- Activation function
- Number of output classes, inferred from the dataset where possible

### Dataset configuration

- Dataset name
- Dataset directory
- EMNIST split
- Training/validation split ratio
- Dataset split seed
- Optional training dataset size limit
- Optional validation dataset size limit

### Training configuration

- Number of epochs
- Mini-batch size per GPU
- Optional global mini-batch size
- Learning rate
- Optimizer
- Momentum
- Weight decay
- Learning rate scheduler
- Random seed
- Maximum number of training steps
- Target validation loss
- Target validation accuracy
- Early stopping configuration

### Compression configuration

- Compression method
- Quantization levels
- Quantization bit width
- Normalization method
- Quantization bucket size
- Encoding method
- Minimum tensor size for compression
- Compression-specific random seed

### Instrumentation configuration

- Gradient error measurement mode
- Gradient error measurement frequency
- Step-level metrics enabled or disabled
- Detailed timing enabled or disabled
- Per-layer error measurement enabled or disabled
- Model checkpoint saving enabled or disabled
- Number of experimental repetitions
- Output file path

Validate the configuration before starting any distributed processes.

Clearly distinguish between per-GPU mini-batch size and global mini-batch size.

For equal-sized local mini-batches:

`global_batch_size = local_batch_size * num_gpus`

When the global batch size is explicitly specified, validate that it can be divided appropriately across the selected GPUs.

---

## 5. Neural network implementation

Implement a configurable multilayer perceptron using PyTorch.

Use the following as the initial default architecture:

    Input: 28 × 28 grayscale image
        ↓
    Flatten: 784 features
        ↓
    Linear: 784 → 1024
        ↓
    ReLU
        ↓
    Linear: 1024 → 512
        ↓
    ReLU
        ↓
    Linear: 512 → 256
        ↓
    ReLU
        ↓
    Linear: 256 → num_classes

Make all hidden-layer dimensions configurable.

Use `torch.nn.CrossEntropyLoss` for classification. Return raw logits from the model rather than applying softmax before calculating the loss.

Use FP32 model parameters, activations, and gradients for the initial implementation. Do not introduce mixed-precision training because it would introduce an additional experimental variable.

Use `torch.optim.SGD` as the default optimizer, with configurable learning rate, momentum, and weight decay.

Implement a model factory such as:

    create_model(model_config, dataset_metadata) -> nn.Module

The training loop must not directly depend on the MLP architecture.

The model factory should eventually support additional architectures, including CNNs, RNNs, and transformers, without modifying the gradient synchronization infrastructure.

Allow larger MLP configurations to be selected from the notebook so that communication-intensive workloads can also be investigated.

---

## 6. Dataset loading and partitioning

Use `torchvision.datasets.EMNIST`.

Use the `byclass` split as the initial default because it includes digits, uppercase letters, and lowercase letters across 62 classes. Also support the `balanced` split and other available EMNIST splits.

Implement:

- Dataset downloading and caching.
- Image conversion to PyTorch tensors.
- Appropriate normalization.
- Verification of image orientation and label mappings.
- Reproducible training/validation partitioning.
- Independent official test-set access for final model evaluation.

Do not use the official test set as the validation dataset for hyperparameter tuning.

Create a deterministic training/validation split from the official training dataset. Ensure that every GPU and every comparable experimental run uses the same underlying split.

Use `DistributedSampler` or an equivalent reproducible partitioning mechanism to partition training samples across GPU processes.

Ensure that:

- Every GPU receives a distinct subset of training samples, except where explicitly documented sampler padding is necessary.
- Every GPU performs the same number of gradient synchronization steps.
- Training data are shuffled consistently between epochs.
- `sampler.set_epoch(epoch)` is called where applicable.
- Dropped, duplicated, and padded samples are accounted for.
- Validation loss and accuracy are not biased by duplicated validation samples.

Prefer equal-sized local mini-batches for the initial implementation. If unequal-sized local mini-batches are supported, implement sample-count-weighted gradient aggregation.

Calculate epoch-level loss using sample-weighted aggregation rather than averaging rank-level or batch-level loss values without accounting for their sample counts.

Provide a dataset factory such as:

    create_datasets(dataset_config) -> DatasetBundle

The returned object should contain the training, validation, and test datasets, as well as input dimensions, class counts, and other metadata required for model initialization.

---

## 7. Multi-GPU data-parallel training

Implement synchronous data-parallel training using `torch.distributed`.

Do not use `torch.nn.DataParallel`.

Use one distributed process per GPU. Each process should:

1. Initialize its distributed process group.
2. Select its assigned CUDA device.
3. Construct a complete model replica.
4. Synchronize initial model parameters from rank 0.
5. Initialize its optimizer.
6. Load its partition of the training dataset.
7. Perform local forward and backward passes.
8. Synchronize gradients across all participating GPUs.
9. Apply the same aggregated gradient through its optimizer.

For the initial implementation, explicitly perform gradient synchronization after `loss.backward()` rather than relying on automatic gradient synchronization from `DistributedDataParallel`.

This is intentional: I want direct control over gradient compression, communication, decompression, aggregation, and performance instrumentation.

Design the communication component such that an optimized DDP communication-hook implementation could be added in the future.

The training loop should conceptually resemble:

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            loss.backward()

            synchronization_result = gradient_synchronizer.synchronize(
                model.parameters()
            )

            optimizer.step()

            metrics.record_step(
                loss=loss,
                synchronization_result=synchronization_result
            )

        evaluate_model()
        metrics.record_epoch()

All replicas must apply mathematically equivalent aggregated gradients and remain synchronized after each optimizer step, within floating-point numerical tolerances.

Explicitly avoid performing both automatic DDP synchronization and manual synchronization on the same gradients.

Support:

- Single-GPU execution for debugging and reference experiments.
- Multi-GPU execution using NCCL.
- CPU/Gloo execution for distributed correctness tests.

Ensure that process groups are destroyed correctly when experiments finish or encounter exceptions.

---

## 8. Gradient synchronization interface

Separate gradient synchronization from the training loop.

Create an interface conceptually similar to:

    class GradientSynchronizer:
        def synchronize(self, parameters) -> SynchronizationResult:
            ...

The synchronizer should handle:

- Collecting model gradients.
- Flattening gradients into contiguous communication buffers.
- Partitioning gradients into configurable communication buckets.
- Invoking the selected compression algorithm.
- Exchanging the encoded gradient representations.
- Decompressing received gradients.
- Aggregating gradients across ranks.
- Restoring the original gradient shapes.
- Writing aggregated gradients back to model parameters.
- Returning synchronization metrics.

Keep gradient layout metadata consistent across all ranks.

Distinguish between:

- **Communication buckets:** groups of gradients communicated in a collective operation.
- **Quantization buckets:** groups of gradient elements that share a quantization normalization factor.

These are separate parameters and should not be treated as interchangeable.

### Required synchronization modes

Implement three synchronization strategies.

**1. `fp32_allreduce`**

Use `torch.distributed.all_reduce()` on the full-precision gradient buffer, followed by division by the world size.

This is the primary conventional data-parallel baseline.

**2. `fp32_allgather`**

Use `torch.distributed.all_gather()` or an appropriate tensor-based equivalent to exchange uncompressed gradients. Each GPU should then aggregate and average the received gradients.

This provides a communication-pattern-matched baseline for compressed all-gather.

**3. `compressed_allgather`**

Compress local gradients, exchange compressed representations, decompress the received representations, and aggregate the reconstructed gradients.

This strategy should accept any compatible gradient compressor.

For equal-sized local mini-batches, the uncompressed aggregated gradient is:

`g_global = (1 / world_size) * sum(g_local_rank)`

The compressed aggregated gradient is:

`g_global_quantized = (1 / world_size) * sum(decompress(compress(g_local_rank)))`

Account for every rank's gradient exactly once, including the current rank.

The current rank must use its own reconstructed gradient in the compressed aggregation so that every GPU aggregates the same set of values.

### Important communication constraint

Do not blindly apply `all_reduce()` to independently quantized integer gradients.

Different ranks may use different normalization factors, and consequently, directly summing their quantized integer values generally does not produce a valid aggregated gradient.

For the initial QSGD implementation, exchange encoded gradient representations and their associated normalization metadata using tensor-based all-gather operations.

Decompress each rank's gradient independently before aggregation.

Avoid Python object collectives, such as `all_gather_object()`, in the performance-critical path.

Document the communication and memory complexity of the synchronization strategies, particularly the scaling limitations of all-gather relative to all-reduce.

---

## 9. Gradient compression interface

Implement a common interface for gradient compression algorithms.

For example:

    class GradientCompressor:
        def compress(self, gradient) -> CompressedGradient:
            ...

        def decompress(self, compressed_gradient) -> torch.Tensor:
            ...

Use a structured `CompressedGradient` representation containing, as appropriate:

- Encoded tensor payload
- Original tensor shape
- Original tensor data type
- Quantization normalization factors
- Quantization levels
- Encoding metadata
- Original number of elements
- Original size in bytes
- Encoded payload size in bytes
- Metadata size in bytes

Distinguish between metadata required by the receiver and metadata that is already known from the shared model configuration.

The gradient compressor should not directly depend on distributed process management or the neural network architecture.

Similarly, the distributed communication implementation should not contain QSGD-specific mathematical logic.

Create a compressor factory such as:

    create_compressor(compression_config) -> GradientCompressor

Implement the following compression methods.

### 9.1. Identity compressor

Implement an identity compressor for testing and uncompressed communication.

Compression and decompression must preserve the input exactly.

This provides a simple method for testing the generalized compression and communication pipeline independently of lossy quantization.

### 9.2. QSGD L2-norm stochastic quantization

Implement the original stochastic quantization operator described in Section 3.1 of the QSGD paper.

For a gradient vector `g` and a positive integer quantization level parameter `s`, calculate:

    a = ||g||₂

    u_i = s * |g_i| / a

    l_i = floor(u_i)

    p_i = u_i - l_i

Generate an independent Bernoulli random variable for each gradient element:

    B_i ~ Bernoulli(p_i)

Calculate the quantized magnitude:

    k_i = l_i + B_i

Reconstruct the gradient as:

    Q_s(g_i) = a * sign(g_i) * k_i / s

The compressed representation should contain the information required to reconstruct the quantized gradient, rather than storing reconstructed FP32 values as the transmitted payload.

Handle:

- Zero-norm gradients.
- Zero-valued gradient elements.
- Floating-point rounding near quantization boundaries.
- Maximum representable quantization levels.
- Non-contiguous input tensors.
- Arbitrary supported gradient dimensions.
- Partial quantization buckets.

Use independent, reproducible quantization random-number streams for each rank, separate from model initialization, dataset shuffling, and other training randomness.

Document the mathematical expectation that the quantizer is unbiased, subject to numerical precision:

    E[Q_s(g)] = g

Also document the L2-normalized QSGD quantization error bound described in the paper.

### 9.3. Bucket-based QSGD

Implement configurable bucket-based quantization.

Flatten each gradient buffer and partition it into fixed-sized groups of elements. Quantize each group independently using its own normalization factor.

Support configurable quantization bucket sizes, including:

- 128
- 512
- 2048
- 8192
- Entire gradient buffer

Handle the final partial bucket correctly.

The goal is to investigate the trade-off between quantization distortion and normalization metadata overhead.

Do not implement the bucket-based method as an entirely separate duplicate compressor. Reuse the underlying stochastic quantization implementation.

### 9.4. Maximum-magnitude QSGD

Implement the practical normalization variant discussed in Section 4 of the QSGD paper.

Replace the L2 normalization factor with:

    a = max(abs(g))

Perform the same stochastic quantization and reconstruction procedure using this normalization factor.

Expose this through a configurable normalization method, for example:

    normalization = "l2"

or:

    normalization = "max"

Clearly distinguish the theoretical L2-normalized method from the practical maximum-magnitude variant.

Do not automatically attribute all theoretical sparsity or communication guarantees of the original L2-normalized method to the maximum-magnitude variant.

### 9.5. Gradient encoding and physical compression

Separate stochastic quantization from encoding.

Quantization reduces the number of representable gradient values, but communication savings are only achieved when these values are represented using fewer transmitted bytes.

Implement actual compact representations, including:

- Signed integer encoding using appropriate integer tensor types.
- Fixed-width bit packing for sub-byte quantization, including 2-bit and 4-bit configurations.
- An 8-bit representation.
- Storage and transmission of the associated normalization factors.

Explicitly define the relationship between the QSGD parameter `s`, the number of signed representable values, and the number of bits required.

For example, a signed quantization index with values from `-s` through `s` requires sufficient capacity to represent `2s + 1` possible values.

Do not assume that `s = 2**bit_width` fits into a representation of `bit_width` bits.

Ensure that the configured quantization levels can actually be represented without overflow or information loss beyond the intended stochastic quantization.

Packing and unpacking should use vectorized PyTorch operations and remain on the GPU whenever possible.

Also implement the Elias-based gradient encoding approach described in Section 3.1 of the paper as a separate, independently testable reference encoder after the fixed-width implementation is validated.

If the Elias encoder cannot be implemented efficiently using GPU-native operations, clearly label it as a reference implementation and report its CPU execution and host/device transfer overhead. Do not silently substitute fixed-width encoding while claiming to reproduce the paper's complete encoding algorithm.

If variable-length encoded messages require communicating message lengths and padding buffers, account for this communication overhead.

Measure actual communicated tensor sizes, including scale factors, metadata, and padding.

Do not report a 4-bit compression ratio if the actual implementation transmits one 8-bit or 32-bit value per gradient element.

---

## 10. Performance instrumentation

The project is primarily concerned with reducing end-to-end training time, not merely reducing the size of gradient representations.

Therefore, implement detailed, independently configurable performance instrumentation.

### 10.1. Experiment-level timing

Record:

- Total experiment wall-clock runtime
- Distributed initialization time
- Dataset preparation time
- Total training runtime
- Total validation runtime
- Final test evaluation runtime
- Total gradient synchronization runtime
- Total compression runtime
- Total decompression runtime
- Total collective communication runtime
- Total gradient aggregation runtime
- Total data-loading runtime

Clearly distinguish the complete experiment runtime from the training-only runtime.

Dataset downloading and initial setup should not contaminate the primary training performance comparisons.

### 10.2. Epoch-level timing

For each epoch, record:

- Epoch wall-clock runtime
- Training runtime
- Validation runtime
- Forward-pass runtime
- Backward-pass runtime
- Gradient preparation and flattening runtime
- Compression and encoding runtime
- Collective communication runtime
- Decompression and decoding runtime
- Gradient aggregation runtime
- Gradient restoration runtime
- Optimizer runtime
- Data-loading and host-to-device transfer runtime
- Cumulative training time

### 10.3. Training-step timing

Optionally record the same timing breakdown for each individual training step.

Implement step-level instrumentation without requiring step-level file writes.

Use `time.perf_counter()` for appropriate CPU and wall-clock measurements and CUDA events for GPU operation timing.

Account for asynchronous CUDA execution and NCCL communication.

Do not use unsynchronized CPU timers around asynchronous CUDA operations and assume the resulting measurements represent completed GPU execution.

Avoid introducing unnecessary `torch.cuda.synchronize()` calls, barriers, tensor-to-CPU copies, or `.item()` operations in the normal training path.

### 10.4. Separate benchmarking and profiling modes

Implement two instrumentation modes.

**Benchmark mode:**

- Minimize instrumentation overhead.
- Measure representative end-to-end training runtime.
- Disable expensive gradient error diagnostics.
- Avoid additional gradient communication.
- Retain lightweight experiment-level and epoch-level metrics.

**Diagnostic mode:**

- Enable detailed timing.
- Enable gradient reconstruction error measurements.
- Enable additional reference gradient synchronization where necessary.
- Allow per-step and per-layer analysis.

Do not directly compare a diagnostic-mode QSGD runtime against a benchmark-mode baseline runtime.

Report diagnostic-mode timing separately from primary performance results.

For distributed timing, distinguish the maximum rank runtime, mean rank runtime, and individual rank measurements where useful.

Do not sum elapsed times across GPUs and report that sum as the parallel wall-clock runtime.

---

## 11. Communication volume and compression metrics

For each gradient synchronization operation, record:

- Original gradient size in bytes
- Encoded gradient payload size in bytes
- Normalization metadata size
- Additional transmitted metadata
- Communication padding
- Number of communicated tensors
- Number of collective calls
- Compression ratio
- Effective bits per gradient element
- Collective communication runtime
- End-to-end synchronization runtime

Calculate:

    compression_ratio = original_bytes / encoded_total_bytes

    effective_bits_per_element = 8 * encoded_total_bytes / num_elements

Distinguish between:

1. The size of each rank's gradient representation.
2. The logical communication volume of the selected collective operation.
3. Estimated aggregate communication traffic across ranks.
4. Actual physical interconnect traffic, if hardware profiling is available.

Do not claim that tensor payload sizes represent measured physical interconnect traffic.

Clearly document assumptions used to estimate communication volume for all-gather and all-reduce.

Account for all compression and decompression work when determining whether compressed synchronization is faster than uncompressed synchronization.

---

## 12. Gradient reconstruction error

Implement rigorous measurement of the error introduced by gradient quantization.

Do not calculate quantization error by directly comparing gradients from two independently trained models. Their parameters will diverge after quantization is introduced, so their gradients are no longer evaluated at the same model state.

Implement two distinct forms of gradient error.

### 12.1. Local gradient reconstruction error

For each rank, let:

    g = original local gradient
    g_hat = decompress(compress(g))

Calculate the difference:

    e = g_hat - g

Calculate:

    MSE = sum(e**2) / num_elements

    RMSE = sqrt(MSE)

    max_error = max(abs(e))

Aggregate these statistics across all participating ranks and gradient tensors.

This measures distortion introduced directly by the compression algorithm.

Use the actual quantized representation generated during training. Do not independently quantize the gradient a second time for error measurement.

### 12.2. Aggregated gradient reconstruction error

Measure the distortion in the gradient actually passed to the optimizer.

For the original local gradients from the same training step, calculate the reference full-precision aggregate:

    g_reference = mean(g_local_rank)

Calculate the compressed aggregate:

    g_quantized = mean(g_hat_local_rank)

Then calculate:

    e_global = g_quantized - g_reference

Report the same MSE, RMSE, and maximum absolute error metrics.

The reference aggregate must use the same local gradients, model parameters, and mini-batches as the compressed synchronization.

When required, use an additional full-precision `all_reduce()` to calculate the reference gradient.

This additional collective must only occur in diagnostic mode and must not influence the gradient used by the optimizer.

Record diagnostic collective calls and runtimes separately.

### 12.3. Error aggregation

For each step, epoch, and entire experiment, maintain streaming accumulators containing:

- Sum of squared errors
- Sum of absolute errors
- Number of gradient elements evaluated
- Maximum absolute error

Calculate total and epoch-level MSE by dividing the accumulated sum of squared errors by the accumulated number of evaluated elements.

Calculate RMSE as the square root of the aggregated MSE.

Do not calculate total RMSE by simply averaging step-level RMSE values.

Calculate maximum error using the maximum across all included observations.

Support optional additional metrics:

- Mean absolute error
- Relative L2 error
- Gradient cosine similarity
- Gradient L2 norm
- Gradient sparsity before and after quantization

Record local reconstruction errors separately from aggregated gradient reconstruction errors.

For aggregated error, avoid treating identical replicas of the global gradient on different ranks as independent observations.

Support optional per-layer error analysis using the stored gradient layout metadata.

Avoid storing complete gradient tensors for every training step. Store summary statistics instead.

---

## 13. Training and convergence metrics

Record the following metrics at step, epoch, or experiment granularity as appropriate:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Final test loss and accuracy
- Learning rate
- Number of optimizer steps
- Number of local mini-batches processed
- Number of global mini-batches processed
- Number of training samples processed
- Number of logical gradient synchronization rounds
- Number of actual distributed collective calls
- Number of completed epochs
- Cumulative runtime
- Training throughput in samples per second

Calculate training loss using sample-weighted aggregation.

Clearly distinguish training loss accumulated during the optimization process from loss evaluated using the final model at the end of an epoch. If an optional end-of-epoch training evaluation is implemented, record its cost separately.

Distinguish between a logical gradient synchronization round and the number of collective calls issued.

For example, one optimizer step may involve multiple collectives because of gradient bucketing, quantization scale communication, or variable-length encoding metadata.

### 13.1. Fixed-training-budget experiments

Train each configuration for the same number of epochs or optimizer steps using the same data and hyperparameters.

Compare:

- Total runtime
- Per-epoch runtime
- Training and validation loss
- Classification accuracy
- Gradient reconstruction error
- Communication volume

For identical GPU counts, batch sizes, and epoch counts, the number of optimizer steps and logical synchronization rounds should generally be identical.

Gradient compression alone should not change the number of training steps in a fixed-budget experiment.

### 13.2. Convergence-target experiments

Support training until a configurable target validation loss or target validation accuracy is reached.

Also support a maximum epoch or step budget.

Record:

- Runtime required to reach the target
- Number of epochs required
- Number of global mini-batches required
- Number of optimizer steps required
- Number of logical gradient synchronization rounds required
- Total gradient communication required

Use the same validation frequency and target definition for all configurations.

Record the first evaluation at which the target is achieved. Do not claim finer-grained time-to-target measurements than the validation frequency permits.

If the model does not reach the specified target, explicitly record that the target was not reached within the experimental budget.

Do not incorrectly report the final runtime as the time required to reach an unattained target.

Ensure that stopping decisions are communicated consistently to all ranks so that no rank exits while another is waiting on a collective operation.

---

## 14. Experimental methodology

Ensure comparisons between compressed and uncompressed training are scientifically meaningful.

For each paired experiment, maintain identical:

- Model architecture
- Initial model parameters
- Dataset partitioning
- Dataset shuffling
- Mini-batch order
- Learning rate
- Optimizer configuration
- Number of GPUs
- Mini-batch size
- Learning rate scheduling
- Validation procedure
- Training budget or convergence target

The compression method should initially be the only independent variable.

Maintain separate random-number generators for stochastic gradient quantization and other training operations.

Ensure that enabling gradient quantization does not inadvertently change dataset sampling or model initialization.

Support multiple random seeds and repeated experimental trials.

Report mean and standard deviation across repetitions. Do not treat individual training steps as independent experimental repetitions.

### GPU scaling experiments

Support two distinct GPU scaling configurations.

**Fixed global mini-batch size:**

When the GPU count increases, decrease the local mini-batch size accordingly.

This allows strong-scaling experiments with comparable global optimization steps.

**Fixed local mini-batch size:**

When the GPU count increases, the global mini-batch size increases.

This investigates the effect of processing more training samples concurrently, but it also changes the optimization dynamics.

Label these experiments separately.

Do not conflate their results.

### Additional controls

Support:

- Repeated runs with different experiment execution orders.
- Optional GPU warm-up before primary measurements.
- Restoring model parameters, optimizer state, and relevant random states after warm-up, or using a separate warm-up workload.
- Recording GPU model, PyTorch version, CUDA version, NCCL version, and available hardware topology information.
- Recording whether experiments use PCIe, NVLink, or another known interconnect.
- Recording the total number of model parameters and gradient buffer size.

Use the same instrumentation settings for comparable performance experiments.

Do not assume compressed training will necessarily be faster. Report slowdowns, failed convergence targets, and negative results without filtering them out.

---

## 15. Jupyter notebook integration

Create a fully functional `experiments.ipynb` that imports the Python implementation.

The notebook should contain the following sections.

### Section 1: Imports and environment information

Import the training module, PyTorch, pandas, NumPy, and Matplotlib.

Display the available GPU count and relevant GPU properties.

### Section 2: Experiment configuration

Expose all major configuration parameters in a clearly identifiable cell.

For example:

    NUM_GPUS = 2

    LOCAL_BATCH_SIZE = 128

    EPOCHS = 10

    LEARNING_RATE = 0.01

    HIDDEN_DIMS = [1024, 512, 256]

    EMNIST_SPLIT = "byclass"

    QUANTIZATION_BITS = [2, 4, 8]

    QUANTIZATION_BUCKET_SIZES = [128, 512, 2048]

    NORMALIZATION_METHODS = ["l2", "max"]

    SEEDS = [42, 43, 44]

    OUTPUT_FILE = "./results/experiments.json"

These are initial experimental defaults rather than fixed requirements.

The user should be able to change these values without editing the training script.

### Section 3: Experiment execution

Provide a straightforward public interface such as:

    results = run_experiment(config)

    results = run_experiments(configurations)

    results = load_results(output_file)

The notebook should support both individual experiments and configurable parameter sweeps.

Use a notebook-safe distributed launch mechanism. Prefer launching the importable Python script through `torchrun` using a subprocess rather than defining distributed worker functions directly inside the notebook.

The script should also be executable independently from the command line.

Avoid recursive process creation, CUDA fork initialization issues, and multiple ranks independently launching new experiments.

Include an appropriate `if __name__ == "__main__":` entry point.

Run paired experiments sequentially by default so that experiments do not compete for the same GPUs.

Ensure failed subprocesses propagate useful error messages to the notebook.

Do not automatically launch a prohibitively large parameter sweep. Provide a small initial experiment suite and separately configurable expanded sweeps.

### Section 4: Runtime comparisons

Generate:

- Total training runtime comparison bar charts.
- Per-epoch runtime line graphs.
- Per-step runtime graphs.
- Cumulative runtime graphs.
- Stacked runtime breakdown charts.
- Gradient synchronization runtime comparisons.
- Compression and decompression overhead comparisons.
- Training throughput comparisons.
- Speedup versus GPU count.

Calculate speedup as:

    speedup = baseline_runtime / compressed_runtime

Calculate runtime reduction percentage as:

    runtime_reduction_percent = (
        (baseline_runtime - compressed_runtime)
        / baseline_runtime
    ) * 100

Include both the FP32 all-reduce baseline and FP32 all-gather baseline where applicable.

Do not display diagnostic-mode timings as primary benchmark-mode results.

### Section 5: Training and validation loss comparisons

Generate:

- Training loss versus epoch.
- Validation loss versus epoch.
- Training loss versus optimizer step.
- Training loss versus cumulative runtime.
- Validation loss versus cumulative runtime.
- Training and validation accuracy versus epoch.
- Validation accuracy versus cumulative runtime.
- Final training and validation loss comparison.
- Final test accuracy comparison.

Use epoch and optimizer-step plots to compare optimization efficiency.

Use wall-clock-time plots to compare practical training efficiency.

### Section 6: Convergence comparisons

Generate comparisons of:

- Epochs required to reach a specified validation loss.
- Mini-batches required to reach the target.
- Gradient synchronization rounds required to reach the target.
- Actual collective operations required to reach the target.
- Training time required to reach the target.
- Total communication volume required to reach the target.

Explicitly identify configurations that fail to reach the target.

### Section 7: Gradient reconstruction error analysis

Generate:

- Local gradient MSE versus epoch.
- Local gradient RMSE versus epoch.
- Local gradient maximum absolute error versus epoch.
- Aggregated gradient MSE versus epoch.
- Aggregated gradient RMSE versus epoch.
- Aggregated gradient maximum absolute error versus epoch.
- Gradient error versus training step.
- Total gradient error versus quantization bit width.
- Gradient error versus quantization bucket size.
- Gradient error versus compression ratio.

Compare L2-normalized and maximum-magnitude QSGD.

Support optional logarithmic axes for error metrics spanning multiple orders of magnitude.

### Section 8: Relationship between quantization error and convergence

For paired experiments, calculate:

    delta_train_loss = compressed_train_loss - baseline_train_loss

    delta_val_loss = compressed_val_loss - baseline_val_loss

    delta_val_accuracy = (
        compressed_val_accuracy - baseline_val_accuracy
    )

Calculate these differences at matching epochs or optimizer steps where appropriate.

Generate:

- Training loss difference versus epoch.
- Validation loss difference versus epoch.
- Gradient MSE versus training loss difference.
- Gradient MSE versus validation loss difference.
- Gradient RMSE versus final validation accuracy.
- Compression ratio versus final validation loss.
- Compression ratio versus time to convergence.
- Quantization bit width versus runtime and final accuracy.
- Quantization bucket size versus runtime and gradient distortion.

Where appropriate, calculate descriptive Pearson or Spearman correlations between gradient reconstruction error and loss differences.

Do not interpret correlation as proof that quantization distortion caused a particular convergence outcome.

Clearly identify whether a comparison is aligned by epoch, optimizer step, processed sample count, or wall-clock runtime.

### Section 9: Summary tables

Generate a pandas DataFrame containing one summary row per experiment.

Include:

- Experiment ID
- Compression method
- Normalization method
- Quantization bit width
- Quantization bucket size
- Encoding method
- GPU count
- Local and global mini-batch sizes
- Total model parameters
- Total runtime
- Mean epoch runtime
- Mean step runtime
- Total synchronization runtime
- Total compression runtime
- Total decompression runtime
- Compression ratio
- Communication volume
- Final training loss
- Final validation loss
- Final validation accuracy
- Final test accuracy
- Number of optimizer steps
- Number of synchronization rounds
- Number of collective calls
- Total local gradient MSE, RMSE, and maximum error
- Total aggregated gradient MSE, RMSE, and maximum error
- Time to convergence target
- Target reached status
- Speedup relative to the matching baseline

Support exporting summary tables to CSV.

---

## 16. Result storage

Store all relevant experimental data in an output file specified inside the Jupyter notebook.

Use a structured JSON file as the primary experiment result format.

Each experiment should have a unique experiment ID and contain:

    {
        "schema_version": 1,
        "experiment_id": "...",
        "status": "...",
        "configuration": {},
        "hardware": {},
        "software_environment": {},
        "dataset_metadata": {},
        "model_metadata": {},
        "summary": {},
        "epoch_metrics": [],
        "step_metrics": [],
        "gradient_error_metrics": {},
        "communication_metrics": {},
        "convergence_metrics": {}
    }

Support storing multiple experiments in the same output file.

Prevent previously completed experiments from being accidentally overwritten.

Record the full experiment configuration rather than only the parameters that differ from the defaults.

Only the designated coordinator process should write experiment results. Avoid concurrent writes from multiple distributed ranks.

Use atomic writes where practical.

Store unavailable or unmeasured values as null rather than falsely representing them as zero.

Support optional separate CSV or Parquet files for large step-level metric collections, referenced from the primary JSON file.

The result schema should allow additional metrics to be added in the future without requiring existing experiment files or notebook analysis code to be substantially rewritten.

The notebook should be able to reload and analyze previously completed experiments without retraining the models.

---

## 17. Correctness and testing

Before conducting full training experiments, implement and execute correctness tests.

### Compression unit tests

Test:

- Exact identity compression and decompression.
- Zero-valued gradient vectors.
- Positive, negative, and mixed-sign gradients.
- Gradient vectors with large differences in element magnitudes.
- Partial quantization buckets.
- Non-contiguous tensors.
- Correct preservation of tensor shape and device.
- Quantization index range.
- Bit-packing and unpacking correctness.
- Correct decoding of normalization factors.
- Actual encoded representation size.
- Deterministic results given the same compression seed.
- Independent quantization randomness across ranks.

Test the empirical unbiasedness of stochastic quantization by repeatedly quantizing a fixed gradient vector and comparing the mean reconstructed gradient against the original vector using statistically appropriate tolerances.

For the L2-normalized quantizer, additionally test the expected squared-error bound empirically.

### Distributed correctness tests

Use small synthetic datasets and models.

Verify that:

1. All GPU replicas start with identical model parameters.
2. Each distributed process receives its intended training partition.
3. FP32 all-reduce produces the expected averaged gradient.
4. FP32 all-gather produces an equivalent averaged gradient.
5. Uncompressed distributed training approximately matches single-process training using an equivalent global mini-batch.
6. Compressed aggregation matches an independently calculated reference from the actual encoded gradients.
7. All model replicas remain synchronized after optimizer updates.
8. Every rank performs the same required collective operations.
9. Early stopping terminates all distributed processes correctly.
10. Enabling gradient error measurement does not change the gradients applied by the optimizer.

### Metrics tests

Verify that:

- MSE, RMSE, and maximum error match independently calculated reference values.
- Epoch-level and experiment-level error aggregation correctly weight gradient element counts.
- Loss aggregation correctly weights sample counts.
- Byte counts include transmitted metadata and padding.
- Synchronization-round counters are distinct from collective-call counters.
- Diagnostic communication is excluded from primary benchmark communication statistics.
- Runtime speedup calculations are correct.
- Missing convergence targets are handled correctly.

Provide CPU-based tests that can run without GPUs and separate integration tests that exercise real multi-GPU NCCL communication.

Do not claim that multi-GPU performance or correctness has been validated if the execution environment does not provide multiple CUDA GPUs.

---

## 18. Implementation order

Implement the project incrementally in the following order.

**Phase 1: Single-device training**

- Configuration classes.
- EMNIST dataset loading.
- Configurable MLP.
- Single-device training and validation.
- Basic metrics and result storage.

**Phase 2: Uncompressed distributed training**

- Distributed process initialization.
- Model replication.
- Distributed dataset partitioning.
- Explicit FP32 all-reduce synchronization.
- FP32 all-gather synchronization.
- Distributed correctness tests.

**Phase 3: Gradient compression**

- Common compressor interface.
- Identity compressor.
- L2-normalized QSGD.
- Bucket-based quantization.
- Maximum-magnitude QSGD.
- Fixed-width integer encoding and bit packing.
- Compressed gradient synchronization.
- Compression and distributed correctness tests.
- Elias reference encoder.

**Phase 4: Instrumentation**

- End-to-end runtime measurements.
- Detailed synchronization timing.
- Communication volume measurements.
- Local and aggregated gradient error metrics.
- Benchmark and diagnostic execution modes.
- Convergence-target training.

**Phase 5: Experimental notebook**

- Hyperparameter configuration.
- Experiment launching.
- Paired baseline/compressed experiments.
- Parameter sweeps.
- Result storage and reloading.
- Comparative tables.
- Performance and convergence visualizations.

**Phase 6: Validation and documentation**

- Execute available tests.
- Conduct small smoke-test training experiments.
- Inspect generated result files.
- Verify notebook compatibility.
- Document limitations and assumptions.
- Provide instructions for extending the project.

Before implementing each phase, briefly explain its design, component responsibilities, interfaces, and potential extensibility concerns.

After implementing each phase, review the code for correctness, unnecessary coupling, and opportunities to simplify the implementation.

If you have access to a suitable execution environment, run the applicable tests before proceeding to dependent functionality.

---

## 19. Future extensibility requirements

The architecture should accommodate future additions such as:

### Compression algorithms

- FP16 gradient communication
- BF16 gradient communication
- Uniform deterministic quantization
- Nonuniform quantization
- Top-k gradient sparsification
- Error-feedback compression
- PowerSGD
- Sign-based gradient compression
- Error-bounded lossy compression
- Lossless gradient compression
- Custom GPU-native compressors

### Neural network architectures

- CNNs
- RNNs
- LSTMs
- Transformers
- Larger fully connected models

### Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet
- Custom datasets

### Distributed communication methods

- NCCL all-reduce
- NCCL all-gather
- Reduce-scatter
- Custom point-to-point communication
- DDP communication hooks
- Overlapping communication and computation
- Multi-node training

### Experimental metrics

- GPU utilization
- Memory consumption
- Communication/computation overlap
- Interconnect bandwidth utilization
- Energy consumption
- Per-layer compression sensitivity
- Gradient distribution statistics
- Adaptive compression selection

Do not implement all of these future features now.

Instead, establish sufficiently simple and stable interfaces that allow them to be implemented later without major architectural changes.

Use registration mechanisms or small factories where appropriate, but avoid building an unnecessarily complicated plugin framework.

---

## 20. Documentation and final deliverables

Create all requested project files.

The README must explain:

1. Project objectives.
2. Mathematical foundations of synchronous data-parallel SGD.
3. Mathematical foundations of QSGD.
4. Differences between the original QSGD algorithm and the implemented practical variants.
5. Software architecture and component responsibilities.
6. Installation instructions.
7. Single-GPU execution.
8. Multi-GPU execution using `torchrun`.
9. Jupyter notebook execution.
10. Experiment configuration.
11. Performance measurement methodology.
12. Gradient error measurement methodology.
13. Result file structure.
14. How to add a new compression algorithm.
15. How to add a new neural network architecture.
16. How to add a new dataset.
17. How to add a new experimental metric.
18. Known limitations and future optimization opportunities.

Include a concise architecture diagram and describe the flow of gradients between the model, gradient synchronizer, compressor, distributed communication layer, decompressor, and optimizer.

Document every significant deviation from the QSGD paper and explain why that deviation was made.

Provide an example command-line execution and an example notebook configuration.

Do not fabricate performance measurements. If sufficient hardware is unavailable, provide functional tests and instructions for collecting real measurements.

The completed implementation should be a reproducible research framework, not simply a demonstration that gradient quantization can be integrated into PyTorch.

Prioritize correctness first, experimental validity second, extensibility and readability third, and performance optimization fourth.

Begin by inspecting the existing project directory, if one exists. Then present a concise implementation plan and proceed with creating the requested files. Do not stop after providing only the plan or pseudocode.