# Gradient compression in synchronous data-parallel training

This project measures when gradient compression reduces communication enough to offset encoding, decoding, and any extra optimization steps. It compares standard FP32 all-reduce, FP32 all-gather, and compressed all-gather on an FP32 EMNIST classifier. The default MLP has 784 inputs, hidden widths of 1024, 512, and 256 with ReLU activations, and 62 outputs for EMNIST ByClass.

## Gradient synchronization and QSGD

Each worker processes a different local mini-batch and computes the gradient of its mean cross-entropy loss. With \(P\) equally sized mini-batches, the global gradient is the mean of the \(P\) local gradients. Every worker starts from the rank-zero model and applies the same update. For plain SGD this is \(w_{t+1} = w_t - \eta\,\bar g_t\). With compression, workers average all reconstructed gradients, including their own reconstruction.

Following QSGD, for each bucket with scale \(a = \lVert g\rVert_2\) and \(s\) positive magnitude levels, the quantizer calculates \(u_i = s\lvert g_i\rvert/a\). It rounds \(u_i\) to one of its two adjacent integers with probability \(u_i - \lfloor u_i\rfloor\) of rounding up, then retains the original sign. The receiver multiplies each signed index by \(a/s\). A zero scale encodes an all-zero bucket. In exact arithmetic, \(\mathbb{E}[Q_s(g)] = g\). For an L2 bucket of \(d\) elements, Lemma 3.1 bounds \(\mathbb{E}\lVert Q_s(g)-g\rVert_2^2\) by \(\min(d/s^2,\sqrt d/s)\lVert g\rVert_2^2\). [QSGD paper, §3.1 and Lemma 3.1](https://arxiv.org/pdf/1610.02132)

The default practical codec packs signed indices into 2-, 4-, or 8-bit slots and sends scales as a separate tensor. For width \(b\), \(2s+1\) values must fit into \(2^b\) slots. The default levels are 1, 7, and 127, respectively. The `integer` codec uses one signed byte per index at every quantization width. Smaller quantization buckets trade more scale bytes for lower distortion. The maximum-magnitude variant uses \(a = \max_i\lvert g_i\rvert\) and preserves the stochastic rounding expectation. The L2 sparsity guarantee applies to the L2 variant. [QSGD paper, §4](https://arxiv.org/pdf/1610.02132)

The `elias` codec is a CPU reference for the paper's sparse Elias omega code. It stores the gaps between nonzero entries, their signs, and their magnitudes. A terminal gap makes a byte-aligned stream decodable, and scales travel separately so the stream can span multiple quantization buckets. The tensor collective exchanges lengths before gathering payloads padded to the largest length. This framing and padding adapt the paper's peer messages to PyTorch collectives. The reference codec adds CPU work and, on CUDA, host/device transfers. These costs have their own timing fields. The experiment also uses PyTorch collective implementations, configurable optimizers, and optional learning-rate scheduling. [QSGD paper, Algorithm 1 and Appendix A.2](https://arxiv.org/pdf/1610.02132)

## Implementation

```text
ExperimentConfig
  ├── dataset factory → training shard and disjoint validation/test shards
  ├── model factory → FP32 model replica → forward and backward
  └── gradient synchronizer
       local gradients → flatten → communication buckets
           ├── FP32 all-reduce → divide by worker count
           ├── FP32 all-gather → average peer buffers
           └── compressor → gather packets and scales → decompress → average
       averaged buckets → restore parameter gradients → optimizer
                         ↘ timings, byte counts, and optional reconstruction errors
```

`distributed_training.py` contains the configuration, data and model factories, compressor interface, explicit synchronization loop, instrumentation, and result writer. Each process owns its model, optimizer, and data loader. The launcher starts a fresh worker through Python or single-node `torchrun`. The notebook configures experiments, starts runs sequentially, and reads their result files for analysis. [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)

## Setup and execution

Use Python 3.10 or later on Linux. Select matching PyTorch and torchvision builds for your hardware before installing the remaining requirements. Multi-GPU runs require a CUDA build with NCCL. CPU correctness runs use Gloo.

```bash
cd cosc-6368-project/gradient-compression
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The first EMNIST run downloads into `data_dir`. You can point `data_dir` at an existing torchvision cache. Only rank zero downloads. For a slow first download, prepare the cache with one worker or set a longer `timeout_seconds` for a multi-worker run. The optional `dataset="synthetic"` uses a small deterministic classification problem and makes offline functional checks possible. [Official EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), [Torchvision EMNIST loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)

A short single-GPU baseline run uses plain Python.

```bash
python distributed_training.py --set device=cuda --set num_gpus=1 \
  --set epochs=1 --set train_size_limit=512 --set validation_size_limit=128 \
  --set evaluate_test=false
```

A two-GPU run uses `torchrun` with one local rank per GPU. Run the same command twice more with `synchronization=fp32_allreduce` and `synchronization=fp32_allgather` for the paired controls.

```bash
torchrun --standalone --nproc-per-node=2 distributed_training.py \
  --set device=cuda --set num_gpus=2 --set epochs=3 \
  --set synchronization=compressed_allgather --set quantization_bits=4 \
  --set quantization_bucket_size=512 --set output_file=results/paired.json
```

Use `--config path/to/config.json` to pass a JSON object and repeat `--set NAME=VALUE` to override individual fields. Use JSON for lists and booleans. For CPU/Gloo with two local ranks, keep the `torchrun` command and set `device=cpu`, `num_gpus=0`, and `cpu_processes=2`. GPU IDs are indices in the process's visible CUDA device list.

Start `jupyter lab experiments.ipynb` from this directory. The notebook loads prior results by default. Its controls enable a small paired suite, optional diagnostics, and an expanded sweep. You can also use the public API from a notebook or script.

```python
from dataclasses import replace
from distributed_training import ExperimentConfig, run_experiments

baseline = ExperimentConfig(
    device="cuda", num_gpus=2, local_batch_size=128,
    epochs=3, seed=42, output_file="results/paired.json",
)
gather = replace(baseline, synchronization="fp32_allgather")
qsgd = replace(baseline, synchronization="compressed_allgather",
               normalization="max", quantization_bits=4)
results = run_experiments([baseline, gather, qsgd], order_seed=42)
```

## Experiment controls

| Settings | What they control |
| --- | --- |
| `device`, `num_gpus`, `gpu_ids`, `cpu_processes`, `backend`, `timeout_seconds` | Single-node worker placement and process group. `device="auto"` chooses CUDA where available for the requested devices and can fall back to one CPU worker with the default GPU request. |
| `hidden_dims`, `activation`, `input_dims`, `num_classes` | Model shape. The dataset supplies its input shape and class count by default. An empty list of hidden widths gives a linear classifier. |
| `emnist_split`, `validation_fraction`, `split_seed`, dataset size limits | Dataset partition. All six torchvision splits are supported. The training split supplies validation. The official test split supplies the optional final evaluation. |
| `local_batch_size`, `global_batch_size`, `batch_scaling` | Local batch per rank and global batch across all ranks. A supplied global size determines the local size and must be divisible by the worker count. Use `fixed_global` to label strong-scaling runs and `fixed_local` when each worker's batch stays fixed. |
| `epochs`, `max_steps`, optimizer and scheduler settings | Training budget and optimizer. Set validation targets to measure first success at an evaluation boundary. All configured targets must be met. `stop_on_target=False` continues collecting a fixed-budget trajectory. Early stopping monitors validation loss. |
| `communication_bucket_bytes`, `quantization_bucket_size`, `min_compression_elements` | Maximum FP32 gradient span per communication bucket, number of elements that share a scale, and minimum original parameter size eligible for compression. Use `null` for one scale per communication bucket. |
| `compression`, `normalization`, `quantization_bits`, `quantization_levels`, `encoding` | Compressor and wire format. `identity` exercises the compressed all-gather interface with exact FP32 payloads. |
| `seed`, `compression_seed`, `repetitions`, `warmup_steps` | Model/data randomness, independent rank-local stochastic rounding, repeated measurements, and optional warm-up using a separate model and optimizer. |
| `mode`, `detailed_timing`, `step_metrics`, `gradient_error`, `gradient_error_frequency`, `per_layer_error` | Measurement cost and granularity. Detailed timing and gradient diagnostics require `mode="diagnostic"`. |
| `output_file`, `step_metrics_format`, `save_checkpoint` | Result destination, inline or external per-step rows, and optional final model weights. |

The loader transposes the stored EMNIST images to upright orientation and maps pixel values to \([-1,1]\). It validates image dimensions and the selected split's label range. The Letters split maps raw labels 1–26 to class indices 0–25. Training shards use a shared permuted index order each epoch. They drop the incomplete global batch, and the results distinguish these dropped samples from samples left unprocessed by a step budget. Validation and test use strided shards with no padding. Reported training loss and accuracy come from the mini-batches used during optimization. Evaluation aggregates summed loss and correct predictions by sample count. [Torchvision label mappings](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py), [EMNIST image orientation](https://www.tensorflow.org/datasets/catalog/emnist)

## Measurement and comparison

- **Timing.** Benchmark mode measures completed training wall time with lightweight epoch instrumentation. `training_runtime` is the sum of the slowest rank's training time in each epoch. It includes loading and optimizer work. Worker wall time includes setup, warm-up, training, validation, and the final test. On CUDA, detailed diagnostic intervals use CUDA events on the training stream. Host loading and the CPU Elias reference use host timers. Detailed epoch phase totals use the training wall-time rank selected for that epoch. Rank totals and the mean rank training time are available. For benchmark-mode CUDA runs, unmeasured internal timings are `null` or absent from the timing dictionary.
- **Communication.** A logical synchronization round occurs once per optimizer step. Each bucket can cause one or more gradient collective calls. Original and encoded sizes describe rank-zero gradient representations, including transmitted scales, length metadata, the unused slots in the packed bytes, and its variable-length padding. `logical_bytes` totals the all-reduce input or the all-gather output per rank. Estimated aggregate ring traffic is \(2(P-1)B\) for an all-reduce of \(B\) bytes and \(P(P-1)B\) for an all-gather with \(B\) equal-sized bytes from each rank after padding. Counts cover gradient collectives, with separate diagnostic call counts. Physical interconnect traffic is unavailable. `interconnect` can record a known link type, and visible GPU properties and available `nvidia-smi` topology are recorded.
- **Gradient error.** Local error compares the actual decoded packet to its own rank's original gradient. Aggregate error compares the update with a separate FP32 reference reduction of the same local gradients. The reference is used for diagnostics only. MSE and MAE weight each observed element, RMSE is the square root of the pooled MSE, and maximum absolute error takes the maximum over observations. Aggregate error counts one observation per gradient coordinate; optional per-layer reporting uses named parameter tensors.
- **Pairing.** Use the same model, initialization, sampler, local and global batch sizes, optimizer, training budget, validation schedule, and instrumentation in each comparison. The notebook compares training time to both FP32 baselines, uses matched epochs or steps for optimization deltas, and shows benchmark and diagnostic timing separately. Report trial-level means and standard deviations over repeated runs. Only epoch-end validation can mark a target achieved. An unmet target has no time to target. Use validation results for tuning and reserve official test evaluation for the final configurations.

## Results and extensions

Results append to a versioned JSON document with an `experiments` list. Each record contains its ID, status, complete configuration, environment, hardware, model and dataset metadata, summary, epoch and optional step rows, gradient error, communication, and convergence statistics. Optional external steps use relative CSV or Parquet references. CSV uses JSON cells so nested per-step fields round-trip. Install `pyarrow` for Parquet. `load_results` and `load_step_metrics` reopen stored runs. Final model state dictionaries and external steps live beside the output file in its artifact directory. Rank zero writes, and the writer locks and atomically replaces the JSON file while preserving completed records. Missing measurements use JSON `null` or absent optional timing keys.

- **Compression.** Implement the `GradientCompressor` protocol, return a compatible `CompressedGradient`, and register the name in `create_compressor` and the config validator. Test the codec independently and compare the distributed update against a reference reconstructed from its actual packets.
- **Models.** Add a branch in `create_model` and validate the configuration there and in `ExperimentConfig`. The trainer expects batched inputs and logits. New architectures with mutable buffers or unused parameters need explicit distributed handling.
- **Datasets.** Add a dataset branch returning `DatasetBundle` and its input shape, classes, and distinct evaluation splits. Extend validation and rank-zero download handling if it needs a shared cache.
- **Metrics.** Use additive sufficient statistics in the scope where the observations occur, serialize at the epoch or experiment boundary, and add notebook analysis for the new fields. Keep diagnostics out of benchmark measurements.

The implementation uses dense FP32 gradients with a static parameter layout, synchronous collectives, and one node. FP32 all-reduce uses \(O(B)\) buffer space per rank for a bucket, while all-gather holds \(O(PB)\) received data per rank. Compressed all-gather still scales with the worker count. Each peer's packet is decoded on every rank. Fixed-width packing uses vectorized PyTorch operations and Elias coding runs on the CPU. Future work can explore reusable communication buffers, fused packing, overlapping communication and compute, and a DDP communication hook. Use repeated runs on the target hardware to draw performance conclusions.
