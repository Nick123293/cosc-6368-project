"""Reproducible synchronous data-parallel experiments for EMNIST and QSGD."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import fcntl
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset, TensorDataset


# Configuration is serialized in full and validated before a process group starts.
# Batch-size resolution is shared by single-device and distributed runs.
@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for a single research configuration and its repeated trials."""

    device: str = "auto"
    num_gpus: int = 1
    gpu_ids: tuple[int, ...] | None = None
    cpu_processes: int = 1
    cpu_threads: int = 1
    backend: str | None = None
    timeout_seconds: int = 180
    interconnect: str | None = None
    num_workers: int = 0
    communication_bucket_bytes: int = 4 * 1024**2

    model: str = "mlp"
    input_dims: tuple[int, ...] | None = None
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    num_hidden_layers: int | None = None
    activation: str = "relu"
    num_classes: int | None = None

    dataset: str = "emnist"
    data_dir: str = "./data"
    emnist_split: str = "byclass"
    validation_fraction: float = 0.1
    split_seed: int = 42
    train_size_limit: int | None = None
    validation_size_limit: int | None = None
    test_size_limit: int | None = None
    evaluate_test: bool = True

    epochs: int = 10
    local_batch_size: int = 128
    global_batch_size: int | None = None
    batch_scaling: str = "fixed_local"
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    scheduler: str = "none"
    seed: int = 42
    max_steps: int | None = None
    target_validation_loss: float | None = None
    target_validation_accuracy: float | None = None
    stop_on_target: bool = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    synchronization: str = "fp32_allreduce"
    compression: str = "qsgd"
    quantization_levels: int | None = None
    quantization_bits: int = 4
    normalization: str = "l2"
    quantization_bucket_size: int | None = 512
    encoding: str = "packed"
    min_compression_elements: int = 0
    compression_seed: int = 1000

    mode: str = "benchmark"
    gradient_error: str = "none"
    gradient_error_frequency: int = 1
    step_metrics: bool = False
    detailed_timing: bool = False
    per_layer_error: bool = False
    save_checkpoint: bool = False
    repetitions: int = 1
    warmup_steps: int = 0
    output_file: str = "./results/experiments.json"
    step_metrics_format: str = "inline"

    def __post_init__(self) -> None:
        # JSON arrays and notebook lists have the same in-memory representation.
        for name in ("gpu_ids", "input_dims", "hidden_dims"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in ("data_dir", "output_file"):
            object.__setattr__(self, name, os.fspath(getattr(self, name)))

    def validate(self) -> None:
        """Validate research settings and local backend/device availability."""
        choices = {
            "device": ("auto", "cpu", "cuda"),
            "backend": (None, "gloo", "nccl"),
            "model": ("mlp",),
            "activation": ("relu", "gelu", "tanh"),
            "dataset": ("emnist", "synthetic"),
            "emnist_split": ("byclass", "bymerge", "balanced", "letters", "digits", "mnist"),
            "batch_scaling": ("fixed_local", "fixed_global"),
            "optimizer": ("sgd", "adamw"),
            "scheduler": ("none", "cosine"),
            "synchronization": ("fp32_allreduce", "fp32_allgather", "compressed_allgather"),
            "compression": ("qsgd", "identity"),
            "normalization": ("l2", "max"),
            "quantization_bits": (2, 4, 8),
            "encoding": ("packed", "integer", "elias"),
            "mode": ("benchmark", "diagnostic"),
            "gradient_error": ("none", "local", "aggregated", "both"),
            "step_metrics_format": ("inline", "csv", "parquet"),
        }
        for name, options in choices.items():
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}")
        for name in (
            "timeout_seconds", "cpu_processes", "cpu_threads", "local_batch_size", "epochs",
            "gradient_error_frequency", "repetitions", "quantization_bits",
        ):
            _validate_integer(name, getattr(self, name), 1)
        for name in (
            "global_batch_size", "max_steps", "train_size_limit", "validation_size_limit",
            "test_size_limit", "early_stopping_patience", "quantization_bucket_size",
            "quantization_levels", "num_classes",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_integer(name, value, 1)
        for name in (
            "num_gpus", "num_workers", "min_compression_elements", "warmup_steps",
            "seed", "split_seed", "compression_seed",
        ):
            _validate_integer(name, getattr(self, name), 0)
        _validate_integer("communication_bucket_bytes", self.communication_bucket_bytes, 4)
        for name in ("input_dims", "hidden_dims", "gpu_ids"):
            for value in getattr(self, name) or ():
                _validate_integer(name, value, 0 if name == "gpu_ids" else 1)
        if self.num_hidden_layers is not None:
            _validate_integer("num_hidden_layers", self.num_hidden_layers, 0)
            if self.num_hidden_layers != len(self.hidden_dims):
                raise ValueError("num_hidden_layers must match the length of hidden_dims")
        if self.input_dims == ():
            raise ValueError("input_dims must contain at least one dimension")
        if self.input_dims and self.dataset == "emnist" and math.prod(self.input_dims) != 784:
            raise ValueError("EMNIST input_dims must contain 784 features")
        if self.num_classes is not None and self.dataset == "emnist":
            expected_classes = {
                "byclass": 62, "bymerge": 47, "balanced": 47,
                "letters": 26, "digits": 10, "mnist": 10,
            }
            if self.num_classes != expected_classes[self.emnist_split]:
                raise ValueError("num_classes must match the selected EMNIST split")
        for name in ("learning_rate", "weight_decay", "momentum", "early_stopping_min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.momentum >= 1:
            raise ValueError("learning_rate must be positive and momentum must be below 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")
        if self.target_validation_loss is not None and (
            not math.isfinite(self.target_validation_loss) or self.target_validation_loss < 0
        ):
            raise ValueError("target_validation_loss must be finite and nonnegative")
        if self.target_validation_accuracy is not None and not 0 <= self.target_validation_accuracy <= 1:
            raise ValueError("target_validation_accuracy must be between 0 and 1")
        if self.quantization_levels and self.quantization_levels > 2 ** (self.quantization_bits - 1) - 1:
            raise ValueError("quantization_bits cannot represent all signed quantization levels")
        if self.mode == "benchmark" and (self.gradient_error != "none" or self.detailed_timing):
            raise ValueError("Use mode='diagnostic' for gradient error or detailed timing")
        if self.per_layer_error and self.gradient_error == "none":
            raise ValueError("per_layer_error requires gradient_error to be enabled")
        if self.step_metrics_format != "inline" and not self.step_metrics:
            raise ValueError("External step storage requires step_metrics=True")
        if self.step_metrics_format == "parquet":
            try:
                metadata.version("pyarrow")
            except metadata.PackageNotFoundError as error:
                raise ValueError("Parquet step storage requires pyarrow") from error
        if self.batch_scaling == "fixed_global" and self.global_batch_size is None:
            raise ValueError("fixed_global requires global_batch_size")
        if not self.output_file or not self.data_dir:
            raise ValueError("output_file and data_dir must be nonempty paths")
        plan = resolve_execution(self)
        if self.train_size_limit is not None and self.train_size_limit < plan.global_batch_size:
            raise ValueError("train_size_limit must allow at least one full global batch")


def _validate_integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must contain integers >= {minimum}")


@dataclass(frozen=True)
class ExecutionPlan:
    device_type: str
    device_ids: tuple[int, ...]
    backend: str
    world_size: int
    local_batch_size: int

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.world_size

    def device(self, local_rank: int) -> torch.device:
        if self.device_type == "cuda":
            return torch.device("cuda", self.device_ids[local_rank])
        return torch.device("cpu")


def resolve_execution(config: ExperimentConfig) -> ExecutionPlan:
    """Resolve visible device indices, process count and equal-sized local batches."""
    available = 0
    if config.device != "cpu" and config.num_gpus and torch.cuda.is_available():
        available = torch.cuda.device_count()
    use_cuda = config.device == "cuda" or (config.device == "auto" and available > 0)
    if not use_cuda and (config.gpu_ids is not None or (config.device == "auto" and config.num_gpus > 1)):
        raise ValueError("Requested GPUs are unavailable. Use device='cpu' and cpu_processes for Gloo.")
    device_ids = config.gpu_ids if config.gpu_ids is not None else tuple(range(config.num_gpus))
    if use_cuda and (
        not device_ids or len(device_ids) != config.num_gpus
        or len(set(device_ids)) != len(device_ids)
        or any(index >= available for index in device_ids)
    ):
        raise ValueError("gpu_ids must select num_gpus distinct visible CUDA devices")
    if use_cuda and config.cpu_processes != 1:
        raise ValueError("cpu_processes only applies to CPU execution")
    backend = config.backend or ("nccl" if use_cuda else "gloo")
    if backend != ("nccl" if use_cuda else "gloo"):
        raise ValueError("Supported combinations are NCCL/CUDA and Gloo/CPU")
    world_size = len(device_ids) if use_cuda else config.cpu_processes
    if world_size > 1 and not getattr(dist, f"is_{backend}_available")():
        raise ValueError(f"This PyTorch build does not provide {backend}")
    local_batch = config.local_batch_size
    if config.global_batch_size is not None:
        local_batch, remainder = divmod(config.global_batch_size, world_size)
        if remainder or not local_batch:
            raise ValueError("global_batch_size must be divisible by the worker count")
    return ExecutionPlan(
        "cuda" if use_cuda else "cpu", device_ids if use_cuda else (), backend, world_size, local_batch
    )


# Dataset metadata defines the model input/output contract. Dataset and shuffle
# generators are independent so changes to a compressor cannot alter the samples.
@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    split: str | None
    input_dims: tuple[int, ...]
    classes: tuple[str, ...]
    train_samples: int
    validation_samples: int
    test_samples: int | None
    orientation: str | None = None
    label_offset: int = 0
    normalization_mean: float | None = None
    normalization_std: float | None = None

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset | None
    metadata: DatasetMetadata


class EMNISTTransform:
    """Convert PIL grayscale images to upright FP32 tensors with range [-1, 1]."""

    def __call__(self, image: Any) -> torch.Tensor:
        from torchvision.transforms.functional import to_tensor

        return to_tensor(image).transpose(-2, -1).contiguous().sub_(0.5).div_(0.5)


def _zero_index_letter(label: int) -> int:
    return label - 1


def _validate_emnist(dataset: Dataset, classes: tuple[str, ...], offset: int) -> None:
    target_min, target_max = torch.aminmax(dataset.targets)
    if (int(target_min), int(target_max)) != (offset, len(classes) - 1 + offset):
        raise ValueError("EMNIST labels do not match the selected split's class mapping")
    if dataset.data.shape[1:] != (28, 28):
        raise ValueError("EMNIST images must have the official 28 × 28 layout")


def _subset_samples(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _synthetic_data(config: ExperimentConfig) -> tuple[Dataset, Dataset, tuple[int, ...], tuple[str, ...]]:
    dims = config.input_dims or (1, 8, 8)
    classes = config.num_classes or 4
    generator = torch.Generator().manual_seed(_rank_seed(config.split_seed, 0))
    weights = torch.randn(math.prod(dims), classes, generator=generator, dtype=torch.float32)

    def sample(count: int) -> Dataset:
        inputs = torch.randn(count, *dims, generator=generator, dtype=torch.float32)
        targets = (inputs.reshape(count, -1) @ weights).argmax(dim=1)
        return TensorDataset(inputs, targets)

    return sample(1024), sample(256), dims, tuple(map(str, range(classes)))


def create_datasets(config: ExperimentConfig, *, download: bool = True) -> DatasetBundle:
    """Load the official split, partition train/validation, and expose the held-out test set."""
    if config.dataset == "emnist":
        from torchvision.datasets import EMNIST

        offset = int(config.emnist_split == "letters")
        kwargs = {
            "root": config.data_dir,
            "split": config.emnist_split,
            "download": download,
            "transform": EMNISTTransform(),
            "target_transform": _zero_index_letter if offset else None,
        }
        training = EMNIST(train=True, **kwargs)
        test = EMNIST(train=False, **kwargs) if config.evaluate_test else None
        classes = tuple(training.classes[offset:])
        _validate_emnist(training, classes, offset)
        if test is not None:
            _validate_emnist(test, classes, offset)
        dims = (1, 28, 28)
        orientation = "transpose height and width from stored IDX order"
        mean = std = 0.5
    elif config.dataset == "synthetic":
        training, test, dims, classes = _synthetic_data(config)
        offset = 0
        orientation = mean = std = None
    else:
        raise ValueError(f"Unknown dataset {config.dataset!r}")

    if config.num_classes is not None and config.num_classes != len(classes):
        raise ValueError("num_classes must match the dataset's class count")
    if not config.evaluate_test:
        test = None
    split_generator = torch.Generator().manual_seed(_rank_seed(config.split_seed, 0))
    permutation = torch.randperm(len(training), generator=split_generator)
    validation_size = max(1, min(len(training) - 1, round(len(training) * config.validation_fraction)))
    validation_indices = permutation[:validation_size].tolist()
    train_indices = permutation[validation_size:].tolist()
    train = _subset_samples(Subset(training, train_indices), config.train_size_limit)
    validation = _subset_samples(Subset(training, validation_indices), config.validation_size_limit)
    if test is not None:
        test = _subset_samples(test, config.test_size_limit)

    metadata = DatasetMetadata(
        config.dataset, config.emnist_split if config.dataset == "emnist" else None,
        dims, classes, len(train), len(validation), len(test) if test is not None else None,
        orientation, offset, mean, std,
    )
    return DatasetBundle(train, validation, test, metadata)


def create_model(config: ExperimentConfig, dataset_metadata: DatasetMetadata) -> nn.Module:
    """Build an FP32 classifier from the dataset contract and return raw logits."""
    if config.model != "mlp":
        raise ValueError(f"Unknown model {config.model!r}")
    input_size = math.prod(config.input_dims or dataset_metadata.input_dims)
    if input_size != math.prod(dataset_metadata.input_dims):
        raise ValueError("Configured input dimensions do not match the dataset")
    activation = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[config.activation]
    sizes = (input_size, *config.hidden_dims, dataset_metadata.num_classes)
    layers: list[nn.Module] = [nn.Flatten()]
    for start, end in zip(sizes[:-2], sizes[1:-1]):
        layers.extend((nn.Linear(start, end, dtype=torch.float32), activation()))
    layers.append(nn.Linear(sizes[-2], sizes[-1], dtype=torch.float32))
    return nn.Sequential(*layers)


def _loader(
    dataset: Dataset, config: ExperimentConfig, rank: int, batch_size: int, device: torch.device,
    **kwargs: Any,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=config.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(_rank_seed(config.seed, rank)),
        multiprocessing_context="spawn" if config.num_workers else None, **kwargs,
    )


def create_train_loader(
    dataset: Dataset, config: ExperimentConfig, rank: int, world_size: int, device: torch.device,
) -> tuple[DataLoader, DistributedSampler]:
    """Shard a common epoch permutation and discard incomplete rank/global batches."""
    batch_size = config.global_batch_size // world_size if config.global_batch_size is not None else config.local_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=_rank_seed(config.seed, 0), drop_last=True
    )
    return _loader(dataset, config, rank, batch_size, device, sampler=sampler, drop_last=True), sampler


def create_eval_loader(
    dataset: Dataset, config: ExperimentConfig, rank: int, world_size: int, device: torch.device,
) -> DataLoader:
    """Evaluate each sample exactly once even when shard sizes differ."""
    batch_size = config.global_batch_size // world_size if config.global_batch_size is not None else config.local_batch_size
    return _loader(Subset(dataset, range(rank, len(dataset), world_size)), config, rank, batch_size, device)


def broadcast_model(model: nn.Module, world_size: int) -> None:
    """Copy rank-zero parameters and buffers to each replica before training."""
    if world_size > 1:
        with torch.no_grad():
            for tensor in (*model.parameters(), *model.buffers()):
                dist.broadcast(tensor, src=0)


def _rank_seed(seed: int, rank: int) -> int:
    return (seed + rank) % (2**63)



# Compression
# Quantization chooses signed levels. The codecs only decide how to represent
# those levels on the wire, so changing codecs leaves the stochastic operator
# and the reconstructed gradient unchanged.

def _measure(timer: Any, name: str) -> Any:
    return nullcontext() if timer is None else timer.measure(name)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class CompressedGradient:
    """A device-resident packet and its shared layout.

    Payload and scales are transmitted. Shape, dtype, level count, and bucket
    size are known by all ranks from their shared configuration and layout.
    """

    payload: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    scales: torch.Tensor | None = None
    levels: int | None = None
    bits: int | None = None
    bucket_size: int | None = None
    encoding: str = "identity"
    variable_size: bool = False

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def original_bytes(self) -> int:
        return self.num_elements * torch.finfo(self.dtype).bits // 8

    @property
    def payload_bytes(self) -> int:
        return _tensor_bytes(self.payload)

    @property
    def metadata_bytes(self) -> int:
        return 0 if self.scales is None else _tensor_bytes(self.scales)


class GradientCompressor(Protocol):
    def compress(self, gradient: torch.Tensor, timer: Any = None) -> CompressedGradient:
        """Encode a floating-point gradient tensor without mutating it."""

    def decompress(
        self, packet: CompressedGradient, timer: Any = None
    ) -> torch.Tensor:
        """Reconstruct the gradient with its original shape, dtype, and device."""


class IdentityCompressor:
    def compress(self, gradient: torch.Tensor, timer: Any = None) -> CompressedGradient:
        return CompressedGradient(
            gradient.detach().reshape(-1).contiguous(), tuple(gradient.shape), gradient.dtype
        )

    def decompress(self, packet: CompressedGradient, timer: Any = None) -> torch.Tensor:
        return packet.payload.reshape(packet.shape)


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack in-range signed indices into uint8 using little-endian bit slots.

    The input must fit in signed 2's-complement integers of the given width.
    The unused slots in a partial final byte are zero.
    """
    if bits not in (2, 4, 8):
        raise ValueError("Packed indices require a width of 2, 4, or 8 bits")
    slots_per_byte = 8 // bits
    mask = (1 << bits) - 1
    codes = indices.reshape(-1).to(torch.int16) & mask
    padding = -codes.numel() % slots_per_byte
    if padding:
        codes = F.pad(codes, (0, padding))
    shifts = torch.arange(slots_per_byte, device=indices.device, dtype=torch.int16) * bits
    return (codes.reshape(-1, slots_per_byte) << shifts).sum(dim=1, dtype=torch.int16).to(torch.uint8)


def unpack_indices(payload: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Decode a packed uint8 buffer into a flat, signed int32 tensor."""
    if bits not in (2, 4, 8):
        raise ValueError("Packed indices require a width of 2, 4, or 8 bits")
    if num_elements < 0 or payload.numel() != (num_elements * bits + 7) // 8:
        raise ValueError("Encoded payload length does not match the gradient length")
    slots_per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    shifts = torch.arange(slots_per_byte, device=payload.device, dtype=torch.int16) * bits
    codes = ((payload.reshape(-1, 1).to(torch.int16) >> shifts) & mask).reshape(-1)
    return ((codes[:num_elements] ^ sign_bit) - sign_bit).to(torch.int32)


class _EliasWriter:
    """Byte-oriented bit writer for the CPU reference codec."""

    def __init__(self) -> None:
        self.bytes = bytearray()
        self.tail = 0
        self.tail_bits = 0

    def write(self, value: int, width: int) -> None:
        self.tail = (self.tail << width) | value
        self.tail_bits += width
        while self.tail_bits >= 8:
            self.tail_bits -= 8
            self.bytes.append(self.tail >> self.tail_bits)
            self.tail &= (1 << self.tail_bits) - 1

    def omega(self, value: int) -> None:
        if value < 1:
            raise ValueError("Elias omega encodes only positive integers")
        prefixes = []
        while value > 1:
            width = value.bit_length()
            prefixes.append((value, width))
            value = width - 1
        for prefix, width in reversed(prefixes):
            self.write(prefix, width)
        self.write(0, 1)

    def finish(self) -> bytearray:
        if self.tail_bits:
            self.bytes.append(self.tail << (8 - self.tail_bits))
        return self.bytes


class _EliasReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read(self, width: int) -> int:
        end = self.position + width
        if end > len(self.data) * 8:
            raise ValueError("Truncated Elias payload")
        first_byte = self.position // 8
        end_byte = (end + 7) // 8
        value = int.from_bytes(self.data[first_byte:end_byte], "big")
        self.position = end
        return (value >> (-end % 8)) & ((1 << width) - 1)

    def omega(self) -> int:
        value = 1
        while self.read(1):
            value = (1 << value) | self.read(value)
        return value


def encode_elias_indices(indices: torch.Tensor) -> torch.Tensor:
    """Encode CPU integer indices as omega-coded gaps, signs, and magnitudes.

    A final gap to the known array length terminates the stream. The scales
    travel separately so one sparse stream can use several quantization buckets.
    """
    if indices.device.type != "cpu":
        raise ValueError("The reference Elias encoder expects CPU indices")
    flat = indices.reshape(-1)
    positions = flat.nonzero().flatten()
    values = flat[positions]
    writer = _EliasWriter()
    previous = -1
    for position, value in zip(positions.tolist(), values.tolist()):
        writer.omega(position - previous)
        writer.write(int(value < 0), 1)
        writer.omega(abs(value))
        previous = position
    writer.omega(flat.numel() - previous)
    return torch.frombuffer(writer.finish(), dtype=torch.uint8)


def decode_elias_indices(payload: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Decode the CPU reference stream to flat signed int32 indices."""
    if payload.device.type != "cpu" or payload.dtype != torch.uint8:
        raise ValueError("The reference Elias decoder expects a CPU uint8 payload")
    if num_elements < 0:
        raise ValueError("The decoded element count must be nonnegative")
    reader = _EliasReader(payload.contiguous().numpy().tobytes())
    positions, values = [], []
    position = -1
    while True:
        position += reader.omega()
        if position == num_elements:
            break
        if position > num_elements:
            raise ValueError("Invalid position in Elias payload")
        negative = reader.read(1)
        magnitude = reader.omega()
        if magnitude >= 1 << 31:
            raise ValueError("Elias magnitude does not fit a signed int32 index")
        positions.append(position)
        values.append(-magnitude if negative else magnitude)
    decoded = torch.zeros(num_elements, dtype=torch.int32)
    if positions:
        decoded[torch.tensor(positions, dtype=torch.long)] = torch.tensor(values, dtype=torch.int32)
    return decoded


class QSGDCompressor:
    """Bucketwise QSGD with a separate reproducible RNG on each input device."""

    def __init__(
        self,
        levels: int = 7,
        bits: int = 4,
        normalization: str = "l2",
        bucket_size: int | None = None,
        encoding: str = "packed",
        seed: int = 0,
    ) -> None:
        if bits not in (2, 4, 8) or not 1 <= levels <= (1 << (bits - 1)) - 1:
            raise ValueError("QSGD levels must fit the selected signed 2-, 4-, or 8-bit width")
        if normalization not in ("l2", "max"):
            raise ValueError("QSGD normalization must be l2 or max")
        if bucket_size is not None and bucket_size < 1:
            raise ValueError("Quantization buckets must have positive size")
        if encoding not in ("packed", "integer", "elias"):
            raise ValueError("Unknown QSGD encoding")
        self.levels = levels
        self.bits = bits
        self.normalization = normalization
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.seed = seed
        self.generators: dict[torch.device, torch.Generator] = {}

    @torch.no_grad()
    def quantize(self, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flat signed int32 levels and one floating-point scale per bucket."""
        if not gradient.is_floating_point():
            raise TypeError("QSGD requires floating-point gradients")
        count = gradient.numel()
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        if count == 0:
            return gradient.new_empty(0, dtype=torch.int32), gradient.new_empty(0, dtype=dtype)

        size = min(self.bucket_size or count, count)
        flat = gradient.reshape(-1).to(dtype)
        padding = -count % size
        if padding:
            flat = F.pad(flat, (0, padding))
        buckets = flat.view(-1, size)
        magnitudes = buckets.abs()
        max_abs = magnitudes.amax(dim=1, keepdim=True)
        safe_max = torch.where(max_abs > 0, max_abs, 1.0)
        relative = magnitudes / safe_max
        scales = max_abs
        if self.normalization == "l2":
            # Scaling before the sum of squares handles small and large FP32 values.
            relative_norm = torch.linalg.vector_norm(relative, dim=1, keepdim=True)
            relative = relative / torch.where(relative_norm > 0, relative_norm, 1.0)
            scales = max_abs * relative_norm

        grid = relative.mul_(self.levels).clamp_(0, self.levels)
        lower = grid.floor()
        generator = self.generators.get(gradient.device)
        if generator is None:
            generator = torch.Generator(device=gradient.device).manual_seed(self.seed)
            self.generators[gradient.device] = generator
        draw = torch.rand(grid.shape, generator=generator, device=grid.device, dtype=grid.dtype)
        rounded = lower + (draw < grid - lower)
        signed = (rounded * buckets.sign()).to(torch.int32)
        return signed.reshape(-1)[:count], scales.reshape(-1)

    def compress(self, gradient: torch.Tensor, timer: Any = None) -> CompressedGradient:
        indices, scales = self.quantize(gradient)
        if self.encoding == "elias":
            with _measure(timer, "elias_host_transfer"):
                host_indices = indices.cpu()
            with _measure(timer, "elias_encode_cpu"):
                host_payload = encode_elias_indices(host_indices)
            with _measure(timer, "elias_host_transfer"):
                payload = host_payload.to(gradient.device)
        elif self.encoding == "integer":
            payload = indices.to(torch.int8)
        else:
            payload = pack_indices(indices, self.bits)
        return CompressedGradient(
            payload=payload,
            shape=tuple(gradient.shape),
            dtype=gradient.dtype,
            scales=scales,
            levels=self.levels,
            bits=self.bits,
            bucket_size=self.bucket_size,
            encoding=self.encoding,
            variable_size=self.encoding == "elias",
        )

    def decompress(self, packet: CompressedGradient, timer: Any = None) -> torch.Tensor:
        if packet.scales is None or packet.levels is None or packet.bits is None:
            raise ValueError("QSGD packet is missing quantization metadata")
        count = packet.num_elements
        if packet.encoding == "elias":
            with _measure(timer, "elias_host_transfer"):
                host_payload = packet.payload.cpu()
            with _measure(timer, "elias_decode_cpu"):
                host_indices = decode_elias_indices(host_payload, count)
            with _measure(timer, "elias_host_transfer"):
                indices = host_indices.to(packet.payload.device)
        elif packet.encoding == "integer":
            indices = packet.payload
        elif packet.encoding == "packed":
            indices = unpack_indices(packet.payload, packet.bits, count)
        else:
            raise ValueError(f"Unknown QSGD packet encoding {packet.encoding!r}")

        if count == 0:
            return indices.to(packet.dtype).reshape(packet.shape)
        size = min(packet.bucket_size or count, count)
        padding = -count % size
        values = indices.to(packet.scales.dtype).div_(packet.levels)
        if padding:
            values = F.pad(values, (0, padding))
        values.view(-1, size).mul_(packet.scales[:, None])
        return values[:count].to(packet.dtype).reshape(packet.shape)


def create_compressor(config: ExperimentConfig, rank: int = 0) -> GradientCompressor:
    if config.compression == "identity":
        return IdentityCompressor()
    if config.compression != "qsgd":
        raise ValueError(f"Unknown compressor {config.compression!r}")
    levels = config.quantization_levels
    if levels is None:
        levels = (1 << (config.quantization_bits - 1)) - 1
    # A rank-local generator leaves the default torch RNG and sampler untouched.
    seed = (config.compression_seed + rank * 0x9E3779B97F4A7C15) % (1 << 63)
    return QSGDCompressor(
        levels, config.quantization_bits, config.normalization,
        config.quantization_bucket_size, config.encoding, seed,
    )


# Synchronization statistics
# Totals retain additive sufficient statistics; ratios and means are calculated
# only after aggregation so buckets and epochs have the appropriate weighting.

@dataclass
class CommunicationStats:
    rounds: int = 0
    collective_calls: int = 0
    diagnostic_collective_calls: int = 0
    communicated_tensors: int = 0
    num_elements: int = 0
    original_bytes: int = 0
    payload_bytes: int = 0
    normalization_bytes: int = 0
    metadata_bytes: int = 0
    padding_bytes: int = 0
    logical_bytes: int = 0
    estimated_aggregate_bytes: int = 0

    @property
    def encoded_total_bytes(self) -> int:
        return self.payload_bytes + self.normalization_bytes + self.metadata_bytes + self.padding_bytes

    @property
    def compression_ratio(self) -> float | None:
        size = self.encoded_total_bytes
        return self.original_bytes / size if size else None

    @property
    def effective_bits_per_element(self) -> float | None:
        return 8 * self.encoded_total_bytes / self.num_elements if self.num_elements else None

    def add(self, other: "CommunicationStats") -> None:
        for name, amount in vars(other).items():
            setattr(self, name, getattr(self, name) + amount)

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            **vars(self),
            "encoded_total_bytes": self.encoded_total_bytes,
            "compression_ratio": self.compression_ratio,
            "effective_bits_per_element": self.effective_bits_per_element,
            "physical_interconnect_bytes": None,
        }


@dataclass
class ErrorStats:
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    elements: int = 0
    max_error: float | None = None

    def add(self, other: "ErrorStats") -> None:
        self.squared_error_sum += other.squared_error_sum
        self.absolute_error_sum += other.absolute_error_sum
        self.elements += other.elements
        if other.max_error is not None:
            self.max_error = max(self.max_error or 0.0, other.max_error)

    def to_dict(self) -> dict[str, int | float | None]:
        mse = self.squared_error_sum / self.elements if self.elements else None
        return {
            **vars(self),
            "mse": mse,
            "rmse": math.sqrt(mse) if mse is not None else None,
            "mae": self.absolute_error_sum / self.elements if self.elements else None,
        }


class _DeviceErrors:
    """Keep sums, maxima and counts on device until the end of a measured step."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.rows: dict[str, tuple[torch.Tensor, int]] = {}

    def add(self, key: str, elements: int, error: torch.Tensor | None) -> None:
        if error is None:
            stats = torch.zeros(3, device=self.device, dtype=torch.float64)
        else:
            abs_error = error.abs().to(torch.float64)
            stats = torch.stack((abs_error.square().sum(), abs_error.sum(), abs_error.amax()))
        previous = self.rows.get(key)
        if previous is not None:
            previous_stats, previous_elements = previous
            stats[:2] += previous_stats[:2]
            stats[2] = torch.maximum(stats[2], previous_stats[2])
            elements += previous_elements
        self.rows[key] = stats, elements

    def finish(
        self, world_size: int, measurement: str, communication: CommunicationStats,
        timer: Any,
    ) -> dict[str, ErrorStats]:
        if not self.rows:
            return {}
        rows = list(self.rows.items())
        with _measure(timer, "diagnostic_error"):
            packed = torch.stack([row[0] for _, row in rows])
        if measurement == "local" and world_size > 1:
            with _measure(timer, "diagnostic_collective"):
                sums, maxima = packed[:, :2].contiguous(), packed[:, 2].contiguous()
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                communication.diagnostic_collective_calls += 2
                packed = torch.cat((sums, maxima[:, None]), dim=1)
        with _measure(timer, "diagnostic_error"):
            host = packed.tolist()
        rank_count = world_size if measurement == "local" else 1
        return {
            key: ErrorStats(squared, absolute, elements * rank_count, maximum)
            for (key, (_, elements)), (squared, absolute, maximum) in zip(rows, host)
        }


@dataclass
class SynchronizationResult:
    communication: CommunicationStats = field(default_factory=CommunicationStats)
    errors: dict[str, ErrorStats] = field(default_factory=dict)


@dataclass
class _CommunicationBucket:
    start: int
    end: int
    use_compressor: bool
    sections: list[tuple[str, int, int]] = field(default_factory=list)


# Gradient transport
# A static layout defines tensor collectives in the same order on every rank.
# Collectives transport packets independently of how the compressor made them.

class GradientSynchronizer:
    """Synchronize dense FP32 gradients for a replica with a static parameter layout."""

    def __init__(
        self,
        config: ExperimentConfig,
        parameters: Iterable[tuple[str, nn.Parameter]],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.parameters = [
            (name, parameter) for name, parameter in parameters
            if parameter.requires_grad and parameter.numel()
        ]
        if not self.parameters:
            raise ValueError("Gradient synchronization requires at least one trainable element")
        self.compressor = create_compressor(config, rank)
        self.buckets = self._build_buckets()

    def _build_buckets(self) -> list[_CommunicationBucket]:
        max_elements = max(1, self.config.communication_bucket_bytes // 4)
        buckets: list[_CommunicationBucket] = []
        offset = 0
        for name, parameter in self.parameters:
            count = parameter.numel()
            use_compressor = (
                self.config.synchronization == "compressed_allgather"
                and count >= self.config.min_compression_elements
            )
            while count:
                if (
                    not buckets
                    or buckets[-1].use_compressor != use_compressor
                    or buckets[-1].end - buckets[-1].start == max_elements
                ):
                    buckets.append(_CommunicationBucket(offset, offset, use_compressor))
                bucket = buckets[-1]
                length = min(count, max_elements - (bucket.end - bucket.start))
                bucket.sections.append((name, offset - bucket.start, length))
                bucket.end += length
                offset += length
                count -= length
        return buckets

    def _all_gather(
        self, local: torch.Tensor, stats: CommunicationStats, timer: Any,
    ) -> list[torch.Tensor]:
        if self.world_size == 1:
            return [local]
        with _measure(timer, "collective"):
            peers = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(peers, local)
        size = _tensor_bytes(local)
        stats.collective_calls += 1
        stats.communicated_tensors += 1
        stats.logical_bytes += self.world_size * size
        stats.estimated_aggregate_bytes += self.world_size * (self.world_size - 1) * size
        return peers

    def _exchange(
        self, packet: CompressedGradient, stats: CommunicationStats, timer: Any,
    ) -> Iterable[CompressedGradient]:
        if self.world_size == 1:
            return [packet]
        payload = packet.payload
        sizes = None
        if packet.variable_size:
            with _measure(timer, "elias_host_transfer"):
                size = torch.tensor([payload.numel()], device=payload.device, dtype=torch.int64)
            peer_sizes = self._all_gather(size, stats, timer)
            stats.metadata_bytes += _tensor_bytes(size)
            with _measure(timer, "elias_host_transfer"):
                sizes = torch.cat(peer_sizes).tolist()
            padding = max(sizes) - payload.numel()
            if padding:
                with _measure(timer, "compression"):
                    payload = F.pad(payload, (0, padding))
                stats.padding_bytes += padding * payload.element_size()
        peers = self._all_gather(payload, stats, timer)
        scales = (
            self._all_gather(packet.scales, stats, timer)
            if packet.scales is not None
            else [None] * self.world_size
        )
        if sizes is not None:
            peers = [peer[:size] for peer, size in zip(peers, sizes)]
        return [
            replace(packet, payload=peer, scales=scale)
            for peer, scale in zip(peers, scales)
        ]

    def _all_reduce(
        self, buffer: torch.Tensor, stats: CommunicationStats, timer: Any,
    ) -> torch.Tensor:
        if self.world_size > 1:
            with _measure(timer, "collective"):
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            with _measure(timer, "aggregation"):
                buffer.div_(self.world_size)
            size = _tensor_bytes(buffer)
            stats.collective_calls += 1
            stats.communicated_tensors += 1
            stats.logical_bytes += size
            stats.estimated_aggregate_bytes += 2 * (self.world_size - 1) * size
        return buffer

    @staticmethod
    def _record_bucket_error(
        ledger: _DeviceErrors,
        measurement: str,
        bucket: _CommunicationBucket,
        difference: torch.Tensor | None,
        per_layer: bool,
    ) -> None:
        ledger.add(measurement, bucket.end - bucket.start, difference)
        if per_layer:
            for name, start, length in bucket.sections:
                section_error = None if difference is None else difference[start:start + length]
                ledger.add(f"{measurement}:{name}", length, section_error)

    def _prepare(self) -> torch.Tensor:
        pieces = []
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"Missing gradient for trainable parameter {name!r}")
            if gradient.dtype != torch.float32 or gradient.layout != torch.strided:
                raise TypeError("Gradient synchronization requires dense FP32 gradients")
            pieces.append(gradient.reshape(-1))
        return torch.cat(pieces)

    def _aggregate_packets(
        self,
        peers: Iterable[CompressedGradient],
        timer: Any,
        retain_local: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        total = None
        local_reconstruction = None
        for peer_rank, peer in enumerate(peers):
            with _measure(timer, "decompression"):
                reconstructed = self.compressor.decompress(peer, timer)
            if retain_local and peer_rank == self.rank:
                local_reconstruction = reconstructed
            with _measure(timer, "aggregation"):
                if total is None:
                    # Decoded buffers can alias payloads or the local gradient.
                    total = reconstructed.clone()
                else:
                    total.add_(reconstructed)
        if total is None:
            raise RuntimeError("Expected at least the local gradient packet")
        with _measure(timer, "aggregation"):
            total.div_(self.world_size)
        return total, local_reconstruction

    @torch.no_grad()
    def synchronize(self, step: int = 1, timer: Any = None) -> SynchronizationResult:
        """Replace local gradients with their rank average; step is one-based."""
        result = SynchronizationResult()
        stats = result.communication
        stats.rounds = 1
        config = self.config
        measure_errors = (
            config.mode == "diagnostic"
            and config.gradient_error != "none"
            and step % config.gradient_error_frequency == 0
        )
        measure_local = measure_errors and config.gradient_error in ("local", "both")
        measure_aggregate = measure_errors and config.gradient_error in ("aggregated", "both")
        with _measure(timer, "synchronize"):
            with _measure(timer, "prepare_gradients"):
                flat = self._prepare()
            local_errors = _DeviceErrors(flat.device) if measure_local else None
            aggregate_errors = _DeviceErrors(flat.device) if measure_aggregate else None

            for bucket in self.buckets:
                local = flat[bucket.start:bucket.end]
                size = _tensor_bytes(local)
                stats.num_elements += local.numel()
                stats.original_bytes += size
                local_reconstruction = None
                lossy = False
                if bucket.use_compressor:
                    with _measure(timer, "compression"):
                        packet = self.compressor.compress(local, timer)
                    stats.payload_bytes += packet.payload_bytes
                    stats.normalization_bytes += packet.metadata_bytes
                    lossy = packet.encoding != "identity"
                    peers = self._exchange(packet, stats, timer)
                    average, local_reconstruction = self._aggregate_packets(
                        peers, timer, measure_local and lossy,
                    )
                elif config.synchronization == "fp32_allreduce":
                    stats.payload_bytes += size
                    average = self._all_reduce(local, stats, timer)
                else:
                    stats.payload_bytes += size
                    peers = self._all_gather(local, stats, timer)
                    with _measure(timer, "aggregation"):
                        average = peers[0]
                        if self.world_size > 1:
                            for peer in peers[1:]:
                                average.add_(peer)
                            average.div_(self.world_size)

                if local_errors is not None:
                    with _measure(timer, "diagnostic_error"):
                        difference = None if local_reconstruction is None else local_reconstruction - local
                        self._record_bucket_error(
                            local_errors, "local", bucket, difference, config.per_layer_error,
                        )
                if aggregate_errors is not None:
                    difference = None
                    if lossy:
                        reference = local
                        if self.world_size > 1:
                            with _measure(timer, "diagnostic_collective"):
                                reference = local.clone()
                                dist.all_reduce(reference, op=dist.ReduceOp.SUM)
                                reference.div_(self.world_size)
                            stats.diagnostic_collective_calls += 1
                        with _measure(timer, "diagnostic_error"):
                            difference = average - reference
                    with _measure(timer, "diagnostic_error"):
                        self._record_bucket_error(
                            aggregate_errors, "aggregated", bucket, difference, config.per_layer_error,
                        )
                if average is not local:
                    with _measure(timer, "restore_gradients"):
                        local.copy_(average)

            with _measure(timer, "restore_gradients"):
                offset = 0
                for _, parameter in self.parameters:
                    end = offset + parameter.numel()
                    parameter.grad.copy_(flat[offset:end].view_as(parameter))
                    offset = end
        if local_errors is not None:
            result.errors.update(local_errors.finish(self.world_size, "local", stats, timer))
        if aggregate_errors is not None:
            result.errors.update(aggregate_errors.finish(self.world_size, "aggregated", stats, timer))
        return result


# Performance and convergence records depend on sample totals and completed
# operations. The training loop keeps CUDA measurements on-device until an epoch
# boundary, where it already waits to evaluate and report global metrics.
TIMING_PHASES = (
    "data_loading", "host_to_device", "forward", "backward", "optimizer", "synchronize",
    "prepare_gradients", "compression", "collective", "decompression", "aggregation",
    "restore_gradients", "diagnostic_collective", "diagnostic_error", "elias_host_transfer",
    "elias_encode_cpu", "elias_decode_cpu",
)
_HOST_TIMING_PHASES = {"data_loading", "elias_host_transfer", "elias_encode_cpu", "elias_decode_cpu"}


class PhaseTimer:
    """Accumulate synchronous host intervals or completed intervals on the CUDA stream."""

    def __init__(self, device: torch.device, detailed: bool) -> None:
        self.device = device
        self.detailed = detailed
        self.host_seconds: dict[str, float] = defaultdict(float)
        self.device_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def measure(self, name: str):
        if (
            self.detailed or name.startswith("elias_")
            or (self.device.type == "cpu" and name == "synchronize")
        ):
            return self._measure(name)
        return nullcontext()

    @contextmanager
    def _measure(self, name: str) -> Iterator[None]:
        if self.device.type == "cpu" or name in _HOST_TIMING_PHASES:
            start = time.perf_counter()
            yield
            self.host_seconds[name] += time.perf_counter() - start
        else:
            start = _cuda_timestamp()
            yield
            self.device_events[name].append((start, _cuda_timestamp()))

    def to_dict(self) -> dict[str, float]:
        seconds = dict(self.host_seconds)
        for name, events in self.device_events.items():
            seconds[name] = sum(start.elapsed_time(end) for start, end in events) / 1000
        return seconds


def _cuda_timestamp() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _finish_device_work(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@dataclass
class ClassificationStats:
    loss_sum: float = 0.0
    correct: int = 0
    samples: int = 0

    def add(self, other: ClassificationStats) -> None:
        self.loss_sum += other.loss_sum
        self.correct += other.correct
        self.samples += other.samples

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "loss": self.loss_sum / self.samples if self.samples else None,
            "accuracy": self.correct / self.samples if self.samples else None,
            "samples": self.samples,
        }


def _batch_totals(logits: torch.Tensor, targets: torch.Tensor, loss_sum: torch.Tensor) -> torch.Tensor:
    """Keep loss sum, correct count, and sample count together on the model device."""
    return torch.stack((
        loss_sum.detach().to(torch.float64),
        (logits.detach().argmax(dim=1) == targets).sum(),
        targets.new_full((), targets.numel()),
    ))


def _classification_row(row: list[float]) -> dict:
    loss_sum, correct, samples = row
    return ClassificationStats(loss_sum, int(correct), int(samples)).to_dict()


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, world_size: int,
) -> dict:
    """Globally aggregate totals from disjoint evaluation shards without padding."""
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        totals += _batch_totals(logits, targets, criterion(logits, targets))
    if world_size > 1:
        dist.all_reduce(totals)
    return _classification_row(totals.tolist())


def compare_runtime(baseline_runtime: float | None, runtime: float | None) -> dict[str, float | None]:
    """Return paired speedup and percentage reduction, preserving unavailable values."""
    if (
        baseline_runtime is None or runtime is None
        or not math.isfinite(baseline_runtime) or not math.isfinite(runtime)
        or baseline_runtime <= 0 or runtime <= 0
    ):
        return {"speedup": None, "runtime_reduction_percent": None}
    return {
        "speedup": baseline_runtime / runtime,
        "runtime_reduction_percent": 100 * (baseline_runtime - runtime) / baseline_runtime,
    }


@dataclass
class ConvergenceTracker:
    config: ExperimentConfig
    first_hit: dict = field(default_factory=dict)

    def observe(self, epoch_metrics: dict, communication: CommunicationStats) -> bool:
        """Record the first completed validation that satisfies every configured target."""
        loss, accuracy = self.config.target_validation_loss, self.config.target_validation_accuracy
        observed_loss = epoch_metrics.get("validation_loss")
        observed_accuracy = epoch_metrics.get("validation_accuracy")
        reached = (
            (loss is not None or accuracy is not None)
            and (loss is None or (observed_loss is not None and observed_loss <= loss))
            and (accuracy is None or (observed_accuracy is not None and observed_accuracy >= accuracy))
        )
        if reached and not self.first_hit:
            self.first_hit = {
                "time_to_target": epoch_metrics.get("cumulative_runtime", epoch_metrics.get("cumulative_training_time")),
                "epoch_to_target": epoch_metrics["epoch"],
                "step_to_target": epoch_metrics["step"],
                "global_batches_to_target": epoch_metrics["step"],
                "samples_to_target": epoch_metrics.get("samples_processed"),
                "synchronization_rounds_to_target": communication.rounds,
                "collective_calls_to_target": communication.collective_calls,
                "communication_bytes_to_target": communication.estimated_aggregate_bytes,
            }
        return reached

    def to_dict(self) -> dict:
        configured = self.config.target_validation_loss is not None or self.config.target_validation_accuracy is not None
        return {
            "target_validation_loss": self.config.target_validation_loss,
            "target_validation_accuracy": self.config.target_validation_accuracy,
            "target_reached": bool(self.first_hit) if configured else None,
            **dict.fromkeys((
                "time_to_target", "epoch_to_target", "step_to_target", "global_batches_to_target",
                "samples_to_target", "synchronization_rounds_to_target", "collective_calls_to_target",
                "communication_bytes_to_target",
            )),
            **self.first_hit,
        }


def _create_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    settings = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=config.momentum, **settings)
    return torch.optim.AdamW(model.parameters(), **settings)


def _warm_up(
    config: ExperimentConfig, model: nn.Module, rank: int, plan: ExecutionPlan, device: torch.device,
    metadata: DatasetMetadata,
) -> None:
    """Exercise the same compute and collective shapes on a separate model and RNG."""
    if not config.warmup_steps:
        return
    warm_model = copy.deepcopy(model)
    optimizer = _create_optimizer(config, warm_model)
    synchronizer = GradientSynchronizer(
        replace(config, gradient_error="none", per_layer_error=False),
        warm_model.named_parameters(), rank, plan.world_size,
    )
    generator = torch.Generator().manual_seed(_rank_seed(config.seed, rank))
    inputs = torch.randn(
        plan.local_batch_size, *metadata.input_dims, generator=generator, dtype=torch.float32
    ).to(device)
    targets = torch.randint(metadata.num_classes, (plan.local_batch_size,), generator=generator).to(device)
    criterion = nn.CrossEntropyLoss()
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for step in range(1, config.warmup_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            criterion(warm_model(inputs), targets).backward()
            synchronizer.synchronize(step)
            optimizer.step()
    _finish_device_work(device)


def _gather_timing_rows(local: list[list[float]], device: torch.device, world_size: int) -> list:
    if world_size == 1:
        return [local]
    packed = torch.tensor(local, device=device, dtype=torch.float64)
    gathered = [torch.empty_like(packed) for _ in range(world_size)]
    dist.all_gather(gathered, packed)
    return torch.stack(gathered).cpu().tolist()


def _sum_timing(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for timing in rows:
        for name, value in timing.items():
            total[name] += value
    return dict(total)


def _encode_timing(timing: dict[str, float]) -> list[float]:
    return [timing.get(key, -1.0) for key in TIMING_PHASES]


def _decode_timing(values: list[float]) -> dict[str, float]:
    return {key: value for key, value in zip(TIMING_PHASES, values) if value >= 0}


def _merge_errors(total: dict[str, ErrorStats], more: dict[str, ErrorStats]) -> None:
    for name, stats in more.items():
        total.setdefault(name, ErrorStats()).add(stats)


def _serialize_errors(errors: dict[str, ErrorStats]) -> dict:
    return {name: stats.to_dict() for name, stats in errors.items()}


@dataclass
class RecordedStep:
    totals: torch.Tensor
    timer: PhaseTimer
    sync: SynchronizationResult
    elapsed: float | tuple[torch.cuda.Event, torch.cuda.Event]
    epoch_elapsed: float | torch.cuda.Event

    def timing_row(
        self, epoch_start_event: torch.cuda.Event | None, epoch_offset: float, training_offset: float,
    ) -> list[float]:
        if epoch_start_event is None:
            elapsed, end = self.elapsed, self.epoch_elapsed
        else:
            first, last = self.elapsed
            elapsed = first.elapsed_time(last) / 1000
            end = epoch_start_event.elapsed_time(self.epoch_elapsed) / 1000
        return [elapsed, epoch_offset + end, training_offset + end, *_encode_timing(self.timer.to_dict())]


# The train loop deals in batches and structured synchronization results. Model
# layouts, gradient transport and encoded payloads are handled by the synchronizer.
def _train(
    config: ExperimentConfig, model: nn.Module, datasets: DatasetBundle, device: torch.device,
    rank: int, plan: ExecutionPlan,
) -> dict:
    train_loader, sampler = create_train_loader(datasets.train, config, rank, plan.world_size, device)
    validation_loader = create_eval_loader(datasets.validation, config, rank, plan.world_size, device)
    if not len(train_loader):
        raise ValueError(
            f"Training needs at least one full global batch ({plan.global_batch_size} samples). "
            "Reduce the batch size or increase the training sample limit."
        )

    optimizer = _create_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine" else None
    )
    synchronizer = GradientSynchronizer(config, model.named_parameters(), rank, plan.world_size)
    tracker = ConvergenceTracker(config)
    epoch_metrics, step_metrics = [], []
    total_communication = CommunicationStats()
    total_errors: dict[str, ErrorStats] = {}
    rank_timings: list[list[dict]] = [[] for _ in range(plan.world_size)]
    cumulative_training = 0.0
    samples_processed, steps, full_epochs = 0, 0, 0
    best_validation_loss, epochs_without_improvement = math.inf, 0
    stop_reason = "epoch_budget"

    warmup_start = time.perf_counter()
    _warm_up(config, model, rank, plan, device, datasets.metadata)
    warmup_runtime = time.perf_counter() - warmup_start if config.warmup_steps else None
    if plan.world_size > 1:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    training_start = time.perf_counter()
    uses_step_events = config.step_metrics and device.type == "cuda"
    for epoch in range(1, config.epochs + 1):
        model.train()
        sampler.set_epoch(epoch - 1)
        epoch_start = time.perf_counter()
        epoch_start_event = _cuda_timestamp() if uses_step_events else None
        epoch_totals = torch.zeros(3, dtype=torch.float64, device=device)
        epoch_communication = CommunicationStats()
        epoch_errors: dict[str, ErrorStats] = {}
        epoch_timer = PhaseTimer(device, config.detailed_timing)
        recorded: list[RecordedStep] = []
        steps_this_epoch = (
            min(len(train_loader), config.max_steps - steps) if config.max_steps is not None else len(train_loader)
        )
        with epoch_timer.measure("data_loading"):
            batches = iter(train_loader)
        learning_rate = optimizer.param_groups[0]["lr"]
        for _ in range(steps_this_epoch):
            start_step = _cuda_timestamp() if uses_step_events else time.perf_counter() if config.step_metrics else None
            timer = PhaseTimer(device, config.detailed_timing) if config.step_metrics else epoch_timer
            with timer.measure("data_loading"):
                inputs, targets = next(batches)
            with timer.measure("host_to_device"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            with timer.measure("optimizer"):
                optimizer.zero_grad(set_to_none=True)
            with timer.measure("forward"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            with timer.measure("backward"):
                loss.backward()
            steps += 1
            sync = synchronizer.synchronize(steps, timer)
            with timer.measure("optimizer"):
                optimizer.step()
            totals = _batch_totals(logits, targets, loss * targets.numel())
            if not config.step_metrics:
                epoch_totals += totals
            epoch_communication.add(sync.communication)
            _merge_errors(epoch_errors, sync.errors)
            if config.step_metrics:
                end = _cuda_timestamp() if uses_step_events else time.perf_counter()
                elapsed = (start_step, end) if uses_step_events else end - start_step
                epoch_elapsed = end if uses_step_events else end - epoch_start
                recorded.append(RecordedStep(totals, timer, sync, elapsed, epoch_elapsed))

        _finish_device_work(device)
        training_runtime = time.perf_counter() - epoch_start
        # A single epoch-boundary reduction handles optional step statistics.
        if recorded:
            measured_totals = torch.stack([entry.totals for entry in recorded])
        else:
            measured_totals = epoch_totals.unsqueeze(0)
        if plan.world_size > 1:
            dist.all_reduce(measured_totals)
        train_stats = _classification_row(measured_totals.sum(dim=0).tolist())
        validation_start = time.perf_counter()
        validation = evaluate_model(model, validation_loader, device, plan.world_size)
        validation_runtime = time.perf_counter() - validation_start
        end_epoch = time.perf_counter()
        step_rows = [
            entry.timing_row(epoch_start_event, epoch_start - training_start, cumulative_training)
            for entry in recorded
        ]
        combined = _sum_timing([epoch_timer.to_dict(), *(_decode_timing(row[3:]) for row in step_rows)])
        timing_by_rank = _gather_timing_rows(
            [[end_epoch - epoch_start, training_runtime, validation_runtime,
              end_epoch - training_start, *_encode_timing(combined)],
             *([row[0], row[1], row[2], 0.0, *row[3:]] for row in step_rows)],
            device, plan.world_size,
        )
        primary_rank = max(range(plan.world_size), key=lambda index: timing_by_rank[index][0][1])
        for index, rows in enumerate(timing_by_rank):
            local = rows[0]
            rank_timings[index].append({
                "epoch_runtime": local[0], "training_runtime": local[1], "validation_runtime": local[2],
                **_decode_timing(local[4:]),
            })
        training_runtime = timing_by_rank[primary_rank][0][1]
        cumulative_training += training_runtime
        total_communication.add(epoch_communication)
        _merge_errors(total_errors, epoch_errors)

        train_samples = int(train_stats["samples"])
        samples_before_epoch = samples_processed
        samples_processed += train_samples
        full_epoch = steps_this_epoch == len(train_loader)
        full_epochs += int(full_epoch)
        full_epoch_samples = len(train_loader) * plan.global_batch_size
        epoch_row = {
            "epoch": epoch, "step": steps, "learning_rate": learning_rate,
            "train_loss": train_stats["loss"], "train_accuracy": train_stats["accuracy"],
            "train_samples": train_samples, "samples_processed": samples_processed,
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "validation_samples": validation["samples"], "full_epoch": full_epoch,
            "dropped_train_samples": len(datasets.train) - full_epoch_samples,
            "unique_train_samples": train_samples, "padded_train_samples": 0,
            "unprocessed_budget_samples": full_epoch_samples - train_samples,
            "epoch_runtime": max(rows[0][0] for rows in timing_by_rank),
            "training_runtime": training_runtime,
            "validation_runtime": max(rows[0][2] for rows in timing_by_rank),
            "cumulative_training_time": cumulative_training,
            "cumulative_runtime": max(rows[0][3] for rows in timing_by_rank),
            "samples_per_second": train_samples / training_runtime,
            "timing": _decode_timing(timing_by_rank[primary_rank][0][4:]),
            "gradient_errors": _serialize_errors(epoch_errors), "communication": epoch_communication.to_dict(),
        }
        epoch_metrics.append(epoch_row)
        if recorded and rank == 0:
            for index, (entry, totals) in enumerate(zip(recorded, measured_totals.cpu().tolist()), start=1):
                step_time = max(rows[index] for rows in timing_by_rank)
                stats = _classification_row(totals)
                step_metrics.append({
                    "epoch": epoch, "step": steps - steps_this_epoch + index, "learning_rate": learning_rate,
                    "train_loss": stats["loss"], "train_accuracy": stats["accuracy"],
                    "train_samples": stats["samples"],
                    "samples_processed": samples_before_epoch + index * plan.global_batch_size,
                    "step_runtime": step_time[0],
                    "cumulative_runtime": max(rows[index][1] for rows in timing_by_rank),
                    "cumulative_training_time": max(rows[index][2] for rows in timing_by_rank),
                    "timing": _decode_timing(step_time[4:]),
                    "gradient_errors": _serialize_errors(entry.sync.errors),
                    "communication": entry.sync.communication.to_dict(),
                })

        if tracker.observe(epoch_row, total_communication) and config.stop_on_target:
            stop_reason = "target_reached"
            break
        if config.max_steps is not None and steps >= config.max_steps:
            stop_reason = "step_budget"
            break
        if validation["loss"] < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss, epochs_without_improvement = validation["loss"], 0
        else:
            epochs_without_improvement += 1
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stop_reason = "early_stopping"
            break
        if scheduler is not None:
            scheduler.step()

    test_stats = test_runtime = None
    if datasets.test is not None:
        test_start = time.perf_counter()
        test_stats = evaluate_model(
            model, create_eval_loader(datasets.test, config, rank, plan.world_size, device),
            device, plan.world_size,
        )
        test_runtime = time.perf_counter() - test_start
    phase_totals = _sum_timing(row["timing"] for row in epoch_metrics)
    last = epoch_metrics[-1]
    validation_total = sum(row["validation_runtime"] for row in epoch_metrics)
    summary = {
        "training_runtime": cumulative_training, "validation_runtime": validation_total,
        "train_and_validation_runtime": last["cumulative_runtime"],
        "warmup_runtime": warmup_runtime, "test_runtime": test_runtime,
        "mean_epoch_runtime": sum(row["epoch_runtime"] for row in epoch_metrics) / len(epoch_metrics),
        "mean_step_runtime": cumulative_training / steps,
        "synchronization_runtime": phase_totals.get("synchronize"),
        "compression_runtime": phase_totals.get("compression"),
        "decompression_runtime": phase_totals.get("decompression"),
        "collective_runtime": phase_totals.get("collective"),
        "aggregation_runtime": phase_totals.get("aggregation"),
        "data_loading_runtime": phase_totals.get("data_loading"),
        "final_train_loss": last["train_loss"], "final_train_accuracy": last["train_accuracy"],
        "final_validation_loss": last["validation_loss"], "final_validation_accuracy": last["validation_accuracy"],
        "final_test_loss": test_stats["loss"] if test_stats else None,
        "final_test_accuracy": test_stats["accuracy"] if test_stats else None,
        "test_samples": test_stats["samples"] if test_stats else None,
        "optimizer_steps": steps, "local_batches": steps, "global_batches": steps,
        "total_rank_batches": steps * plan.world_size,
        "samples_processed": samples_processed, "samples_per_second": samples_processed / cumulative_training,
        "epochs_completed": full_epochs, "validation_evaluations": len(epoch_metrics), "stop_reason": stop_reason,
        "local_batch_size": plan.local_batch_size, "global_batch_size": plan.global_batch_size,
        "world_size": plan.world_size, "num_gpus": len(plan.device_ids),
        "timing": phase_totals, "rank_timing": [_sum_timing(timings) for timings in rank_timings],
    }
    summary["mean_rank_training_runtime"] = sum(
        timing["training_runtime"] for timing in summary["rank_timing"]
    ) / plan.world_size
    return {
        "summary": summary, "epoch_metrics": epoch_metrics, "step_metrics": step_metrics,
        "gradient_error_metrics": _serialize_errors(total_errors),
        "communication_metrics": total_communication.to_dict(), "convergence_metrics": tracker.to_dict(),
    }


# Process management stays at the experiment boundary. The same worker function
# serves plain Python and externally launched torchrun processes.
def environment_info() -> dict:
    """Describe visible hardware, interconnect topology and installed software."""
    cuda_available = torch.cuda.is_available()
    visible = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            visible.append({
                "index": index, "name": props.name, "memory_bytes": props.total_memory,
                "compute_capability": [props.major, props.minor],
                "uuid": str(props.uuid) if getattr(props, "uuid", None) is not None else None,
            })
    topology = None
    if visible and shutil.which("nvidia-smi"):
        try:
            topology = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True, timeout=4, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    nccl_version = None
    if cuda_available and dist.is_nccl_available():
        nccl_version = list(torch.cuda.nccl.version())
    return {
        "hardware": {
            "hostname": platform.node(), "platform": platform.platform(), "cpu": platform.processor() or None,
            "logical_cpus": os.cpu_count(), "visible_gpus": visible, "topology": topology, "interconnect": None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "software_environment": {
            "python": platform.python_version(), "pytorch": str(torch.__version__),
            "torchvision": _package_version("torchvision"), "cuda": torch.version.cuda,
            "cuda_available": cuda_available, "nccl": nccl_version,
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _prepare_rank_datasets(config: ExperimentConfig, rank: int, plan: ExecutionPlan) -> DatasetBundle:
    if config.dataset != "emnist":
        return create_datasets(config)
    device_ids = [plan.device_ids[rank]] if plan.device_type == "cuda" else None
    if rank == 0:
        with _file_lock(Path(config.data_dir) / ".emnist-download.lock"):
            datasets = create_datasets(config)
        if plan.world_size > 1:
            dist.barrier(device_ids=device_ids)
        return datasets
    dist.barrier(device_ids=device_ids)
    return create_datasets(config, download=False)


def _execute_worker(config: ExperimentConfig, experiment_id: str, repetition: int) -> None:
    wall_start = time.perf_counter()
    plan = resolve_execution(config)
    launched_world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if launched_world != plan.world_size or rank != local_rank or not 0 <= rank < launched_world:
        raise ValueError(
            f"Expected {plan.world_size} local workers, got WORLD_SIZE={launched_world}. "
            f"Launch with torchrun --standalone --nproc-per-node={plan.world_size}."
        )
    device = plan.device(local_rank)
    result = {
        "schema_version": 1, "experiment_id": experiment_id, "repetition": repetition,
        "status": "failed", "configuration": asdict(config),
        "hardware": {}, "software_environment": {}, "dataset_metadata": {}, "model_metadata": {},
        "summary": {}, "epoch_metrics": [], "step_metrics": [], "gradient_error_metrics": {},
        "communication_metrics": {}, "convergence_metrics": {},
    }
    try:
        init_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_num_threads(config.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        if plan.world_size > 1:
            dist.init_process_group(backend=plan.backend, timeout=timedelta(seconds=config.timeout_seconds))
        init_runtime = time.perf_counter() - init_start

        if rank == 0:
            result.update(environment_info())
            result["hardware"].update({
                "device_type": plan.device_type, "selected_gpu_ids": list(plan.device_ids),
                "cpu_threads": config.cpu_threads, "world_size": plan.world_size,
                "interconnect": config.interconnect,
            })
        prep_start = time.perf_counter()
        datasets = _prepare_rank_datasets(config, rank, plan)
        prep_runtime = time.perf_counter() - prep_start
        result["dataset_metadata"] = {**asdict(datasets.metadata), "num_classes": datasets.metadata.num_classes}

        torch.manual_seed(_rank_seed(config.seed, rank))
        model = create_model(config, datasets.metadata).to(device=device, dtype=torch.float32)
        broadcast_model(model, plan.world_size)
        result["model_metadata"] = {
            "architecture": config.model,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "gradient_bytes": sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
        }
        result.update(_train(config, model, datasets, device, rank, plan))
        setup_and_test = _gather_timing_rows(
            [[init_runtime, prep_runtime, result["summary"]["test_runtime"] or 0.0]],
            device, plan.world_size,
        )
        result["summary"].update({
            "initialization_runtime": max(rows[0][0] for rows in setup_and_test),
            "dataset_preparation_runtime": max(rows[0][1] for rows in setup_and_test),
            "test_runtime": max(rows[0][2] for rows in setup_and_test) if datasets.test is not None else None,
        })
    except Exception as error:
        if rank == 0:
            result["error"] = f"{type(error).__name__}: {error}"
            result["summary"]["total_runtime"] = time.perf_counter() - wall_start
            try:
                _store_result(config, result)
            except Exception as storage_error:
                print(f"Could not save failure record: {storage_error}", file=sys.stderr)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        result["status"] = "completed"
        result["summary"]["total_runtime"] = time.perf_counter() - wall_start
        _store_result(config, result, model if config.save_checkpoint else None)
        print(f"Completed {experiment_id} in {config.output_file}", flush=True)


# One coordinator appends each result under a file lock. Per-experiment archive
# names and atomic replacements keep prior records readable across failed writes.
def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(filename)
    try:
        write(temporary)
        with temporary.open("rb") as contents:
            os.fsync(contents.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_document(output: Path) -> dict:
    if not output.exists():
        return {"schema_version": 1, "experiments": []}
    document = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported results format in {output}")
    if not isinstance(document.get("experiments"), list):
        raise ValueError(f"Missing experiments list in {output}")
    return document


def load_results(output_file: str | Path) -> list[dict]:
    """Read stored experiment records, or return an empty list for a new file."""
    return _read_document(Path(output_file))["experiments"]


def _write_step_archive(path: Path, rows: list[dict], file_format: str) -> None:
    if file_format == "parquet":
        import pandas as pd

        frame = pd.DataFrame(rows)
        for key in ("timing", "gradient_errors", "communication"):
            if key in frame:
                frame[key] = frame[key].map(json.dumps)
        _atomic_write(path, lambda temporary: frame.to_parquet(temporary, index=False))
        return

    def write_csv(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) for key, value in row.items()})

    _atomic_write(path, write_csv)


def load_step_metrics(result: dict, output_file: str | Path) -> list[dict]:
    """Load a result's inline or external per-step rows."""
    archive = result.get("step_metrics_archive")
    if archive is None:
        return result.get("step_metrics", [])
    path = Path(output_file).parent / archive["path"]
    if archive["format"] == "parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        for key in ("timing", "gradient_errors", "communication"):
            if key in frame:
                frame[key] = frame[key].map(json.loads)
        return _json_value(frame.to_dict(orient="records"))
    if archive["format"] == "csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return [{key: json.loads(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    raise ValueError(f"Unknown step archive format {archive['format']!r}")


def _store_result(config: ExperimentConfig, result: dict, model: nn.Module | None = None) -> None:
    output = Path(config.output_file)
    with _file_lock(output.with_name(f"{output.name}.lock")):
        document = _read_document(output)
        experiment_id = result["experiment_id"]
        if any(previous["experiment_id"] == experiment_id for previous in document["experiments"]):
            raise ValueError(f"Result {experiment_id} is already stored in {output}")

        artifact_directory = output.parent / f"{output.stem}-artifacts"
        if result["step_metrics"] and config.step_metrics_format != "inline":
            file_format = config.step_metrics_format
            archive = artifact_directory / f"{experiment_id}-steps.{file_format}"
            _write_step_archive(archive, _json_value(result["step_metrics"]), file_format)
            result["step_metrics_archive"] = {
                "format": file_format, "path": archive.relative_to(output.parent).as_posix(),
                "rows": len(result["step_metrics"]),
            }
            result["step_metrics"] = []
        if model is not None:
            checkpoint = artifact_directory / f"{experiment_id}-model.pt"
            _atomic_write(checkpoint, lambda temporary: torch.save(model.state_dict(), temporary))
            result["checkpoint"] = checkpoint.relative_to(output.parent).as_posix()
        document["experiments"].append(_json_value(result))
        _atomic_write(
            output,
            lambda temporary: temporary.write_text(
                json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            ),
        )


def _launch_trial(config: ExperimentConfig, plan: ExecutionPlan, repetition: int) -> dict:
    experiment_id = uuid4().hex
    command = [sys.executable]
    if plan.world_size > 1:
        command.extend(("-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={plan.world_size}"))
    command.extend((str(Path(__file__).resolve()), "--experiment-id", experiment_id, "--repetition", str(repetition)))
    with tempfile.TemporaryDirectory(prefix="gradient-compression-") as directory:
        config_file = Path(directory) / "config.json"
        config_file.write_text(json.dumps(asdict(config)), encoding="utf-8")
        command.extend(("--config", str(config_file)))
        completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"Experiment {experiment_id} exited with status {completed.returncode}.\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return next(result for result in load_results(config.output_file) if result["experiment_id"] == experiment_id)


def run_experiment(config: ExperimentConfig) -> dict:
    """Launch through a fresh Python process, returning the first trial when repeated."""
    return run_experiments([config])[0]


def run_experiments(
    configurations: Iterable[ExperimentConfig], *, order_seed: int | None = None,
) -> list[dict]:
    """Validate all settings, then run repeated trials sequentially in a reproducible order."""
    configurations = list(configurations)
    for config in configurations:
        config.validate()
    trials = [
        (config, resolve_execution(config), repetition)
        for config in configurations
        for repetition in range(config.repetitions)
    ]
    if order_seed is not None:
        random.Random(order_seed).shuffle(trials)
    return [_launch_trial(config, plan, repetition) for config, plan, repetition in trials]


def _setting(value: str) -> tuple[str, Any]:
    name, separator, raw = value.partition("=")
    if not separator or name not in {item.name for item in fields(ExperimentConfig)}:
        raise argparse.ArgumentTypeError("Use a known ExperimentConfig field as NAME=VALUE")
    try:
        return name, json.loads(raw)
    except json.JSONDecodeError:
        return name, raw


def main(argv: list[str] | None = None) -> None:
    """Run a configuration directly or act as one worker under torchrun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to an ExperimentConfig JSON object")
    parser.add_argument("--set", action="append", default=[], type=_setting, metavar="NAME=VALUE")
    parser.add_argument("--experiment-id", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(values, dict):
            raise ValueError("A configuration file must contain one JSON object")
        values.update(dict(args.set))
        config = ExperimentConfig(**values)
        config.validate()
        if args.experiment_id and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.experiment_id):
            raise ValueError("Invalid experiment ID")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.experiment_id:
        _execute_worker(config, args.experiment_id, args.repetition)
    else:
        for repetition in range(config.repetitions):
            # torchrun ranks must independently generate the same identifier.
            launch_id = os.environ.get("TORCHELASTIC_RUN_ID")
            experiment_id = uuid4().hex if not launch_id else _shared_experiment_id(launch_id, repetition)
            _execute_worker(config, experiment_id, repetition)


def _shared_experiment_id(launch_id: str, repetition: int) -> str:
    from uuid import NAMESPACE_URL, uuid5

    return uuid5(NAMESPACE_URL, f"gradient-compression/{launch_id}/{repetition}").hex


if __name__ == "__main__":
    main()
