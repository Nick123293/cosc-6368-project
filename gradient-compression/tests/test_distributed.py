from dataclasses import replace
from datetime import timedelta
import socket
from string import ascii_lowercase

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

from distributed_training import (
    ExperimentConfig,
    GradientSynchronizer,
    broadcast_model,
    create_datasets,
    create_eval_loader,
    create_train_loader,
    load_results,
    run_experiment,
)


class RecordingCompressor:
    def __init__(self, compressor):
        self.compressor = compressor
        self.packets = []

    def compress(self, *args, **kwargs):
        packet = self.compressor.compress(*args, **kwargs)
        self.packets.append(packet)
        return packet

    def __getattr__(self, name):
        return getattr(self.compressor, name)


def test_emnist_letters_orientation_labels_and_held_out_test(monkeypatch):
    from PIL import Image
    from torchvision import datasets

    class LettersFixture:
        classes = datasets.EMNIST.classes_split_dict["letters"]

        def __init__(self, *, train, transform, target_transform, **kwargs):
            assert kwargs["split"] == "letters"
            self.train = train
            self.transform = transform
            self.target_transform = target_transform
            self.targets = torch.arange(1, 27)
            self.data = torch.zeros((26, 28, 28), dtype=torch.uint8)
            self.data[:, 2, 7] = 255

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            image = Image.fromarray(self.data[index].numpy())
            return self.transform(image), self.target_transform(int(self.targets[index]))

    monkeypatch.setattr(datasets, "EMNIST", LettersFixture)
    config = ExperimentConfig(emnist_split="letters")
    bundle = create_datasets(config, download=False)
    repeated = create_datasets(config, download=False)

    assert bundle.metadata.classes == tuple(ascii_lowercase)
    assert bundle.train.dataset.train and bundle.validation.dataset.train
    assert not bundle.test.train
    assert bundle.train.indices == repeated.train.indices
    assert bundle.validation.indices == repeated.validation.indices
    assert set(bundle.train.indices).isdisjoint(bundle.validation.indices)
    assert len(bundle.train) + len(bundle.validation) == 26
    image, first_label = bundle.test[0]
    assert first_label == 0
    assert bundle.test[25][1] == 25
    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert image[0, 7, 2] == 1
    assert image[0, 2, 7] == -1


def _check_network(backend):
    if not dist.is_available():
        pytest.skip("PyTorch distributed unavailable")
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < 2):
        pytest.skip("NCCL integration requires two visible CUDA GPUs")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local sockets are unavailable in the execution sandbox")


def _all_gather(tensor, world_size):
    received = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(received, tensor)
    return received


def _decode_nibbles(payload, scales, num_elements, bucket_size):
    payload_bytes = payload.tolist()
    decoded = []
    for element in range(num_elements):
        nibble = (payload_bytes[element // 2] >> (4 * (element % 2))) & 15
        signed_index = nibble - 16 if nibble >= 8 else nibble
        decoded.append(float(scales[element // bucket_size]) * signed_index / 7)
    return scales.new_tensor(decoded)


def _check_partitions(config, rank, world_size, device):
    dataset = TensorDataset(torch.arange(19), torch.zeros(19, dtype=torch.long))
    loader, sampler = create_train_loader(dataset, config, rank, world_size, device)
    sampler.set_epoch(0)
    first_epoch = torch.cat([ids for ids, _ in loader]).to(device)
    sampler.set_epoch(1)
    second_epoch = torch.cat([ids for ids, _ in loader]).to(device)
    sampler.set_epoch(0)
    torch.testing.assert_close(torch.cat([ids for ids, _ in loader]), first_epoch.cpu())

    assert first_epoch.numel() == 8
    gathered = torch.cat(_all_gather(first_epoch, world_size))
    assert gathered.numel() == gathered.unique().numel() == 16
    assert not torch.equal(first_epoch, second_epoch)

    evaluation = create_eval_loader(dataset, config, rank, world_size, device)
    evaluated = torch.cat([ids for ids, _ in evaluation])
    torch.testing.assert_close(evaluated, torch.arange(rank, len(dataset), world_size))


def _check_compressed_aggregation(config, model, local_gradient, rank, world_size, reference):
    compressed = replace(
        config,
        synchronization="compressed_allgather",
        compression="qsgd",
        encoding="packed",
        normalization="max",
        quantization_levels=7,
        quantization_bits=4,
        quantization_bucket_size=3,
    )
    synchronizer = GradientSynchronizer(compressed, model.named_parameters(), rank, world_size)
    recorder = RecordingCompressor(synchronizer.compressor)
    synchronizer.compressor = recorder
    model.weight.grad = local_gradient.clone()

    result = synchronizer.synchronize(step=1)

    assert len(recorder.packets) == 1
    packet = recorder.packets[0]
    payloads = _all_gather(packet.payload, world_size)
    scales = _all_gather(packet.scales, world_size)
    reconstructed = torch.stack(
        [_decode_nibbles(payload, scale, 5, 3) for payload, scale in zip(payloads, scales)]
    )
    compressed_reference = reconstructed.mean(dim=0).view_as(reference)
    torch.testing.assert_close(model.weight.grad, compressed_reference)
    assert result.communication.rounds == 1
    assert result.communication.collective_calls == 2
    assert result.communication.normalization_bytes == 2 * 4
    assert result.communication.payload_bytes == 3
    assert result.communication.original_bytes == 5 * 4
    assert result.communication.diagnostic_collective_calls == 0

    diagnostic = replace(compressed, mode="diagnostic", gradient_error="both", per_layer_error=True)
    diagnostic_sync = GradientSynchronizer(diagnostic, model.named_parameters(), rank, world_size)
    model.weight.grad = local_gradient.clone()
    measured = diagnostic_sync.synchronize(step=1)
    torch.testing.assert_close(model.weight.grad, compressed_reference)

    full_precision = torch.cat(_all_gather(local_gradient, world_size))
    local_error = (reconstructed - full_precision).abs()
    aggregated_error = (compressed_reference - reference).abs()
    local = measured.errors["local"].to_dict()
    aggregated = measured.errors["aggregated"].to_dict()
    assert local["elements"] == world_size * 5
    assert aggregated["elements"] == 5
    assert local["mse"] == pytest.approx(local_error.square().mean().item(), abs=2e-6)
    assert local["max_error"] == pytest.approx(local_error.max().item(), abs=2e-6)
    assert aggregated["mse"] == pytest.approx(aggregated_error.square().mean().item(), abs=2e-6)
    assert aggregated["max_error"] == pytest.approx(aggregated_error.max().item(), abs=2e-6)
    assert measured.errors["local:weight"].to_dict() == local
    assert measured.errors["aggregated:weight"].to_dict() == aggregated
    assert measured.communication.diagnostic_collective_calls > 0
    assert measured.communication.collective_calls == result.communication.collective_calls
    assert measured.communication.logical_bytes == result.communication.logical_bytes

    diagnostic_step_two = GradientSynchronizer(
        replace(diagnostic, gradient_error_frequency=2), model.named_parameters(), rank, world_size
    )
    model.weight.grad = local_gradient.clone()
    unmeasured = diagnostic_step_two.synchronize(step=1)
    assert not unmeasured.errors
    assert unmeasured.communication.diagnostic_collective_calls == 0
    torch.testing.assert_close(model.weight.grad, compressed_reference)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    starting_weight = model.weight.detach().clone()
    optimizer.step()
    expected_weight = starting_weight - 0.125 * compressed_reference
    for replica in _all_gather(model.weight.detach(), world_size):
        torch.testing.assert_close(replica, expected_weight)


def _check_variable_length_metadata(config, model, rank, world_size):
    elias = replace(
        config,
        synchronization="compressed_allgather",
        compression="qsgd",
        encoding="elias",
        normalization="max",
    )
    synchronizer = GradientSynchronizer(elias, model.named_parameters(), rank, world_size)
    recorder = RecordingCompressor(synchronizer.compressor)
    synchronizer.compressor = recorder
    model.weight.grad = torch.full_like(model.weight, float(rank))

    result = synchronizer.synchronize(step=1)
    packet = recorder.packets[0]
    length = model.weight.new_tensor([packet.payload_bytes], dtype=torch.int64)
    lengths = [size.item() for size in _all_gather(length, world_size)]

    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 0.5))
    assert lengths[0] < lengths[1]
    assert result.communication.payload_bytes == lengths[rank]
    assert result.communication.padding_bytes == max(lengths) - lengths[rank]
    assert result.communication.normalization_bytes == 4
    assert result.communication.metadata_bytes == 8
    assert result.communication.collective_calls == 3


def _distributed_checks(rank, world_size, rendezvous, backend):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend, init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=35)
    )
    try:
        config = ExperimentConfig(
            device=device.type,
            num_gpus=world_size if device.type == "cuda" else 0,
            cpu_processes=world_size if device.type == "cpu" else 1,
            local_batch_size=2,
            communication_bucket_bytes=4096,
        )
        model = nn.Linear(5, 1, bias=False, device=device)
        model.register_buffer("scale", torch.tensor(rank + 7.0, device=device))
        with torch.no_grad():
            model.weight.fill_(rank + 0.25)
        broadcast_model(model, world_size)
        assert torch.equal(model.weight, torch.full_like(model.weight, 0.25))
        assert model.scale.item() == 7.0

        _check_partitions(config, rank, world_size, device)

        gradients = torch.tensor([[1, -2, 3, -4, 5], [6, -5, 4, -3, 2]], device=device).float()
        local_gradient = gradients[rank : rank + 1]
        reference = gradients.mean(dim=0, keepdim=True)
        for mode in ("fp32_allreduce", "fp32_allgather", "compressed_allgather"):
            baseline = replace(config, synchronization=mode, compression="identity")
            model.weight.grad = local_gradient.clone()
            synchronizer = GradientSynchronizer(baseline, model.named_parameters(), rank, world_size)
            measured = synchronizer.synchronize(step=1)
            torch.testing.assert_close(model.weight.grad, reference)
            assert measured.communication.rounds == measured.communication.collective_calls == 1
            assert measured.communication.diagnostic_collective_calls == 0

        bucketed = replace(config, synchronization="fp32_allreduce", communication_bucket_bytes=8)
        model.weight.grad = local_gradient.clone()
        synchronizer = GradientSynchronizer(bucketed, model.named_parameters(), rank, world_size)
        measured = synchronizer.synchronize(step=1)
        torch.testing.assert_close(model.weight.grad, reference)
        assert measured.communication.rounds == 1
        assert measured.communication.collective_calls == 3

        _check_compressed_aggregation(config, model, local_gradient, rank, world_size, reference)
        _check_variable_length_metadata(config, model, rank, world_size)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_replicas_partitions_and_collectives(backend, tmp_path):
    _check_network(backend)
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_distributed_checks, args=(2, rendezvous, backend), nprocs=2, join=True)


def test_gloo_training_matches_equivalent_global_batch_and_stops_on_target(tmp_path):
    _check_network("gloo")
    config = ExperimentConfig(
        dataset="synthetic",
        device="cpu",
        num_gpus=0,
        cpu_processes=2,
        train_size_limit=24,
        validation_size_limit=7,
        test_size_limit=7,
        local_batch_size=2,
        hidden_dims=(8,),
        epochs=3,
        target_validation_loss=100.0,
        output_file=str(tmp_path / "experiments.json"),
    )
    distributed = run_experiment(config)
    single = run_experiment(replace(config, cpu_processes=1, local_batch_size=4))

    for result in (distributed, single):
        assert result["status"] == "completed"
        assert result["summary"]["optimizer_steps"] == 6
        assert result["summary"]["epochs_completed"] == 1
        assert result["summary"]["samples_processed"] == 24
        assert result["convergence_metrics"]["target_reached"] is True
        assert result["convergence_metrics"]["epoch_to_target"] == 1

    for metric in ("final_validation_loss", "final_validation_accuracy", "final_test_accuracy"):
        assert distributed["summary"][metric] == pytest.approx(single["summary"][metric], abs=1e-6)
    saved = load_results(config.output_file)
    assert {entry["experiment_id"] for entry in saved} == {
        single["experiment_id"],
        distributed["experiment_id"],
    }


def test_single_device_compressed_training_records_no_collectives(tmp_path):
    config = ExperimentConfig(
        dataset="synthetic",
        device="cpu",
        num_gpus=0,
        train_size_limit=16,
        validation_size_limit=7,
        test_size_limit=7,
        local_batch_size=4,
        hidden_dims=(8,),
        epochs=2,
        max_steps=3,
        synchronization="compressed_allgather",
        mode="diagnostic",
        gradient_error="both",
        output_file=tmp_path / "experiments.json",
    )

    result = run_experiment(config)

    assert result["status"] == "completed"
    assert result["summary"]["optimizer_steps"] == 3
    assert result["summary"]["samples_processed"] == 12
    communication = result["communication_metrics"]
    assert communication["rounds"] == 3
    assert communication["collective_calls"] == 0
    assert communication["diagnostic_collective_calls"] == 0
    assert communication["logical_bytes"] == 0
