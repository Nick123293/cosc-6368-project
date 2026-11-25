import math

import pytest
import torch

from distributed_training import (
    ExperimentConfig,
    QSGDCompressor,
    create_compressor,
    decode_elias_indices,
    encode_elias_indices,
    pack_indices,
    unpack_indices,
)


def test_identity_preserves_noncontiguous_gradient_exactly():
    gradient = torch.tensor([[0.0, 0.25, -1.0], [-5.0, 20.0, 0.5]]).T
    assert not gradient.is_contiguous()
    compressor = create_compressor(ExperimentConfig(compression="identity"))

    packet = compressor.compress(gradient)
    reconstructed = compressor.decompress(packet)

    assert torch.equal(reconstructed, gradient)
    assert reconstructed.shape == gradient.shape
    assert reconstructed.dtype == gradient.dtype
    assert reconstructed.device == gradient.device
    assert packet.original_bytes == packet.payload_bytes == gradient.numel() * 4
    assert packet.metadata_bytes == 0


@pytest.mark.parametrize("normalization", ["l2", "max"])
@pytest.mark.parametrize("encoding", ["packed", "integer", "elias"])
def test_qsgd_zero_gradient_including_partial_bucket(normalization, encoding):
    gradient = torch.zeros((3, 7))
    compressor = QSGDCompressor(
        levels=7, bits=4, normalization=normalization, bucket_size=8, encoding=encoding
    )

    packet = compressor.compress(gradient)

    assert torch.equal(packet.scales, torch.zeros(3))
    assert torch.equal(compressor.decompress(packet), gradient)
    assert packet.metadata_bytes == 3 * 4


@pytest.mark.parametrize("normalization", ["l2", "max"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_qsgd_shape_device_range_and_scale_decoding(normalization, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    gradient = torch.tensor(
        [[1e7, -6.0, 0.0], [0.001, -1e6, 8.0]], device=device_name
    ).T
    assert not gradient.is_contiguous()
    compressor = QSGDCompressor(
        levels=7, bits=4, normalization=normalization, bucket_size=5, seed=5
    )

    packet = compressor.compress(gradient)
    reconstructed = compressor.decompress(packet)
    indices = unpack_indices(packet.payload, 4, gradient.numel())
    first_bucket = gradient.reshape(-1)[:5]
    first_scale = (
        float(torch.linalg.vector_norm(first_bucket))
        if normalization == "l2"
        else float(first_bucket.abs().max())
    )
    expected_scales = gradient.new_tensor([first_scale, 8.0])
    reconstructed_reference = torch.cat(
        (indices[:5].float() * (first_scale / 7), indices[5:].float() * (8.0 / 7))
    ).reshape(gradient.shape)

    assert indices.min() >= -7 and indices.max() <= 7
    assert indices[4] == 0
    assert indices[5] == 7
    assert torch.all(gradient * reconstructed >= 0)
    assert reconstructed.shape == gradient.shape
    assert reconstructed.device == gradient.device
    assert reconstructed.dtype == gradient.dtype
    torch.testing.assert_close(packet.scales, expected_scales)
    torch.testing.assert_close(reconstructed, reconstructed_reference)
    assert packet.original_bytes == 24
    assert packet.payload_bytes == 3
    assert packet.metadata_bytes == 8


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_fixed_width_pack_round_trip_and_byte_count(bits):
    levels = (1 << (bits - 1)) - 1
    indices = torch.arange(-levels, levels + 1, dtype=torch.int32).repeat(3)
    indices = torch.cat((indices, torch.tensor([-levels, 0], dtype=torch.int32)))

    payload = pack_indices(indices, bits)

    assert payload.dtype == torch.uint8
    assert payload.numel() == math.ceil(indices.numel() * bits / 8)
    assert torch.equal(unpack_indices(payload, bits, indices.numel()), indices)


@pytest.mark.parametrize("levels,bits", [(2, 2), (8, 4), (128, 8)])
def test_quantization_levels_must_fit_signed_encoding(levels, bits):
    with pytest.raises(ValueError, match="levels.*fit"):
        QSGDCompressor(levels=levels, bits=bits)


def test_eight_bit_integer_and_subbyte_packed_wire_sizes():
    gradient = torch.linspace(-1, 1, 33)
    for encoding, levels, bits, expected_bytes in [
        ("integer", 127, 8, 33),
        ("packed", 7, 4, 17),
        ("packed", 1, 2, 9),
    ]:
        compressor = QSGDCompressor(levels=levels, bits=bits, encoding=encoding)
        packet = compressor.compress(gradient)
        assert packet.payload.element_size() * packet.payload.numel() == expected_bytes
        assert packet.payload_bytes == expected_bytes
        assert packet.metadata_bytes == 4
        assert compressor.decompress(packet).shape == gradient.shape


@pytest.mark.parametrize("zero_vector", [False, True])
def test_elias_sparse_stream_round_trip(zero_vector):
    indices = torch.zeros(4096, dtype=torch.int32)
    if not zero_vector:
        indices[[0, 1, 129, 2048, 4095]] = torch.tensor(
            [-1, 2, -7, 127, -3], dtype=torch.int32
        )

    payload = encode_elias_indices(indices)

    assert payload.dtype == torch.uint8
    assert payload.numel() < 32
    assert torch.equal(decode_elias_indices(payload, indices.numel()), indices)


def test_compression_streams_are_reproducible_and_rank_specific():
    config = ExperimentConfig(compression="qsgd", compression_seed=1234)
    gradient = torch.linspace(-0.9, 1.0, 1024)
    first = create_compressor(config, rank=0)
    repeated = create_compressor(config, rank=0)
    other_rank = create_compressor(config, rank=1)
    rng_state = torch.random.get_rng_state()

    for _ in range(3):
        actual = first.compress(gradient).payload
        assert torch.equal(actual, repeated.compress(gradient).payload)
        assert not torch.equal(actual, other_rank.compress(gradient).payload)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize("normalization", ["l2", "max"])
def test_empirical_unbiasedness_and_qsgd_l2_bound(normalization):
    gradient = torch.tensor([-0.91, -0.12, 0.0, 0.25, 0.49, 0.78])
    levels, trials = 3, 3000
    compressor = QSGDCompressor(
        levels=levels, bits=4, normalization=normalization, seed=2027
    )
    reconstructed = torch.stack(
        [compressor.decompress(compressor.compress(gradient)) for _ in range(trials)]
    ).double()
    values = gradient.double()
    scale = values.norm() if normalization == "l2" else values.abs().max()
    coordinates = levels * values.abs() / scale
    probabilities = coordinates - coordinates.floor()
    variance = (scale / levels).square() * probabilities * (1 - probabilities)
    sampling_tolerance = 6 * (variance / trials).sqrt() + 1e-6

    assert torch.all((reconstructed.mean(0) - values).abs() <= sampling_tolerance)

    if normalization == "l2":
        squared_errors = (reconstructed - values).square().sum(1)
        dimension = gradient.numel()
        bound = min(dimension / levels**2, math.sqrt(dimension) / levels) * scale**2
        sampling_error = 6 * squared_errors.std() / math.sqrt(trials)
        assert squared_errors.mean() <= bound + sampling_error
