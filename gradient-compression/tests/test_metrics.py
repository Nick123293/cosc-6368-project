import math

import pytest

from distributed_training import (
    ClassificationStats,
    CommunicationStats,
    ConvergenceTracker,
    ErrorStats,
    ExperimentConfig,
    compare_runtime,
)


def test_reconstruction_metrics_weight_elements_at_every_granularity():
    first = ErrorStats(squared_error_sum=5, absolute_error_sum=3, elements=2, max_error=2)
    second = ErrorStats(squared_error_sum=25, absolute_error_sum=5, elements=1, max_error=5)
    first_epoch = ErrorStats()
    first_epoch.add(first)
    first_epoch.add(second)
    second_epoch = ErrorStats(squared_error_sum=4, absolute_error_sum=4, elements=4, max_error=1)
    overall = ErrorStats()
    overall.add(first_epoch)
    overall.add(second_epoch)

    assert first_epoch.to_dict()["mse"] == pytest.approx(30 / 3)
    assert overall.to_dict() == pytest.approx(
        {
            "squared_error_sum": 34,
            "absolute_error_sum": 12,
            "elements": 7,
            "max_error": 5,
            "mse": 34 / 7,
            "rmse": math.sqrt(34 / 7),
            "mae": 12 / 7,
        }
    )
    assert ErrorStats().to_dict()["mse"] is None


def test_classification_metrics_weight_samples():
    small_batch = ClassificationStats(loss_sum=4.0, correct=0, samples=1)
    large_batch = ClassificationStats(loss_sum=2.0, correct=3, samples=4)
    aggregate = ClassificationStats()
    aggregate.add(small_batch)
    aggregate.add(large_batch)

    values = aggregate.to_dict()
    assert values["loss"] == pytest.approx((4 + 2) / 5)
    assert values["accuracy"] == pytest.approx(3 / 5)
    assert values["samples"] == 5


def test_communication_counters_and_all_transmitted_bytes():
    total = CommunicationStats()
    for _ in range(2):
        total.add(
            CommunicationStats(
                rounds=1,
                collective_calls=3,
                diagnostic_collective_calls=2,
                communicated_tensors=3,
                num_elements=32,
                original_bytes=128,
                payload_bytes=10,
                normalization_bytes=4,
                metadata_bytes=8,
                padding_bytes=2,
                logical_bytes=24,
                estimated_aggregate_bytes=48,
            )
        )

    values = total.to_dict()
    assert values["rounds"] == 2
    assert values["collective_calls"] == 6
    assert values["diagnostic_collective_calls"] == 4
    assert values["compression_ratio"] == pytest.approx(256 / 48)
    assert values["effective_bits_per_element"] == pytest.approx(8 * 48 / 64)
    assert values["logical_bytes"] == 48
    assert values["estimated_aggregate_bytes"] == 96


@pytest.mark.parametrize(
    "baseline,runtime,speedup,reduction", [(6.0, 4.0, 1.5, 100 / 3), (4.0, 5.0, 0.8, -25.0)]
)
def test_runtime_comparison_includes_slowdowns(baseline, runtime, speedup, reduction):
    values = compare_runtime(baseline, runtime)
    assert values["speedup"] == pytest.approx(speedup)
    assert values["runtime_reduction_percent"] == pytest.approx(reduction)


def test_convergence_records_first_observed_target_and_unreached_targets():
    tracker = ConvergenceTracker(ExperimentConfig(target_validation_loss=0.2))
    first_epoch = CommunicationStats(rounds=5, collective_calls=10, estimated_aggregate_bytes=1000)
    communication = CommunicationStats()
    communication.add(first_epoch)
    evaluation = {
        "epoch": 1,
        "step": 5,
        "cumulative_runtime": 2.5,
        "validation_loss": 0.23,
        "validation_accuracy": 0.9,
    }

    tracker.observe(evaluation, communication)
    assert tracker.to_dict()["target_reached"] is False
    assert tracker.to_dict()["time_to_target"] is None

    communication.add(first_epoch)
    tracker.observe(
        {**evaluation, "epoch": 2, "step": 10, "cumulative_runtime": 5.0, "validation_loss": 0.19},
        communication,
    )
    tracker.observe(
        {**evaluation, "epoch": 3, "step": 15, "cumulative_runtime": 7.5, "validation_loss": 0.10},
        communication,
    )
    values = tracker.to_dict()
    assert values["target_reached"] is True
    assert values["time_to_target"] == pytest.approx(5.0)
    assert values["epoch_to_target"] == 2
    assert values["step_to_target"] == 10
    assert values["global_batches_to_target"] == 10
    assert values["synchronization_rounds_to_target"] == 10
    assert values["collective_calls_to_target"] == 20
    assert values["communication_bytes_to_target"] == 2000


def test_unconfigured_convergence_is_unmeasured():
    values = ConvergenceTracker(ExperimentConfig()).to_dict()
    assert values["target_reached"] is None
    assert values["time_to_target"] is None
