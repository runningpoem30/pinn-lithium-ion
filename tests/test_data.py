from pathlib import Path

import numpy as np
import torch

from src.data import (
    DatasetScalers,
    build_data_pipeline_from_arrays,
    create_data_loaders,
)


def pipeline_arrays():
    counts = np.array([[3], [2], [3]], dtype=float)
    battery_ids = np.repeat([1, 2, 3], counts.reshape(-1).astype(int))
    cycles = np.concatenate([np.arange(1, count + 1) for count in (3, 2, 3)])

    # Give each battery a deliberately different feature distribution. This
    # makes it possible to prove validation/test normalization uses train stats.
    base = battery_ids.astype(float)[:, None] * 100.0
    features = base + np.arange(8, dtype=float)[None, :] + cycles[:, None]
    pcl = (battery_ids * 0.1 + cycles * 0.01)[:, None]
    rul = np.concatenate(
        [np.arange(2, -1, -1), np.arange(1, -1, -1), np.arange(2, -1, -1)]
    )
    return {
        "Features_mov_Flt": features,
        "RUL_Flt": rul[:, None].astype(float),
        "PCL_Flt": pcl.astype(float),
        "Cycles_Flt": cycles[:, None].astype(float),
        "Num_Cycles_Flt": counts,
        "train_ind": np.array([[1]], dtype=float),
        "test_ind": np.array([[2]], dtype=float),
        "secondary_test_ind": np.array([[3]], dtype=float),
    }


def test_pipeline_preserves_official_battery_split_without_leakage():
    bundle = build_data_pipeline_from_arrays(pipeline_arrays())
    assert bundle.train.battery_ids == {1}
    assert bundle.validation.battery_ids == {2}
    assert bundle.test.battery_ids == {3}
    bundle.assert_no_battery_leakage()
    assert len(bundle.train) == 3
    assert len(bundle.validation) == 2
    assert len(bundle.test) == 3


def test_scalers_are_fitted_only_on_training_records():
    arrays = pipeline_arrays()
    bundle = build_data_pipeline_from_arrays(arrays)
    expected_train_features = arrays["Features_mov_Flt"][:3]
    assert np.allclose(
        bundle.scalers.features.mean, expected_train_features.mean(axis=0)
    )
    assert np.allclose(
        bundle.scalers.features.scale, expected_train_features.std(axis=0)
    )

    train_normalized = bundle.train.tensors["features_normalized"].numpy()
    validation_normalized = bundle.validation.tensors["features_normalized"].numpy()
    assert np.allclose(train_normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train_normalized.std(axis=0), 1.0, atol=1e-6)
    assert not np.allclose(validation_normalized.mean(axis=0), 0.0, atol=1e-3)
    expected_train_rul = arrays["RUL_Flt"][:3]
    assert bundle.scalers.target_rul is not None
    assert np.allclose(bundle.scalers.target_rul.mean, expected_train_rul.mean(axis=0))
    assert np.allclose(bundle.scalers.target_rul.scale, expected_train_rul.std(axis=0))


def test_raw_cycle_is_preserved_and_can_become_an_autograd_leaf():
    bundle = build_data_pipeline_from_arrays(pipeline_arrays())
    assert torch.equal(
        bundle.train.tensors["cycle_raw"].reshape(-1),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    cycle = bundle.train.tensors["cycle_raw"].clone().detach().requires_grad_(True)
    output = (cycle**2).sum()
    output.backward()
    assert torch.equal(cycle.grad, 2 * cycle.detach())


def test_scalers_round_trip_without_pickle(tmp_path: Path):
    bundle = build_data_pipeline_from_arrays(pipeline_arrays())
    path = tmp_path / "scalers.npz"
    bundle.scalers.save(path)
    restored = DatasetScalers.load(path)
    assert np.array_equal(restored.features.mean, bundle.scalers.features.mean)
    assert np.array_equal(restored.features.scale, bundle.scalers.features.scale)
    assert np.array_equal(restored.cycle.mean, bundle.scalers.cycle.mean)
    assert np.array_equal(restored.target_pcl.scale, bundle.scalers.target_pcl.scale)
    assert restored.target_rul is not None
    assert bundle.scalers.target_rul is not None
    assert np.array_equal(restored.target_rul.mean, bundle.scalers.target_rul.mean)
    assert np.array_equal(restored.target_rul.scale, bundle.scalers.target_rul.scale)


def test_rul_targets_match_audited_formula_and_preserve_identity():
    bundle = build_data_pipeline_from_arrays(pipeline_arrays())
    for dataset in (bundle.train, bundle.validation, bundle.test):
        tensors = dataset.tensors
        assert torch.equal(
            tensors["target_rul"], tensors["eol_cycle"] - tensors["cycle_raw"]
        )
        assert torch.all(tensors["target_rul"] >= 0)
        assert tensors["battery_id"].dtype == torch.int64
    assert torch.equal(
        bundle.train.tensors["target_rul"].reshape(-1),
        torch.tensor([2.0, 1.0, 0.0]),
    )


def test_validation_and_test_rul_cannot_influence_training_scaler():
    original = pipeline_arrays()
    shifted = {name: values.copy() for name, values in original.items()}
    # Whole-battery shifts preserve the audited cycle + RUL endpoint invariant.
    shifted["RUL_Flt"][3:, 0] += 10_000
    first = build_data_pipeline_from_arrays(original)
    second = build_data_pipeline_from_arrays(shifted)
    assert first.scalers.target_rul is not None
    assert second.scalers.target_rul is not None
    assert np.array_equal(first.scalers.target_rul.mean, second.scalers.target_rul.mean)
    assert np.array_equal(first.scalers.target_rul.scale, second.scalers.target_rul.scale)
    first.assert_no_battery_leakage()
    second.assert_no_battery_leakage()


def test_data_loaders_return_named_batched_tensors():
    bundle = build_data_pipeline_from_arrays(pipeline_arrays())
    loaders = create_data_loaders(bundle, batch_size=2, seed=7)
    batch = next(iter(loaders.train))
    assert set(batch) == {
        "features_raw",
        "features_normalized",
        "cycle_raw",
        "cycle_normalized",
        "target_pcl",
        "target_pcl_normalized",
        "target_rul",
        "target_rul_normalized",
        "eol_cycle",
        "battery_id",
    }
    assert batch["features_normalized"].shape == (2, 8)
    assert batch["cycle_raw"].shape == (2, 1)
    assert batch["battery_id"].dtype == torch.int64
