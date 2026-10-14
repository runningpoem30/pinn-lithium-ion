"""Battery-wise PyTorch data pipeline for the DeepHPM experiments.

The official MATLAB split is interpreted as follows:

* ``train_ind`` -> training batteries
* ``test_ind`` -> validation batteries
* ``secondary_test_ind`` -> final test batteries

All standardization statistics are fitted exclusively on training records.
Raw cycle values remain separate tensors so the DeepHPM model can later make
them autograd leaves and differentiate predictions with respect to physical
cycle number.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from src.data_audit import (
    FEATURE_NAMES,
    audit_arrays,
    reconstruct_battery_boundaries,
)


@dataclass(frozen=True)
class StandardizationStats:
    """Column-wise population mean and standard deviation."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, name: str) -> "StandardizationStats":
        values = _as_float_matrix(values, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Cannot fit {name} scaler on non-finite values")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        constant_columns = np.flatnonzero(scale == 0)
        if constant_columns.size:
            raise ValueError(
                f"Cannot standardize constant {name} columns: "
                f"{constant_columns.tolist()}"
            )
        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, values: np.ndarray, name: str = "values") -> np.ndarray:
        values = _as_float_matrix(values, name)
        if values.shape[1] != self.mean.size:
            raise ValueError(
                f"{name} has {values.shape[1]} columns; scaler expects {self.mean.size}"
            )
        return (values - self.mean) / self.scale

    def inverse_transform(
        self, values: np.ndarray, name: str = "values"
    ) -> np.ndarray:
        values = _as_float_matrix(values, name)
        if values.shape[1] != self.mean.size:
            raise ValueError(
                f"{name} has {values.shape[1]} columns; scaler expects {self.mean.size}"
            )
        return values * self.scale + self.mean


@dataclass(frozen=True)
class DatasetScalers:
    features: StandardizationStats
    cycle: StandardizationStats
    target_pcl: StandardizationStats

    def save(self, path: Path) -> None:
        """Save scaler arrays without pickling Python objects."""

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            feature_mean=self.features.mean,
            feature_scale=self.features.scale,
            cycle_mean=self.cycle.mean,
            cycle_scale=self.cycle.scale,
            target_pcl_mean=self.target_pcl.mean,
            target_pcl_scale=self.target_pcl.scale,
            feature_names=np.asarray(FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: Path) -> "DatasetScalers":
        with np.load(path, allow_pickle=False) as values:
            names = tuple(values["feature_names"].tolist())
            if names != FEATURE_NAMES:
                raise ValueError(
                    f"Scaler feature names {names} do not match expected {FEATURE_NAMES}"
                )
            return cls(
                features=StandardizationStats(
                    values["feature_mean"], values["feature_scale"]
                ),
                cycle=StandardizationStats(
                    values["cycle_mean"], values["cycle_scale"]
                ),
                target_pcl=StandardizationStats(
                    values["target_pcl_mean"], values["target_pcl_scale"]
                ),
            )


class BatteryDataset(Dataset[dict[str, torch.Tensor]]):
    """Cycle-level samples that retain their originating battery ID."""

    def __init__(
        self,
        *,
        features_raw: np.ndarray,
        features_normalized: np.ndarray,
        cycle_raw: np.ndarray,
        cycle_normalized: np.ndarray,
        target_pcl: np.ndarray,
        target_pcl_normalized: np.ndarray,
        battery_id: np.ndarray,
    ) -> None:
        sample_count = features_raw.shape[0]
        arrays = {
            "features_raw": features_raw,
            "features_normalized": features_normalized,
            "cycle_raw": cycle_raw,
            "cycle_normalized": cycle_normalized,
            "target_pcl": target_pcl,
            "target_pcl_normalized": target_pcl_normalized,
            "battery_id": battery_id,
        }
        for name, values in arrays.items():
            if values.shape[0] != sample_count:
                raise ValueError(
                    f"{name} has {values.shape[0]} rows; expected {sample_count}"
                )

        self.tensors = {
            "features_raw": _float_tensor(features_raw),
            "features_normalized": _float_tensor(features_normalized),
            "cycle_raw": _float_tensor(cycle_raw),
            "cycle_normalized": _float_tensor(cycle_normalized),
            "target_pcl": _float_tensor(target_pcl),
            "target_pcl_normalized": _float_tensor(target_pcl_normalized),
            "battery_id": torch.as_tensor(
                np.ascontiguousarray(battery_id), dtype=torch.int64
            ),
        }

    def __len__(self) -> int:
        return self.tensors["battery_id"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: tensor[index] for name, tensor in self.tensors.items()}

    @property
    def battery_ids(self) -> set[int]:
        return set(self.tensors["battery_id"].cpu().numpy().reshape(-1).tolist())


@dataclass(frozen=True)
class BatteryDataBundle:
    train: BatteryDataset
    validation: BatteryDataset
    test: BatteryDataset
    scalers: DatasetScalers
    audit_report: dict[str, Any]

    def assert_no_battery_leakage(self) -> None:
        split_ids = {
            "train": self.train.battery_ids,
            "validation": self.validation.battery_ids,
            "test": self.test.battery_ids,
        }
        names = tuple(split_ids)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sorted(split_ids[left] & split_ids[right])
                if overlap:
                    raise ValueError(
                        f"Battery leakage between {left} and {right}: {overlap}"
                    )


@dataclass(frozen=True)
class BatteryDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def _as_float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got {array.shape}")
    return array


def _float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)


def _record_battery_ids(num_cycles: np.ndarray) -> np.ndarray:
    boundaries = reconstruct_battery_boundaries(num_cycles)
    return np.concatenate(
        [
            np.full(boundary.num_cycles, boundary.battery_id, dtype=np.int64)
            for boundary in boundaries
        ]
    )[:, None]


def build_data_pipeline_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> BatteryDataBundle:
    """Build leakage-safe datasets from already-loaded MATLAB arrays."""

    audit_report = audit_arrays(arrays)
    features = _as_float_matrix(arrays["Features_mov_Flt"], "Features_mov_Flt")
    cycle = _as_float_matrix(
        np.asarray(arrays["Cycles_Flt"]).reshape(-1), "Cycles_Flt"
    )
    target_pcl = _as_float_matrix(
        np.asarray(arrays["PCL_Flt"]).reshape(-1), "PCL_Flt"
    )
    battery_ids = _record_battery_ids(arrays["Num_Cycles_Flt"])

    split_indices = {
        split_name: np.flatnonzero(
            np.isin(battery_ids[:, 0], np.asarray(split_battery_ids))
        )
        for split_name, split_battery_ids in audit_report["splits"].items()
    }
    train_indices = split_indices["train"]
    scalers = DatasetScalers(
        features=StandardizationStats.fit(features[train_indices], "features"),
        cycle=StandardizationStats.fit(cycle[train_indices], "cycle"),
        target_pcl=StandardizationStats.fit(
            target_pcl[train_indices], "target_pcl"
        ),
    )

    def make_dataset(indices: np.ndarray) -> BatteryDataset:
        features_split = features[indices]
        cycle_split = cycle[indices]
        target_split = target_pcl[indices]
        return BatteryDataset(
            features_raw=features_split,
            features_normalized=scalers.features.transform(
                features_split, "features"
            ),
            cycle_raw=cycle_split,
            cycle_normalized=scalers.cycle.transform(cycle_split, "cycle"),
            target_pcl=target_split,
            target_pcl_normalized=scalers.target_pcl.transform(
                target_split, "target_pcl"
            ),
            battery_id=battery_ids[indices],
        )

    bundle = BatteryDataBundle(
        train=make_dataset(split_indices["train"]),
        validation=make_dataset(split_indices["validation"]),
        test=make_dataset(split_indices["test"]),
        scalers=scalers,
        audit_report=audit_report,
    )
    bundle.assert_no_battery_leakage()
    return bundle


def load_battery_data(path: Path) -> BatteryDataBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return build_data_pipeline_from_arrays(loadmat(path))


def create_data_loaders(
    bundle: BatteryDataBundle,
    *,
    batch_size: int,
    seed: int = 1234,
    num_workers: int = 0,
) -> BatteryDataLoaders:
    """Create deterministic loaders; only the training loader is shuffled."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers}
    return BatteryDataLoaders(
        train=DataLoader(
            bundle.train, shuffle=True, generator=generator, **common
        ),
        validation=DataLoader(bundle.validation, shuffle=False, **common),
        test=DataLoader(bundle.test, shuffle=False, **common),
    )


def pipeline_summary(bundle: BatteryDataBundle) -> dict[str, Any]:
    datasets = {
        "train": bundle.train,
        "validation": bundle.validation,
        "test": bundle.test,
    }
    return {
        "split_records": {name: len(dataset) for name, dataset in datasets.items()},
        "split_batteries": {
            name: sorted(dataset.battery_ids) for name, dataset in datasets.items()
        },
        "feature_names": list(FEATURE_NAMES),
        "scalers_fitted_from": "training batteries only",
        "feature_mean": bundle.scalers.features.mean.tolist(),
        "feature_scale": bundle.scalers.features.scale.tolist(),
        "cycle_mean": bundle.scalers.cycle.mean.tolist(),
        "cycle_scale": bundle.scalers.cycle.scale.tolist(),
        "target_pcl_mean": bundle.scalers.target_pcl.mean.tolist(),
        "target_pcl_scale": bundle.scalers.target_pcl.scale.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("SeversonBattery.mat"))
    parser.add_argument(
        "--scalers-output",
        type=Path,
        default=Path("artifacts/data_scalers.npz"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/data_pipeline_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_battery_data(args.data)
    bundle.scalers.save(args.scalers_output)
    summary = pipeline_summary(bundle)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Battery-wise PyTorch data pipeline")
    for split_name in ("train", "validation", "test"):
        print(
            f"{split_name}: {summary['split_records'][split_name]:,} records / "
            f"{len(summary['split_batteries'][split_name])} batteries"
        )
    print("Scalers fitted from training batteries only")
    print(f"Scalers: {args.scalers_output.resolve()}")
    print(f"Summary: {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
