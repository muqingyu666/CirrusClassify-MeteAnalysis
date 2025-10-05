# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2025-09-27 14:03:06
# @Last Modified by:   Muqy
# @Last Modified time: 2025-09-27 16:30:27

"""Analyze CloudSat radar reflectivity distributions for masked cloud types."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt


# 手动设置运行参数，按需修改
DATA_DIR = Path(r"E:\Processed_GEOPROF_data_with_masks")
OUTPUT_DIR = Path(r"E:\Processed_GEOPROF_stats")
START_DATE = "2006-06-13"  # 起始日期（含）
END_DATE = "2007-06-13"  # 结束日期（不含）
BINS = 80
MAX_SAMPLES = 1_000_000

RNG = np.random.default_rng(20250927)
MAX_SAMPLES = 1000000

# -------------------------------------------------------------------------------------------


def init_pool():
    return {
        "buffer": None,
        "seen": 0,
        "valid_total": 0,
        "raw_total": 0,
    }


def update_pool(pool, values, label, source):
    pool["raw_total"] += values.size

    flattened = values.reshape(-1)
    valid_mask = np.isfinite(flattened)
    valid_values = flattened[valid_mask]
    valid_count = valid_values.size
    pool["valid_total"] += valid_count

    print(
        "文件",
        source,
        f"{label} 有效点数:",
        valid_count,
        "总点数:",
        flattened.size,
    )

    if valid_count == 0:
        return

    if pool["buffer"] is None:
        pool["buffer"] = np.empty(MAX_SAMPLES, dtype=np.float32)

    buffer = pool["buffer"]
    seen = pool["seen"]

    if seen < MAX_SAMPLES:
        space = min(MAX_SAMPLES - seen, valid_count)
        buffer[seen : seen + space] = valid_values[:space]
        seen += space
        valid_values = valid_values[space:]
        valid_count = valid_values.size
        if valid_count == 0:
            pool["seen"] = seen
            return

    indices = np.arange(seen, seen + valid_count, dtype=np.int64)
    random_positions = np.floor(RNG.random(valid_count) * (indices + 1)).astype(
        np.int64
    )
    mask = random_positions < MAX_SAMPLES
    if mask.any():
        buffer[random_positions[mask]] = valid_values[mask]

    pool["seen"] = seen + valid_count


def collect_reflectivity_values(data_dir, start, end):
    pools = {"anvil": init_pool(), "insitu": init_pool()}

    pattern = "*_mask_split.nc"
    for nc_path in sorted(Path(data_dir).glob(pattern)):
        with xr.open_dataset(nc_path) as ds:
            sliced = ds.sel(time=slice(start, end))
            if sliced.sizes.get("time", 0) == 0:
                print("跳过", nc_path.name, "在时间范围内没有数据。")
                continue

            if "radar_reflectivity_anvil" in sliced:
                anvil_values = sliced["radar_reflectivity_anvil"].values.astype(
                    np.float32, copy=False
                )
                update_pool(pools["anvil"], anvil_values, "anvil", nc_path.name)

            if "radar_reflectivity_insitu" in sliced:
                insitu_values = sliced[
                    "radar_reflectivity_insitu"
                ].values.astype(np.float32, copy=False)
                update_pool(
                    pools["insitu"], insitu_values, "insitu", nc_path.name
                )

    results = {}
    for label, info in pools.items():
        if info["seen"] == 0:
            results[label] = np.array([], dtype=np.float32)
            continue

        if info["seen"] <= MAX_SAMPLES:
            results[label] = info["buffer"][: info["seen"]].copy()
        else:
            results[label] = info["buffer"].copy()
            print(
                f"{label} 总有效点数 {info['valid_total']:,}，随机保留 {MAX_SAMPLES:,} 个样本用于统计。"
            )

    return results


def compute_statistics(values):
    stats = []
    for label, arr in values.items():
        if arr.size == 0:
            stats.append(
                {
                    "cloud_type": label,
                    "count": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                }
            )
            continue

        stats.append(
            {
                "cloud_type": label,
                "count": int(arr.size),
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "std": float(np.nanstd(arr)),
            }
        )
    return stats


def plot_pdf(values, stats, output_dir, start, end, bins):
    # 设置全局字体为Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    for label, color in ("anvil", "tab:orange"), ("insitu", "tab:blue"):
        arr = values[label]
        if arr.size == 0:
            continue

        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"{label} (n={arr.size:,})",
            color=color,
        )

    ax.set_xlabel("Radar Reflectivity (dBZ)")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        f"Radar Reflectivity PDF\n{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    figure_name = f"reflectivity_pdf_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png"
    figure_path = output_dir / figure_name
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    stats_df = pd.DataFrame(stats)
    csv_name = f"reflectivity_stats_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    stats_df.to_csv(output_dir / csv_name, index=False)

    return figure_path


# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start = datetime.fromisoformat(START_DATE)
    end = datetime.fromisoformat(END_DATE)

    print("数据目录:", DATA_DIR)
    print("输出目录:", OUTPUT_DIR)
    print(
        "时间范围:", start.strftime("%Y-%m-%d"), "到", end.strftime("%Y-%m-%d")
    )
    print("直方图 bins:", BINS)

    values = collect_reflectivity_values(DATA_DIR, start, end)
    stats = compute_statistics(values)

    figure_path = plot_pdf(values, stats, OUTPUT_DIR, start, end, BINS)

    print("PDF 图保存至:", figure_path)
    for item in stats:
        print(
            "种类:{cloud_type:>6} | 样本:{count:>8} | 均值:{mean:7.2f} | 中位数:{median:7.2f} | 标准差:{std:7.2f}".format(
                **item
            )
        )
