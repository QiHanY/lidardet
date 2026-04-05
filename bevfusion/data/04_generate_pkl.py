"""
04_generate_pkl.py
生成训练所需的 nuscenes_infos_train.pkl 和 nuscenes_infos_val.pkl

- 无标注版：gt_boxes 全空，用于推理/可视化调试
- 有标注版：读取 annos/ 目录下的 .txt 标注文件

标注文件格式（每行一个框）：
  cx cy cz length width height yaw [class_name]
  示例：  1.5 -2.3 0.2 4.5 1.8 1.6 0.78 vehicle

用法：
  # 无标注（推理模式）
  conda run -n superpoint python data/04_generate_pkl.py

  # 有标注（训练模式）
  conda run -n superpoint python data/04_generate_pkl.py --anno_dir data/annos

  # 自定义划分比例
  conda run -n superpoint python data/04_generate_pkl.py --anno_dir data/annos --val_ratio 0.2
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path


DATA_DIR      = Path(__file__).parent
LIDAR_TOP_DIR = DATA_DIR / "samples" / "LIDAR_TOP"
SWEEPS_DIR    = DATA_DIR / "sweeps"  / "LIDAR_TOP"
TS_FILE       = DATA_DIR / "timestamps.json"
ANNO_DIR      = DATA_DIR / "annos"
OUT_DIR       = DATA_DIR

CLASS_NAME    = "vehicle"   # 改成你的类别，多类别时在标注文件第 8 列写类名
VAL_RATIO     = 0.2
SWEEPS_NUM    = 9           # 历史帧数量（与训练 config 的 sweeps_num 一致）


def load_annotation(anno_path: Path, default_class: str) -> tuple:
    """读取单帧标注，返回 (boxes np[N,7], names list[str])"""
    boxes, names = [], []
    if not anno_path.exists():
        return np.zeros((0, 7), dtype=np.float32), []

    for line in anno_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = list(map(float, parts[:7]))   # cx cy cz l w h yaw
        cls  = parts[7] if len(parts) > 7 else default_class
        boxes.append(vals)
        names.append(cls)

    return np.array(boxes, dtype=np.float32).reshape(-1, 7), names


def build_info(token: str, lidar_path: Path,
               timestamp_ns: int, sweep_paths: list,
               sweep_timestamps_ns: list,
               boxes: np.ndarray, names: list) -> dict:
    """构造单帧 info 字典"""
    N = len(names)

    # sweeps 列表（历史帧，时间从近到远）
    sweeps = []
    for sp, sts in zip(sweep_paths, sweep_timestamps_ns):
        dt = (timestamp_ns - sts) / 1e9   # 秒，正数
        sweeps.append({
            "data_path":                str(sp),
            "timestamp":                sts / 1e3,   # 微秒（loading.py 里除以 1e6 得秒）
            "ego2global_rotation":      [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation":   [0.0, 0.0, 0.0],
            # loading.py 使用 sensor2lidar_rotation/translation 做点云对齐
            # 无外参时雷达坐标系 == LiDAR 坐标系，使用单位矩阵和零平移
            "sensor2lidar_rotation":    np.eye(3, dtype=np.float32),
            "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
            "time_diff":                dt,
        })

    return {
        "token":                   token,
        "lidar_path":              str(lidar_path),
        "timestamp":               timestamp_ns / 1e3,   # 微秒（loading.py 里除以 1e6 得秒）
        "sweeps":                  sweeps,
        "location":                "custom",        # 自定义数据集无地理位置
        # 无外参时使用单位姿态
        "ego2global_rotation":     [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation":  [0.0, 0.0, 0.0],
        "lidar2ego_rotation":      [1.0, 0.0, 0.0, 0.0],
        "lidar2ego_translation":   [0.0, 0.0, 0.0],
        # 标注信息
        "gt_boxes":      boxes,
        "gt_names":      np.array(names) if names else np.array([], dtype=str),
        "gt_velocity":   np.full((N, 2), np.nan, dtype=np.float32),
        "num_lidar_pts": np.ones(N, dtype=np.int32) * 10,
        "valid_flag":    np.ones(N, dtype=bool),    # 所有框均有效
    }


def generate_pkls(lidar_dir: Path, sweeps_dir: Path, ts_file: Path,
                  anno_dir: Path, out_dir: Path,
                  class_name: str, val_ratio: float, sweeps_num: int):

    # ── 加载时间戳索引 ──────────────────────────────────────
    if ts_file.exists():
        with open(ts_file) as f:
            ts_data = json.load(f)
        all_bins  = [Path(d["path"]) for d in ts_data]
        all_ts_ns = [d["timestamp_ns"] for d in ts_data]
    else:
        # 无时间戳文件时按文件名排序，时间戳填 0
        all_bins  = sorted(lidar_dir.glob("*.bin"))
        all_ts_ns = list(range(len(all_bins)))
        print("⚠️  未找到 timestamps.json，时间戳填顺序索引")

    print(f"总帧数: {len(all_bins)}")
    has_anno = anno_dir.exists()
    print(f"标注目录: {'✓ ' + str(anno_dir) if has_anno else '✗ (无标注，推理模式)'}")

    # ── 构建 info 列表 ─────────────────────────────────────
    infos = []
    for i, (bin_path, ts_ns) in enumerate(zip(all_bins, all_ts_ns)):
        token = bin_path.stem

        # 获取历史帧（当前帧之前的 sweeps_num 帧）
        sweep_paths  = all_bins[max(0, i - sweeps_num):i][::-1]   # 从近到远
        sweep_ts_ns  = all_ts_ns[max(0, i - sweeps_num):i][::-1]

        # 标注
        anno_path = anno_dir / f"{token}.txt"
        boxes, names = load_annotation(anno_path, class_name) \
                       if has_anno else (np.zeros((0,7),dtype=np.float32), [])

        info = build_info(token, bin_path, ts_ns,
                          sweep_paths, sweep_ts_ns,
                          boxes, names)
        infos.append(info)

        if i % 20 == 0:
            print(f"  [{i:3d}/{len(all_bins)}] {token}  "
                  f"boxes={len(names)}  sweeps={len(sweep_paths)}")

    # ── 划分训练/验证集 ────────────────────────────────────
    split_idx = int(len(infos) * (1 - val_ratio))
    train_infos = infos[:split_idx]
    val_infos   = infos[split_idx:]

    # ── 保存 pkl ───────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", train_infos), ("val", val_infos)]:
        out_path = out_dir / f"nuscenes_infos_{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"infos": subset,
                         "metadata": {"version": "custom-v1.0"}}, f)
        print(f"\n✓ 保存 {name}: {len(subset)} 帧 → {out_path}")

    # ── 统计报告 ───────────────────────────────────────────
    total_boxes = sum(len(info["gt_names"]) for info in infos)
    print(f"\n[统计]")
    print(f"  训练集: {len(train_infos)} 帧")
    print(f"  验证集: {len(val_infos)} 帧")
    print(f"  标注框总数: {total_boxes}")
    if total_boxes > 0:
        avg = total_boxes / len(infos)
        print(f"  平均每帧框数: {avg:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar_dir",  default=str(LIDAR_TOP_DIR))
    parser.add_argument("--sweeps_dir", default=str(SWEEPS_DIR))
    parser.add_argument("--ts_file",    default=str(TS_FILE))
    parser.add_argument("--anno_dir",   default=str(ANNO_DIR))
    parser.add_argument("--out_dir",    default=str(OUT_DIR))
    parser.add_argument("--class_name", default=CLASS_NAME)
    parser.add_argument("--val_ratio",  type=float, default=VAL_RATIO)
    parser.add_argument("--sweeps_num", type=int,   default=SWEEPS_NUM)
    args = parser.parse_args()

    generate_pkls(
        lidar_dir  = Path(args.lidar_dir),
        sweeps_dir = Path(args.sweeps_dir),
        ts_file    = Path(args.ts_file),
        anno_dir   = Path(args.anno_dir),
        out_dir    = Path(args.out_dir),
        class_name = args.class_name,
        val_ratio  = args.val_ratio,
        sweeps_num = args.sweeps_num,
    )
