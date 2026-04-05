"""
05_infer.py
使用训练好的 checkpoint 对 data/samples/LIDAR_TOP/ 中的点云做推理，
并将检测结果可视化保存为 BEV 图像。

用法：
  # 基本用法（指定 checkpoint）
  conda run -n bevfusion_new python data/05_infer.py \
      --checkpoint work_dirs/ren2_lidar_only/latest.pth

  # 指定 config（默认使用 ren2_lidar_only.yaml）
  conda run -n bevfusion_new python data/05_infer.py \
      --checkpoint work_dirs/ren2_lidar_only/latest.pth \
      --config configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml

  # 调整置信度阈值
  conda run -n bevfusion_new python data/05_infer.py \
      --checkpoint work_dirs/ren2_lidar_only/latest.pth \
      --score-thr 0.3

  # 只推理前 N 帧
  conda run -n bevfusion_new python data/05_infer.py \
      --checkpoint work_dirs/ren2_lidar_only/latest.pth \
      --max-frames 20

注意：需要在 bevfusion/ 目录下运行
  cd CUDA-BEVFusion/bevfusion
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── bevfusion 路径 ──────────────────────────────────────────
BEVFUSION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BEVFUSION_ROOT))

from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.core.bbox import LiDARInstance3DBoxes

# ── 默认路径 ────────────────────────────────────────────────
DEFAULT_CONFIG = str(BEVFUSION_ROOT /
    "configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml")
DATA_DIR   = BEVFUSION_ROOT / "data" / "samples" / "LIDAR_TOP"
OUT_DIR    = BEVFUSION_ROOT / "data" / "infer_results"

# 类别颜色（BEV 图中检测框颜色）
CLASS_COLORS = {
    "vehicle":  (1.0, 0.5, 0.0),   # 橙色
    "car":      (1.0, 0.5, 0.0),
    "pedestrian": (0.0, 0.5, 1.0), # 蓝色
    "cyclist":  (0.0, 1.0, 0.5),   # 绿色
}
DEFAULT_COLOR = (1.0, 0.0, 0.0)    # 红色（未知类别）


# ────────────────────────────────────────────────────────────
# 数据预处理
# ────────────────────────────────────────────────────────────

def load_points(bin_path: Path, point_cloud_range: list) -> torch.Tensor:
    """读取 .bin 文件，裁剪到 point_cloud_range 内，返回 (N,5) Tensor"""
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)

    # 范围过滤
    pcr = point_cloud_range
    mask = ((pts[:, 0] >= pcr[0]) & (pts[:, 0] <= pcr[3]) &
            (pts[:, 1] >= pcr[1]) & (pts[:, 1] <= pcr[4]) &
            (pts[:, 2] >= pcr[2]) & (pts[:, 2] <= pcr[5]))
    pts = pts[mask]

    return torch.from_numpy(pts)


def load_sweeps(bin_path: Path, ts_index: dict,
                point_cloud_range: list, sweeps_num: int = 9) -> torch.Tensor:
    """
    加载历史帧并与主帧合并，历史帧点的 time 列保持原始值（负值表示过去）
    ts_index: {stem: {"timestamp_ns": ..., "path": ...}}
    """
    all_pts = [load_points(bin_path, point_cloud_range)]

    stem = bin_path.stem
    if stem in ts_index:
        ts_list = sorted(ts_index.values(), key=lambda x: x["timestamp_ns"])
        cur_ts  = ts_index[stem]["timestamp_ns"]
        cur_idx = next(i for i, v in enumerate(ts_list)
                       if v["timestamp_ns"] == cur_ts)

        for prev in ts_list[max(0, cur_idx - sweeps_num): cur_idx][::-1]:
            prev_path = Path(prev["path"])
            if not prev_path.exists():
                continue
            pts = np.fromfile(str(prev_path), dtype=np.float32).reshape(-1, 5)
            # 时间差（秒，负值）
            dt = (prev["timestamp_ns"] - cur_ts) / 1e9
            pts[:, 4] = dt
            # 范围过滤
            pcr = point_cloud_range
            mask = ((pts[:, 0] >= pcr[0]) & (pts[:, 0] <= pcr[3]) &
                    (pts[:, 1] >= pcr[1]) & (pts[:, 1] <= pcr[4]) &
                    (pts[:, 2] >= pcr[2]) & (pts[:, 2] <= pcr[5]))
            all_pts.append(torch.from_numpy(pts[mask]))

    return torch.cat(all_pts, dim=0)


def build_input(pts_tensor: torch.Tensor, device: torch.device) -> dict:
    """将点云 Tensor 封装为模型输入格式（含 dummy 相机参数）"""
    pts = pts_tensor.to(device)

    # BEVFusion forward 需要所有传感器参数，即使是 LiDAR-only
    # 传入空的相机数据
    batch_size = 1
    num_cams = 6  # nuScenes 默认 6 个相机

    return {
        "points": [pts],  # List[Tensor]，每个 Tensor 是一帧点云
        "img": torch.zeros(batch_size, num_cams, 3, 256, 704, device=device),
        "camera2ego": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        "lidar2ego": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        "lidar2camera": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        "lidar2image": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        "camera_intrinsics": torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        "camera2lidar": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        "img_aug_matrix": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        "lidar_aug_matrix": torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        "metas": [{
            "lidar2ego":        np.eye(4, dtype=np.float32),
            "lidar_aug_matrix": np.eye(4, dtype=np.float32),
            "token":            "custom",
            "timestamp":        0.0,
            "box_type_3d":      LiDARInstance3DBoxes,
        }],
    }


# ────────────────────────────────────────────────────────────
# 可视化
# ────────────────────────────────────────────────────────────

def draw_bev(pts: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
             labels: np.ndarray, class_names: list,
             point_cloud_range: list, save_path: Path,
             score_thr: float = 0.3, frame_name: str = ""):
    """
    绘制 BEV 点云 + 检测框图像并保存

    boxes: (N, 9) [cx, cy, cz, l, w, h, sin, cos, vx, vy] 或 (N, 7)
    """
    pcr = point_cloud_range
    x_range = (pcr[0], pcr[3])
    y_range = (pcr[1], pcr[4])

    # ── 点云密度图 ──────────────────────────────────────────
    res = 0.2
    W = int((x_range[1] - x_range[0]) / res)
    H = int((y_range[1] - y_range[0]) / res)
    img = np.zeros((H, W), dtype=np.float32)

    x_filt = pts[:, 0]
    y_filt = pts[:, 1]
    mask = ((x_filt >= x_range[0]) & (x_filt <= x_range[1]) &
            (y_filt >= y_range[0]) & (y_filt <= y_range[1]))
    xf, yf = x_filt[mask], y_filt[mask]
    xi = ((xf - x_range[0]) / res).astype(int).clip(0, W - 1)
    yi = ((yf - y_range[0]) / res).astype(int).clip(0, H - 1)
    np.add.at(img, (yi, xi), 1)
    img = np.log1p(img)
    if img.max() > 0:
        img = img / img.max()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1,
              extent=[x_range[0], x_range[1], y_range[0], y_range[1]])

    # ── 原点标记（传感器位置）─────────────────────────────
    ax.plot(0, 0, "c+", markersize=10, markeredgewidth=2, label="sensor")

    # ── 检测框 ─────────────────────────────────────────────
    n_det = 0
    if boxes is not None and len(boxes) > 0:
        keep = scores >= score_thr
        boxes_f  = boxes[keep]
        scores_f = scores[keep]
        labels_f = labels[keep]
        n_det    = len(boxes_f)

        for box, score, label in zip(boxes_f, scores_f, labels_f):
            cx, cy = float(box[0]), float(box[1])
            lx, ly = float(box[3]), float(box[4])     # length, width

            # yaw 角（TransFusion 输出 sin/cos，还原 yaw）
            if box.shape[0] >= 8:
                yaw = np.arctan2(float(box[6]), float(box[7]))
            else:
                yaw = float(box[6]) if box.shape[0] > 6 else 0.0

            cls_name = class_names[int(label)] if int(label) < len(class_names) \
                       else f"cls{int(label)}"
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            # 旋转矩形四个角点
            cos_a, sin_a = np.cos(yaw), np.sin(yaw)
            corners = np.array([
                [ lx/2,  ly/2],
                [-lx/2,  ly/2],
                [-lx/2, -ly/2],
                [ lx/2, -ly/2],
            ])
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rot.T + np.array([cx, cy])

            polygon = plt.Polygon(corners, fill=False,
                                  edgecolor=color, linewidth=1.5)
            ax.add_patch(polygon)

            # 朝向箭头（框前方）
            front = np.array([lx/2, 0]) @ rot.T + np.array([cx, cy])
            ax.annotate("", xy=(front[0], front[1]),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->",
                                        color=color, lw=1.2))

            # 置信度文字
            ax.text(cx, cy, f"{score:.2f}",
                    fontsize=6, color=color,
                    ha="center", va="center")

    # ── 图例和标签 ─────────────────────────────────────────
    title = f"{frame_name}   det={n_det}   thr={score_thr}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    # 图例
    legend_handles = [mpatches.Patch(color="cyan", label="sensor")]
    for cls, clr in CLASS_COLORS.items():
        if cls in class_names:
            legend_handles.append(mpatches.Patch(color=clr, label=cls))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def build_cfg(config_path: str) -> Config:
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)
    return cfg


def run_infer(args):
    # ── 加载 config ────────────────────────────────────────
    cfg = build_cfg(args.config)
    class_names    = cfg.object_classes
    pcr            = cfg.point_cloud_range

    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Classes     : {class_names}")
    print(f"PCR         : {pcr}")
    print(f"Score thr   : {args.score_thr}")

    # ── 构建并加载模型 ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()
    print("Model loaded ✓")

    # ── 点云文件列表 ───────────────────────────────────────
    lidar_dir = Path(args.lidar_dir)
    bin_files = sorted(lidar_dir.glob("*.bin"))
    if args.max_frames > 0:
        bin_files = bin_files[:args.max_frames]
    print(f"Frames      : {len(bin_files)}")

    # ── 时间戳索引（用于加载 sweeps）─────────────────────
    ts_index = {}
    ts_file  = lidar_dir.parent.parent / "timestamps.json"
    if ts_file.exists():
        with open(ts_file) as f:
            ts_list = json.load(f)
        ts_index = {Path(d["path"]).stem: d for d in ts_list}
    else:
        print("⚠  timestamps.json 未找到，不加载历史帧 sweeps")

    # ── 输出目录 ───────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 推理结果汇总 ───────────────────────────────────────
    results_summary = []
    total_time = 0.0

    # ── 逐帧推理 ───────────────────────────────────────────
    for idx, bin_path in enumerate(bin_files):
        # 加载点云（单帧 or 含历史帧）
        if args.no_sweeps or not ts_index:
            pts = load_points(bin_path, pcr)
        else:
            pts = load_sweeps(bin_path, ts_index, pcr, sweeps_num=9)
        pts_np = pts.numpy()

        # 构建模型输入
        inp = build_input(pts, device)

        # 推理
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **inp)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        total_time += elapsed

        # 解析结果
        res = result[0]
        boxes  = res["boxes_3d"].tensor.cpu().numpy()   if "boxes_3d"  in res else np.zeros((0, 9))
        scores = res["scores_3d"].cpu().numpy()         if "scores_3d" in res else np.zeros(0)
        labels = res["labels_3d"].cpu().numpy()         if "labels_3d" in res else np.zeros(0, dtype=int)

        # 过滤低置信度
        keep   = scores >= args.score_thr
        n_det  = keep.sum()

        print(f"  [{idx:3d}/{len(bin_files)}] {bin_path.name}  "
              f"pts={len(pts_np):6d}  det={n_det:3d}  {elapsed:.1f}ms")

        # 可视化
        vis_path = out_dir / "vis" / f"{bin_path.stem}.png"
        draw_bev(
            pts      = pts_np,
            boxes    = boxes,
            scores   = scores,
            labels   = labels,
            class_names = class_names,
            point_cloud_range = pcr,
            save_path = vis_path,
            score_thr = args.score_thr,
            frame_name = bin_path.stem,
        )

        # 保存检测结果到 txt（可用于后续标注参考）
        if args.save_txt and n_det > 0:
            txt_path = out_dir / "pred_txt" / f"{bin_path.stem}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                    cls = class_names[int(label)] if int(label) < len(class_names) \
                          else f"cls{int(label)}"
                    # 格式：cx cy cz l w h yaw score class
                    cx, cy, cz = box[0], box[1], box[2]
                    l,  w,  h  = box[3], box[4], box[5]
                    if box.shape[0] >= 8:
                        yaw = np.arctan2(float(box[6]), float(box[7]))
                    else:
                        yaw = float(box[6]) if box.shape[0] > 6 else 0.0
                    f.write(f"{cx:.3f} {cy:.3f} {cz:.3f} "
                            f"{l:.3f} {w:.3f} {h:.3f} "
                            f"{yaw:.4f} {score:.4f} {cls}\n")

        results_summary.append({
            "frame":   bin_path.stem,
            "n_pts":   int(len(pts_np)),
            "n_det":   int(n_det),
            "time_ms": round(elapsed, 2),
        })

    # ── 统计报告 ───────────────────────────────────────────
    avg_ms  = total_time / max(len(bin_files), 1)
    avg_det = sum(r["n_det"] for r in results_summary) / max(len(results_summary), 1)

    print(f"\n{'='*50}")
    print(f"推理完成")
    print(f"  总帧数     : {len(bin_files)}")
    print(f"  平均耗时   : {avg_ms:.1f} ms/帧  ({1000/avg_ms:.1f} FPS)")
    print(f"  平均检测数 : {avg_det:.1f} 个/帧")
    print(f"  可视化输出 : {out_dir}/vis/")
    if args.save_txt:
        print(f"  预测结果   : {out_dir}/pred_txt/  (可用作标注参考)")
    print(f"{'='*50}")

    # 保存统计 json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "avg_time_ms": round(avg_ms, 2),
            "avg_fps": round(1000 / avg_ms, 1),
            "avg_det": round(avg_det, 2),
            "score_thr": args.score_thr,
            "frames": results_summary,
        }, f, indent=2)
    print(f"  统计报告   : {summary_path}")


# ────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="模型权重路径（.pth）",
    )
    parser.add_argument(
        "--lidar-dir",
        default=str(DATA_DIR),
        dest="lidar_dir",
        help="点云 .bin 文件目录",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        dest="out_dir",
        help="推理结果输出目录",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        dest="score_thr",
        help="置信度阈值（低于此值的检测框被过滤）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        dest="max_frames",
        help="最多推理前 N 帧，0 表示全部",
    )
    parser.add_argument(
        "--no-sweeps",
        action="store_true",
        dest="no_sweeps",
        help="只推理单帧，不加载历史帧（无 pose 时使用）",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        dest="save_txt",
        help="将检测结果保存为 .txt，格式与标注文件相同（可用作标注参考）",
    )
    args = parser.parse_args()
    run_infer(args)
