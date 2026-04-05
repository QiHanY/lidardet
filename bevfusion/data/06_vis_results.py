"""
06_vis_results.py
将推理结果（BEV 图像）编译成视频，或展示单帧对比。

功能：
  1. 将 data/infer_results/vis/ 中的 PNG 图像编译成 MP4 视频
  2. 可选：叠加原始 .bin 点云 + pred_txt 标注框（无需重新推理）

用法：
  # 基本用法：将 PNG 序列合成视频
  conda run -n superpoint python data/06_vis_results.py

  # 指定帧率和输出路径
  conda run -n superpoint python data/06_vis_results.py --fps 10 --output data/infer_results/result.mp4

  # 从原始 .bin + pred_txt 重新生成 BEV 图（不需要 checkpoint）
  conda run -n superpoint python data/06_vis_results.py --from-raw

  # 生成 GIF 动画（文件较小，方便预览）
  conda run -n superpoint python data/06_vis_results.py --gif

注意：需要在 bevfusion/ 目录下运行
  cd CUDA-BEVFusion/bevfusion
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── 默认路径 ────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parent
VIS_DIR     = DATA_DIR / "infer_results" / "vis"
PRED_DIR    = DATA_DIR / "infer_results" / "pred_txt"
LIDAR_DIR   = DATA_DIR / "samples" / "LIDAR_TOP"
OUT_VIDEO   = DATA_DIR / "infer_results" / "result.mp4"
OUT_GIF     = DATA_DIR / "infer_results" / "result.gif"

# 点云范围（与训练 config 一致）
POINT_CLOUD_RANGE = [-100.0, -100.0, -3.0, 100.0, 100.0, 10.0]

# 类别颜色
CLASS_COLORS = {
    "vehicle":    (1.0, 0.5, 0.0),
    "car":        (1.0, 0.5, 0.0),
    "pedestrian": (0.0, 0.5, 1.0),
    "cyclist":    (0.0, 1.0, 0.5),
}
DEFAULT_COLOR = (1.0, 0.0, 0.0)


# ────────────────────────────────────────────────────────────
# 从原始数据重新生成 BEV（含检测框）
# ────────────────────────────────────────────────────────────

def load_pred_txt(txt_path: Path):
    """
    读取 pred_txt 预测文件
    格式：cx cy cz l w h yaw [score] [class]
    返回 boxes (N,7), scores (N,), classes list
    """
    boxes, scores, classes = [], [], []
    if not txt_path.exists():
        return np.zeros((0, 7)), np.zeros(0), []

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            cx, cy, cz, l, w, h, yaw = [float(x) for x in parts[:7]]
            score = float(parts[7]) if len(parts) > 7 else 1.0
            cls   = parts[8] if len(parts) > 8 else "vehicle"
            boxes.append([cx, cy, cz, l, w, h, yaw])
            scores.append(score)
            classes.append(cls)

    if not boxes:
        return np.zeros((0, 7)), np.zeros(0), []
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), classes


def draw_bev_from_raw(bin_path: Path, pred_txt_path: Path,
                      pcr: list, save_path: Path,
                      score_thr: float = 0.0):
    """从原始 .bin 和 pred_txt 生成 BEV 可视化图"""
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)
    boxes, scores, cls_names = load_pred_txt(pred_txt_path)

    x_range = (pcr[0], pcr[3])
    y_range = (pcr[1], pcr[4])
    res = 0.2

    # 点云密度图
    W = int((x_range[1] - x_range[0]) / res)
    H = int((y_range[1] - y_range[0]) / res)
    img = np.zeros((H, W), dtype=np.float32)

    x_f, y_f = pts[:, 0], pts[:, 1]
    mask = ((x_f >= x_range[0]) & (x_f <= x_range[1]) &
            (y_f >= y_range[0]) & (y_f <= y_range[1]))
    xf, yf = x_f[mask], y_f[mask]
    xi = ((xf - x_range[0]) / res).astype(int).clip(0, W - 1)
    yi = ((yf - y_range[0]) / res).astype(int).clip(0, H - 1)
    np.add.at(img, (yi, xi), 1)
    img = np.log1p(img)
    if img.max() > 0:
        img /= img.max()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1,
              extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.plot(0, 0, "c+", markersize=10, markeredgewidth=2)

    n_det = 0
    for i, (box, score, cls) in enumerate(zip(boxes, scores, cls_names)):
        if score < score_thr:
            continue
        cx, cy = float(box[0]), float(box[1])
        lx, ly = float(box[3]), float(box[4])
        yaw    = float(box[6])
        color  = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        corners = np.array([
            [ lx/2,  ly/2],
            [-lx/2,  ly/2],
            [-lx/2, -ly/2],
            [ lx/2, -ly/2],
        ])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T + np.array([cx, cy])

        polygon = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(polygon)

        front = np.array([lx/2, 0]) @ rot.T + np.array([cx, cy])
        ax.annotate("", xy=(front[0], front[1]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.text(cx, cy, f"{score:.2f}", fontsize=6, color=color,
                ha="center", va="center")
        n_det += 1

    ax.set_title(f"{bin_path.stem}   det={n_det}", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close()

    return n_det


# ────────────────────────────────────────────────────────────
# 视频合成
# ────────────────────────────────────────────────────────────

def make_video(png_files: list, output_path: Path, fps: int = 10):
    """将 PNG 列表编译成 MP4 视频"""
    if not png_files:
        print("没有找到 PNG 文件，跳过视频生成")
        return

    # 读取第一帧获取尺寸
    first = cv2.imread(str(png_files[0]))
    if first is None:
        print(f"无法读取图像：{png_files[0]}")
        return
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i, png in enumerate(png_files):
        frame = cv2.imread(str(png))
        if frame is None:
            print(f"  警告：跳过无法读取的帧 {png.name}")
            continue
        writer.write(frame)
        if (i + 1) % 20 == 0 or i == len(png_files) - 1:
            print(f"  [{i+1:3d}/{len(png_files)}] 已写入 {png.name}")

    writer.release()
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n视频已保存：{output_path}  ({size_mb:.1f} MB，{len(png_files)} 帧，{fps} FPS)")


def make_gif(png_files: list, output_path: Path, fps: int = 10):
    """将 PNG 列表编译成 GIF 动画（自动降采样以控制文件大小）"""
    try:
        import imageio
    except ImportError:
        print("imageio 未安装，跳过 GIF 生成")
        return

    if not png_files:
        print("没有找到 PNG 文件，跳过 GIF 生成")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GIF 降分辨率（避免文件过大）
    first = cv2.imread(str(png_files[0]))
    if first is None:
        print(f"无法读取图像：{png_files[0]}")
        return
    h, w = first.shape[:2]
    scale = min(1.0, 600 / max(h, w))   # 限制最大边 600px
    new_w = int(w * scale)
    new_h = int(h * scale)

    frames = []
    for i, png in enumerate(png_files):
        img = cv2.imread(str(png))
        if img is None:
            continue
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        if (i + 1) % 20 == 0 or i == len(png_files) - 1:
            print(f"  [{i+1:3d}/{len(png_files)}] 读取 {png.name}")

    duration = 1.0 / fps
    imageio.mimsave(str(output_path), frames, duration=duration, loop=0)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nGIF 已保存：{output_path}  ({size_mb:.1f} MB，{len(frames)} 帧，{fps} FPS)")


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main(args):
    # ── 可选：从原始数据重新生成 BEV 图 ───────────────────
    if args.from_raw:
        bin_files = sorted(Path(args.lidar_dir).glob("*.bin"))
        if not bin_files:
            print(f"未找到 .bin 文件：{args.lidar_dir}")
            return

        print(f"从原始数据重新生成 BEV 图（{len(bin_files)} 帧）...")
        regen_dir = Path(args.vis_dir)
        regen_dir.mkdir(parents=True, exist_ok=True)

        for i, bin_path in enumerate(bin_files):
            pred_txt = Path(args.pred_dir) / f"{bin_path.stem}.txt"
            save_path = regen_dir / f"{bin_path.stem}.png"
            n = draw_bev_from_raw(bin_path, pred_txt,
                                  POINT_CLOUD_RANGE, save_path,
                                  score_thr=args.score_thr)
            if (i + 1) % 20 == 0 or i == len(bin_files) - 1:
                print(f"  [{i+1:3d}/{len(bin_files)}] {bin_path.name}  det={n}")

        print(f"BEV 图重新生成完毕 → {regen_dir}")

    # ── 读取 PNG 列表 ──────────────────────────────────────
    vis_dir = Path(args.vis_dir)
    png_files = sorted(vis_dir.glob("*.png"))
    if not png_files:
        print(f"未找到 PNG 文件：{vis_dir}")
        print("请先运行 05_infer.py 生成推理结果，或使用 --from-raw 从原始数据生成。")
        return

    print(f"找到 {len(png_files)} 帧 PNG 图像（{vis_dir}）")

    # ── 生成视频 ───────────────────────────────────────────
    if not args.gif_only:
        output_video = Path(args.output)
        make_video(png_files, output_video, fps=args.fps)

    # ── 生成 GIF ───────────────────────────────────────────
    if args.gif or args.gif_only:
        output_gif = Path(args.output_gif)
        make_gif(png_files, output_gif, fps=args.fps)

    # ── 统计信息 ───────────────────────────────────────────
    summary_path = Path(args.vis_dir).parent / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n推理统计摘要：")
        print(f"  平均耗时 : {summary.get('avg_time_ms', '?')} ms/帧")
        print(f"  平均 FPS : {summary.get('avg_fps', '?')}")
        print(f"  平均检测 : {summary.get('avg_det', '?')} 个/帧")
        print(f"  置信度   : ≥ {summary.get('score_thr', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推理结果可视化 & 视频合成")
    parser.add_argument(
        "--vis-dir",
        default=str(VIS_DIR),
        dest="vis_dir",
        help="BEV PNG 图像目录（05_infer.py 的输出）",
    )
    parser.add_argument(
        "--pred-dir",
        default=str(PRED_DIR),
        dest="pred_dir",
        help="预测框 .txt 目录（--save-txt 模式下生成）",
    )
    parser.add_argument(
        "--lidar-dir",
        default=str(LIDAR_DIR),
        dest="lidar_dir",
        help="原始 .bin 点云目录（--from-raw 时使用）",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_VIDEO),
        help="输出视频路径（.mp4）",
    )
    parser.add_argument(
        "--output-gif",
        default=str(OUT_GIF),
        dest="output_gif",
        help="输出 GIF 路径",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="视频帧率（默认 10）",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="同时生成 GIF 动画",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        dest="gif_only",
        help="只生成 GIF，不生成 MP4",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        dest="from_raw",
        help="从原始 .bin + pred_txt 重新生成 BEV 图（无需已有 PNG）",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        dest="score_thr",
        help="--from-raw 时的置信度过滤阈值",
    )
    args = parser.parse_args()
    main(args)
