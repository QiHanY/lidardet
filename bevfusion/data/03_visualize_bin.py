"""
03_visualize_bin.py
可视化 .bin 点云文件（俯视图 BEV 图像）

无需 Open3D，仅依赖 numpy + matplotlib

用法：
  conda run -n superpoint python data/03_visualize_bin.py
  conda run -n superpoint python data/03_visualize_bin.py --bin data/samples/LIDAR_TOP/000000.bin
  conda run -n superpoint python data/03_visualize_bin.py --all   # 可视化所有帧并保存图片
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无显示器环境
import matplotlib.pyplot as plt
from pathlib import Path


LIDAR_TOP_DIR = Path(__file__).parent / "samples" / "LIDAR_TOP"
VIS_DIR       = Path(__file__).parent / "vis"

# 可视化范围（米）—— 与训练 point_cloud_range 一致
X_RANGE = (-100, 100)
Y_RANGE = (-100, 100)
Z_RANGE = (-3, 10)


def bev_image(pts: np.ndarray, resolution: float = 0.1,
              x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE) -> np.ndarray:
    """生成俯视图（BEV）密度图，返回 HxW uint8 图像"""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    mask = ((x >= x_range[0]) & (x <= x_range[1]) &
            (y >= y_range[0]) & (y <= y_range[1]) &
            (z >= z_range[0]) & (z <= z_range[1]))
    x, y = x[mask], y[mask]

    W = int((x_range[1] - x_range[0]) / resolution)
    H = int((y_range[1] - y_range[0]) / resolution)

    xi = ((x - x_range[0]) / resolution).astype(int).clip(0, W - 1)
    yi = ((y - y_range[0]) / resolution).astype(int).clip(0, H - 1)

    img = np.zeros((H, W), dtype=np.float32)
    np.add.at(img, (yi, xi), 1)
    img = np.log1p(img)
    img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
    return img


def visualize_single(bin_path: Path, save_path: Path = None, show: bool = False):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    img = bev_image(pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(bin_path.name, fontsize=12)

    # BEV 密度图
    axes[0].imshow(img, cmap='hot', origin='lower')
    axes[0].set_title(f'BEV (x:{X_RANGE}, y:{Y_RANGE})')
    axes[0].set_xlabel('X →')
    axes[0].set_ylabel('Y →')

    # 高度-密度直方图
    axes[1].hist(pts[:, 2], bins=60, color='steelblue', edgecolor='none')
    axes[1].set_title(f'Z distribution  n={len(pts)}')
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('count')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=0.8, label='z=0')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_all(lidar_dir: Path, vis_dir: Path):
    bins = sorted(lidar_dir.glob("*.bin"))
    print(f"可视化 {len(bins)} 帧 → {vis_dir}")
    for b in bins:
        visualize_single(b, save_path=vis_dir / (b.stem + ".png"))
    print("完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin",  default=None, help="单个 .bin 文件路径")
    parser.add_argument("--dir",  default=str(LIDAR_TOP_DIR))
    parser.add_argument("--out",  default=str(VIS_DIR))
    parser.add_argument("--all",  action="store_true", help="可视化所有帧")
    args = parser.parse_args()

    if args.all:
        visualize_all(Path(args.dir), Path(args.out))
    else:
        bin_path = Path(args.bin) if args.bin else \
                   sorted(Path(args.dir).glob("*.bin"))[0]
        visualize_single(bin_path, save_path=Path(args.out) / (bin_path.stem + ".png"))
        print(f"\n点云统计:")
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
        for i, name in enumerate(['x','y','z','intensity','time']):
            print(f"  {name}: min={pts[:,i].min():.3f}  max={pts[:,i].max():.3f}  "
                  f"mean={pts[:,i].mean():.3f}")
