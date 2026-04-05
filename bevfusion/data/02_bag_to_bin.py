"""
02_bag_to_bin.py
从 ROS bag 提取 /point_calibration 点云，保存为模型所需的 .bin 格式，
同时在相同目录保存同名 .pcd 文件用于可视化。

输出格式：
  .bin  float32，每点 5 列 [x, y, z, intensity, time_offset]
  .pcd  binary PCD v0.7，字段 [x, y, z, intensity]（可被 CloudCompare / PCL 直接打开）

  - x, y, z     : LiDAR 坐标（米）
  - intensity    : 反射强度（原始值，归一化到 [0,1]）
  - time_offset  : 单帧内点的相对时间（秒），主帧填 0.0

用法：
  conda run -n superpoint python data/02_bag_to_bin.py
  conda run -n superpoint python data/02_bag_to_bin.py --bag /path/to/xxx.bag --out /path/to/output
"""

import argparse
import struct
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


BAG_PATH  = Path(__file__).parent / "ren2-obs.bag"
OUT_DIR   = Path(__file__).parent / "samples" / "LIDAR_TOP"
TOPIC     = "/point_calibration"

# PointCloud2 字段 offset（由 01_inspect_bag.py 确认）
# x(FLOAT32,off=0), y(FLOAT32,off=4), z(FLOAT32,off=8),
# _unk(4bytes,off=12), intensity(FLOAT32,off=16),
# ring(UINT16,off=20), _pad(2bytes,off=22), time(FLOAT64,off=24)
POINT_STEP = 32


def parse_pointcloud2(msg) -> np.ndarray:
    """
    解析 PointCloud2 原始字节，返回 (N, 5) float32 数组
    列顺序：[x, y, z, intensity, time_offset]
    """
    n_pts = msg.width * msg.height
    raw = bytes(msg.data)

    # 实际 32 字节布局:
    #   off 0  : x         (float32, 4B)
    #   off 4  : y         (float32, 4B)
    #   off 8  : z         (float32, 4B)
    #   off 12 : _unknown  (float32, 4B) ← 未知字段（可能是 normal 或 padding）
    #   off 16 : intensity (float32, 4B)
    #   off 20 : ring      (uint16,  2B)
    #   off 22 : _pad      (2B)
    #   off 24 : time      (float64, 8B)
    # 总计 32B ✓

    dtype = np.dtype([
        ('x',          np.float32),   # 4B
        ('y',          np.float32),   # 4B
        ('z',          np.float32),   # 4B
        ('_unknown',   np.float32),   # 4B
        ('intensity',  np.float32),   # 4B
        ('ring',       np.uint16),    # 2B
        ('_pad',       np.uint8, 2),  # 2B
        ('time',       np.float64),   # 8B
    ])
    # 32B total ✓

    pts_struct = np.frombuffer(raw, dtype=dtype, count=n_pts)

    # time 字段：单帧内相对时间（秒），直接使用
    # 主帧使用 per-point time（通常 0~0.1s 内）
    x   = pts_struct['x'].astype(np.float32)
    y   = pts_struct['y'].astype(np.float32)
    z   = pts_struct['z'].astype(np.float32)
    intensity = (pts_struct['intensity'] / 255.0).astype(np.float32)  # 归一化
    time_offset = pts_struct['time'].astype(np.float32)

    # 去掉无效点（NaN / Inf）
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    result = np.stack([x, y, z, intensity, time_offset], axis=1)  # (N, 5)
    return result[valid]


def save_pcd(pts: np.ndarray, pcd_path: Path):
    """
    将 (N, 5) float32 点云保存为 binary PCD v0.7 文件。
    写入字段：x, y, z, intensity（time_offset 不写入，PCD 查看器不需要）。
    """
    xyz_i = pts[:, :4].astype(np.float32)  # (N, 4): x y z intensity
    n = len(xyz_i)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(pcd_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz_i.tobytes())


def extract_bins(bag_path: Path, out_dir: Path, topic: str = TOPIC):
    out_dir.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"Bag   : {bag_path}")
    print(f"Topic : {topic}")
    print(f"OutDir: {out_dir}")

    frame_idx = 0
    saved_files = []

    with Reader(str(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"Topic '{topic}' not found in bag")

        for conn, ts, data in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(data, conn.msgtype)
            pts = parse_pointcloud2(msg)

            stem = f"{frame_idx:06d}"
            bin_path = out_dir / f"{stem}.bin"
            pcd_path = out_dir / f"{stem}.pcd"

            pts.tofile(bin_path)
            save_pcd(pts, pcd_path)

            saved_files.append({
                "path": str(bin_path),
                "timestamp_ns": int(ts),      # 纳秒
                "n_points": len(pts),
            })

            if frame_idx % 10 == 0:
                print(f"  [{frame_idx:3d}] {stem}.bin/.pcd  pts={len(pts):6d}  "
                      f"ts={ts/1e9:.3f}s")
            frame_idx += 1

    print(f"\n完成：共保存 {frame_idx} 帧（.bin + .pcd）→ {out_dir}")
    return saved_files


def verify_bin(bin_path: Path):
    """验证单个 .bin 文件"""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    print(f"\n验证 {bin_path.name}:")
    print(f"  shape     : {pts.shape}")
    print(f"  x range   : [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
    print(f"  y range   : [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
    print(f"  z range   : [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    print(f"  intensity : [{pts[:,3].min():.3f}, {pts[:,3].max():.3f}]")
    print(f"  time      : [{pts[:,4].min():.4f}, {pts[:,4].max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",   default=str(BAG_PATH))
    parser.add_argument("--out",   default=str(OUT_DIR))
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    saved = extract_bins(Path(args.bag), Path(args.out), args.topic)

    # 验证第一帧
    if saved:
        verify_bin(Path(saved[0]["path"]))

    # 保存时间戳索引（供后续脚本使用）
    import json
    ts_file = Path(args.out).parent.parent / "timestamps.json"
    with open(ts_file, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\n时间戳索引保存至: {ts_file}")
