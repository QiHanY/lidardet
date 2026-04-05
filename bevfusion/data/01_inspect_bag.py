"""
01_inspect_bag.py
查看 bag 文件的 topic 列表、字段结构和帧统计信息

conda run -n superpoint python data/01_inspect_bag.py
"""

import sys
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

DATATYPE_MAP = {1:'INT8',2:'UINT8',3:'INT16',4:'UINT16',
                5:'INT32',6:'UINT32',7:'FLOAT32',8:'FLOAT64'}

BAG_PATH = Path(__file__).parent / "ren2-obs.bag"


def inspect_bag(bag_path: Path):
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"\n{'='*60}")
    print(f"Bag: {bag_path}")
    print(f"{'='*60}")

    with Reader(str(bag_path)) as reader:
        # ── 1. Topic 列表 ──────────────────────────────────────
        print("\n[Topics]")
        pc_topics = []
        for conn in reader.connections:
            print(f"  {conn.topic:<35s}  {conn.msgtype}")
            if 'PointCloud2' in conn.msgtype:
                pc_topics.append(conn.topic)

        # ── 2. PointCloud2 字段详情 ────────────────────────────
        print(f"\n[PointCloud2 Topics] found: {pc_topics}")
        for topic in pc_topics:
            conns = [c for c in reader.connections if c.topic == topic]
            count = 0
            for conn, ts, data in reader.messages(connections=conns):
                msg = typestore.deserialize_ros1(data, conn.msgtype)
                if count == 0:
                    print(f"\n  Topic: {topic}")
                    print(f"    height={msg.height}, width={msg.width}")
                    print(f"    point_step={msg.point_step} bytes")
                    print(f"    total_points={msg.height * msg.width}")
                    print(f"    fields:")
                    for f in msg.fields:
                        dtype = DATATYPE_MAP.get(f.datatype, f'unknown({f.datatype})')
                        print(f"      {f.name:<15s} offset={f.offset:<4d} dtype={dtype}")
                count += 1
            print(f"    total_frames={count}")

        # ── 3. 时间范围 ────────────────────────────────────────
        conns = [c for c in reader.connections
                 if c.topic == '/point_calibration']
        timestamps = []
        for conn, ts, data in reader.messages(connections=conns):
            timestamps.append(ts)
        if timestamps:
            duration = (timestamps[-1] - timestamps[0]) / 1e9
            freq = len(timestamps) / duration if duration > 0 else 0
            print(f"\n[/point_calibration timing]")
            print(f"  frames : {len(timestamps)}")
            print(f"  start  : {timestamps[0] / 1e9:.3f} s")
            print(f"  end    : {timestamps[-1] / 1e9:.3f} s")
            print(f"  duration: {duration:.2f} s")
            print(f"  freq   : {freq:.1f} Hz")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BAG_PATH
    inspect_bag(path)
