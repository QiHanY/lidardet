# LiDAR-Only 训练指南

## 1. 训练命令

```bash
# 使用官方 lidar-only config
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

# 单卡调试
python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  --gpus 1

# 评估
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  work_dirs/voxelnet_0p075/latest.pth --eval bbox
```

---

## 2. 必须覆盖 Pipeline

`voxelnet_0p075.yaml` 继承自上层 config，**默认 pipeline 仍包含相机/雷达步骤**。
如果你的数据没有相机，需要新建一个 yaml 覆盖 pipeline。

新建 `configs/nuscenes/det/transfusion/secfpn/lidar/my_lidar_only.yaml`：

```yaml
# 继承官方 lidar-only config
_base_: voxelnet_0p075.yaml

train_pipeline:
  - type: LoadPointsFromFile
    coord_type: LIDAR
    load_dim: 5        # x, y, z, intensity, time
    use_dim: 5
  - type: LoadPointsFromMultiSweeps
    sweeps_num: 9      # 无历史帧则设 0
    load_dim: 5
    use_dim: 5
    pad_empty_sweeps: true
    remove_close: true
  - type: LoadAnnotations3D
    with_bbox_3d: true
    with_label_3d: true
    with_attr_label: false
  - type: ObjectPaste   # 无 dbinfos 则删除此步骤
    stop_epoch: -1
    db_sampler:
      dataset_root: ${dataset_root}
      info_path: ${dataset_root + "nuscenes_dbinfos_train.pkl"}
      rate: 1.0
      prepare:
        filter_by_difficulty: [-1]
        filter_by_min_points:
          car: 5
          truck: 5
          pedestrian: 5
      classes: ${object_classes}
      sample_groups:
        car: 2
        truck: 3
        pedestrian: 2
      points_loader:
        type: LoadPointsFromFile
        coord_type: LIDAR
        load_dim: 5
        use_dim: 5
  - type: GlobalRotScaleTrans
    resize_lim: [0.9, 1.1]
    rot_lim: [-0.78539816, 0.78539816]
    trans_lim: 0.5
    is_train: true
  - type: RandomFlip3D
  - type: PointsRangeFilter
    point_cloud_range: ${point_cloud_range}
  - type: ObjectRangeFilter
    point_cloud_range: ${point_cloud_range}
  - type: ObjectNameFilter
    classes: ${object_classes}
  - type: PointShuffle
  - type: DefaultFormatBundle3D
    classes: ${object_classes}
  - type: Collect3D
    keys:
      - points
      - gt_bboxes_3d
      - gt_labels_3d
    meta_keys:
      - lidar2ego
      - lidar_aug_matrix

test_pipeline:
  - type: LoadPointsFromFile
    coord_type: LIDAR
    load_dim: 5
    use_dim: 5
  - type: LoadPointsFromMultiSweeps
    sweeps_num: 9
    load_dim: 5
    use_dim: 5
    pad_empty_sweeps: true
    remove_close: true
  - type: GlobalRotScaleTrans
    resize_lim: [1.0, 1.0]
    rot_lim: [0.0, 0.0]
    trans_lim: 0.0
    is_train: false
  - type: PointsRangeFilter
    point_cloud_range: ${point_cloud_range}
  - type: DefaultFormatBundle3D
    classes: ${object_classes}
  - type: Collect3D
    keys:
      - points
      - gt_bboxes_3d
      - gt_labels_3d
    meta_keys:
      - lidar2ego
      - lidar_aug_matrix
```

---

## 3. 数据格式

### 3.1 目录结构

```
data/nuscenes/
├── samples/
│   └── LIDAR_TOP/
│       └── *.bin          # 点云文件
├── sweeps/
│   └── LIDAR_TOP/
│       └── *.bin          # 历史帧点云（可为空）
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_dbinfos_train.pkl   # GT-Paste 用，可选
```

### 3.2 点云文件格式（.bin）

float32 二进制，每点 5 个值：

```
[x, y, z, intensity, time_offset]
```

- `x, y, z`：LiDAR 坐标系下的坐标（米）
- `intensity`：反射强度，归一化到 [0, 1]
- `time_offset`：相对当前帧的时间差（秒），**单帧点云填 0.0**

Python 读写示例：

```python
import numpy as np

# 写入
points = np.array([[x, y, z, intensity, 0.0], ...], dtype=np.float32)
points.tofile("frame_000.bin")

# 读取验证
pts = np.fromfile("frame_000.bin", dtype=np.float32).reshape(-1, 5)
print(pts.shape)  # (N, 5)
```

### 3.3 标注文件格式（.pkl）

pkl 文件结构：

```python
{
    "infos": [sample_info, ...],   # 按 timestamp 排序
    "metadata": {"version": "v1.0-trainval"}
}
```

每个 `sample_info` 必须包含的字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `token` | str | 唯一帧 ID |
| `lidar_path` | str | 点云文件路径 |
| `timestamp` | float | 时间戳（微秒） |
| `sweeps` | list[dict] | 历史帧列表（无历史帧填 `[]`） |
| `ego2global_rotation` | list[4] | 车体到全局的四元数 `[w,x,y,z]` |
| `ego2global_translation` | list[3] | 车体到全局的平移 |
| `lidar2ego_rotation` | list[4] | LiDAR 到车体的四元数 |
| `lidar2ego_translation` | list[3] | LiDAR 到车体的平移 |
| `gt_boxes` | ndarray [N, 7] | `[cx, cy, cz, l, w, h, yaw]`，LiDAR 坐标系 |
| `gt_names` | ndarray [N] | 类名字符串数组 |
| `gt_velocity` | ndarray [N, 2] | `[vx, vy]`，无速度填 `np.nan` |
| `num_lidar_pts` | ndarray [N] | 每个框内点数（用于过滤空框） |

> `cams` 和 `radars` 字段在 LiDAR-only 模式下**不需要**。

### 3.4 sweeps 字段格式

每个历史帧 dict：

```python
{
    "data_path": "sweeps/LIDAR_TOP/xxx.bin",
    "timestamp": 1234567890.0,
    "ego2global_rotation": [w, x, y, z],
    "ego2global_translation": [tx, ty, tz],
    "sensor2ego_rotation": [w, x, y, z],
    "sensor2ego_translation": [tx, ty, tz],
}
```

### 3.5 生成 pkl 的脚本模板

```python
import pickle
import numpy as np
from pathlib import Path

def create_info(lidar_bin_path, gt_boxes, gt_names, gt_velocity=None):
    """
    gt_boxes: np.ndarray [N, 7]  [cx, cy, cz, l, w, h, yaw]
    gt_names: list of str
    """
    N = len(gt_names)
    if gt_velocity is None:
        gt_velocity = np.full((N, 2), np.nan, dtype=np.float32)

    return {
        "token": Path(lidar_bin_path).stem,
        "lidar_path": lidar_bin_path,
        "timestamp": 0.0,
        "sweeps": [],
        # 无外参时用单位矩阵对应的四元数和零平移
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation": [0.0, 0.0, 0.0],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "gt_boxes": gt_boxes.astype(np.float32),
        "gt_names": np.array(gt_names),
        "gt_velocity": gt_velocity.astype(np.float32),
        "num_lidar_pts": np.ones(N, dtype=np.int32) * 10,  # 或实际统计
    }

# 构建并保存
infos = [create_info(...), ...]
data = {"infos": infos, "metadata": {"version": "custom-v1.0"}}

with open("data/nuscenes/nuscenes_infos_train.pkl", "wb") as f:
    pickle.dump(data, f)
```

---

## 4. 自定义类别

修改 config 中的 `object_classes`：

```yaml
object_classes:
  - car
  - pedestrian
  - cyclist
  # 改成你自己的类别
```

同时修改 `TransFusionHead` 的 `num_classes`：

```yaml
model:
  heads:
    object:
      num_classes: 3   # 与 object_classes 数量一致
```

---

## 5. 调整点云范围和体素大小

根据你的场景修改：

```yaml
point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # [x_min,y_min,z_min,x_max,y_max,z_max]
voxel_size: [0.1, 0.1, 0.2]   # 越小精度越高，显存占用越大

model:
  encoders:
    lidar:
      backbone:
        # sparse_shape = (range / voxel_size).astype(int)
        # x: (51.2*2)/0.1 = 1024, y: 1024, z: (8.0)/0.2 = 40 → 取 41
        sparse_shape: [1024, 1024, 41]
```

> `sparse_shape` 计算公式：`ceil((range_max - range_min) / voxel_size) + 1`

---

## 6. 常见问题

**Q: 报错 `KeyError: 'img'`**
A: `Collect3D` 的 `keys` 里有 `img`，删掉即可。

**Q: 报错 `LoadMultiViewImageFromFiles: file not found`**
A: pipeline 里还有相机加载步骤，按第 2 节覆盖 pipeline。

**Q: 没有历史帧数据**
A: `sweeps` 填 `[]`，`LoadPointsFromMultiSweeps` 设 `sweeps_num: 0` 或保持默认（`pad_empty_sweeps: true` 会自动用当前帧填充）。

**Q: 没有速度标注**
A: `gt_velocity` 填 `np.full((N, 2), np.nan)`，代码会自动将 nan 替换为 0。
