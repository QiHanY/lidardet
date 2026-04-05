# 数据生成说明

本目录记录从 ROS bag 到训练数据的完整生成流程，基于 `ren2-obs.bag` 实测结果。

---

## 目录结构

```
data/
├── ren2-obs.bag               ← 已建索引的 ROS bag（原始文件）
├── ren2-obs.orig.bag          ← 建索引前的原始备份
├── timestamps.json            ← 帧时间戳索引（02 脚本生成）
├── samples/LIDAR_TOP/
│   ├── 000000.bin             ← 主帧点云（float32，5列）
│   ├── 000001.bin
│   └── ...（共 176 帧）
├── annos/                     ← 放置标注文件（手动创建）
│   └── 000000.txt
├── nuscenes_infos_train.pkl   ← 训练集（140 帧，04 脚本生成）
├── nuscenes_infos_val.pkl     ← 验证集（36 帧，04 脚本生成）
├── vis/                       ← BEV 可视化图像（03 脚本生成）
├── 01_inspect_bag.py
├── 02_bag_to_bin.py
├── 03_visualize_bin.py
└── 04_generate_pkl.py
```

---

## Bag 文件说明

| 项目 | 值 |
|------|-----|
| 文件 | `ren2-obs.bag` |
| 时长 | 17.5 秒 |
| 频率 | 10.1 Hz |
| 总帧数 | 176 帧 |
| LiDAR topic | `/point_calibration` |
| 点数/帧 | ~26,000 点 |

**PointCloud2 字段布局（point_step = 32 字节）：**

| 字段 | 类型 | offset | 说明 |
|------|------|--------|------|
| x | float32 | 0 | LiDAR X 坐标（米） |
| y | float32 | 4 | LiDAR Y 坐标（米） |
| z | float32 | 8 | LiDAR Z 坐标（米） |
| _unknown | float32 | 12 | 未知字段（跳过） |
| intensity | float32 | 16 | 反射强度（归一化到 [0,1]） |
| ring | uint16 | 20 | 线束编号 |
| _pad | 2B | 22 | 对齐填充 |
| time | float64 | 24 | 单帧内点时间偏移（秒，0~0.1） |

**点云实际范围（全量统计）：**

| 轴 | 全局范围 | p1~p99 | 推荐训练范围 |
|----|---------|--------|-------------|
| x | [-30.8, 197.4] | [-0.2, 87.5] | [-100, 100] |
| y | [-159.0, 170.3] | [-41.7, 9.2] | [-100, 100] |
| z | [-7.8, 24.1] | [-0.2, 8.0] | [-3, 10] |

---

## 环境准备

> 所有脚本使用 **conda 环境 `superpoint`**，不修改系统环境。

```bash
# 首次使用需安装依赖（仅一次）
conda run -n superpoint pip install rosbags pycryptodome

# bag 文件需要先建索引（仅一次）
cd CUDA-BEVFusion/bevfusion/data
rosbag reindex ren2-obs.bag
# 执行后原始文件备份为 ren2-obs.orig.bag，ren2-obs.bag 变为已索引版本
```

---

## 脚本使用

所有脚本从 `bevfusion/` 目录下运行：

```bash
cd CUDA-BEVFusion/bevfusion
```

### Step 1：查看 bag 信息

```bash
conda run -n superpoint python data/01_inspect_bag.py
# 可指定其他 bag：
conda run -n superpoint python data/01_inspect_bag.py --bag /path/to/other.bag
```

输出 topic 列表、PointCloud2 字段结构、帧数和时间范围。

---

### Step 2：提取点云 → .bin 文件

```bash
conda run -n superpoint python data/02_bag_to_bin.py
```

**输出格式**：`float32`，每点 5 列 `[x, y, z, intensity, time_offset]`，写入 `data/samples/LIDAR_TOP/`。

同时生成 `data/timestamps.json`，记录每帧路径和纳秒时间戳，供 `04_generate_pkl.py` 使用。

可选参数：
```bash
conda run -n superpoint python data/02_bag_to_bin.py \
  --bag  data/ren2-obs.bag \
  --out  data/samples/LIDAR_TOP \
  --topic /point_calibration
```

---

### Step 3：可视化验证

```bash
# 可视化第一帧（保存为 data/vis/000000.png）
conda run -n superpoint python data/03_visualize_bin.py

# 指定某帧
conda run -n superpoint python data/03_visualize_bin.py --bin data/samples/LIDAR_TOP/000010.bin

# 可视化所有帧
conda run -n superpoint python data/03_visualize_bin.py --all
```

输出为 BEV 俯视密度图（热力图） + Z 轴高度分布直方图。

---

### Step 4：生成训练 pkl

**无标注（推理/调试模式）：**

```bash
conda run -n superpoint python data/04_generate_pkl.py
```

**有标注（训练模式）：**

先将标注文件放入 `data/annos/`，每帧一个 `.txt`，文件名与 `.bin` 一致：

```
# 标注格式：每行一个 3D 框
# cx  cy  cz  length  width  height  yaw  [class_name]
 5.2  -3.1  0.3  4.5  1.8  1.6  0.78  vehicle
12.0  -1.5  0.2  4.2  1.7  1.5  -1.57
```

- `cx, cy, cz`：框中心坐标（LiDAR 坐标系，米）
- `length, width, height`：框尺寸（米）
- `yaw`：绕 Z 轴旋转角（弧度）
- `class_name`：可选，默认使用脚本中的 `CLASS_NAME`

```bash
conda run -n superpoint python data/04_generate_pkl.py --anno_dir data/annos
```

可选参数：
```bash
conda run -n superpoint python data/04_generate_pkl.py \
  --anno_dir   data/annos \
  --val_ratio  0.2 \         # 验证集比例
  --sweeps_num 9 \           # 历史帧数量（与训练 config 一致）
  --class_name vehicle       # 默认类别名
```

输出：
- `data/nuscenes_infos_train.pkl`（140 帧）
- `data/nuscenes_infos_val.pkl`（36 帧）

---

## 训练配置

对应的训练 config 已根据实际数据生成：

```
configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml
```

关键参数（基于实际点云分析）：

```yaml
point_cloud_range: [-100.0, -100.0, -3.0, 100.0, 100.0, 10.0]
voxel_size: [0.1, 0.1, 0.2]

model:
  encoders:
    lidar:
      backbone:
        sparse_shape: [2000, 2000, 66]   # ceil(200/0.1), ceil(200/0.1), ceil(13/0.2)+1
  heads:
    object:
      num_classes: 1    # 改为实际类别数
```

> `sparse_shape` 计算公式：`[ceil(Δx/vs_x), ceil(Δy/vs_y), ceil(Δz/vs_z)+1]`

### Step 5：推理当前数据

训练完成后，使用 checkpoint 对 `data/samples/LIDAR_TOP/` 中所有帧做推理：

```bash
cd CUDA-BEVFusion/bevfusion

conda run -n bevfusion_new python data/05_infer.py \
    --checkpoint work_dirs/ren2_lidar_only/latest.pth

# 调整置信度阈值（默认 0.3）
conda run -n bevfusion_new python data/05_infer.py \
    --checkpoint work_dirs/ren2_lidar_only/latest.pth \
    --score-thr 0.5

# 只推理前 20 帧（快速验证）
conda run -n bevfusion_new python data/05_infer.py \
    --checkpoint work_dirs/ren2_lidar_only/latest.pth \
    --max-frames 20

# 同时将预测框保存为 .txt（格式与标注文件相同，可用作标注参考）
conda run -n bevfusion_new python data/05_infer.py \
    --checkpoint work_dirs/ren2_lidar_only/latest.pth \
    --save-txt
```

**输出：**

```
data/infer_results/
├── vis/
│   ├── 000000.png    ← BEV 点云 + 检测框可视化（每帧一张）
│   ├── 000001.png
│   └── ...
├── pred_txt/         ← 仅 --save-txt 时生成
│   ├── 000000.txt    ← 格式：cx cy cz l w h yaw score class
│   └── ...
└── summary.json      ← 推理统计（平均耗时、FPS、检测数）
```

**pred_txt 格式**（与标注格式相同，可直接作为 `04_generate_pkl.py` 的输入）：
```
5.23 -2.10 0.31 4.52 1.81 1.62 0.7854 0.85 vehicle
```

---

### Step 6：可视化推理结果（合成视频）

```bash
cd CUDA-BEVFusion/bevfusion

# 将推理结果 PNG 合成 MP4 视频（需先运行 05_infer.py）
conda run -n superpoint python data/06_vis_results.py

# 指定帧率
conda run -n superpoint python data/06_vis_results.py --fps 10

# 同时生成 GIF 动画（方便预览）
conda run -n superpoint python data/06_vis_results.py --gif

# 从原始 .bin + pred_txt 重新生成 BEV 图（不需要已有 PNG）
conda run -n superpoint python data/06_vis_results.py --from-raw
```

**输出：**

```
data/infer_results/
├── result.mp4     ← MP4 视频（所有帧 BEV 连续播放）
└── result.gif     ← GIF 动画（--gif 时生成，最大边限制 600px）
```

---

```bash
cd CUDA-BEVFusion/bevfusion

# 单卡调试（验证流程跑通）
python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml \
  --gpus 1

# 多卡训练
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml

# 评估
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/ren2_lidar_only.yaml \
  work_dirs/ren2_lidar_only/latest.pth --eval bbox
```

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `rosbags.rosbag1.reader.ReaderError: Could not read uint8 field 'op'` | bag 未建索引 | 运行 `rosbag reindex ren2-obs.bag` |
| `ModuleNotFoundError: No module named 'Cryptodome'` | 系统 rosbag 依赖缺失 | `conda run -n superpoint pip install pycryptodome` |
| pkl 中 boxes 全为空 | 未找到标注文件 | 确认 `annos/` 目录存在且文件名与 `.bin` 匹配 |
| 训练时 `KeyError: 'img'` | pipeline 含相机步骤 | 确认使用 `ren2_lidar_only.yaml`，不是默认 config |
| 训练时 OOM | batch_size 过大 | config 中设 `data.samples_per_gpu: 1` |
