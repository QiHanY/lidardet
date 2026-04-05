# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

BEVFusion 是 MIT HAN Lab 开发的多任务多传感器（相机 + LiDAR + 雷达）3D 感知框架，在统一的鸟瞰图（BEV）表示中融合多传感器数据，支持 3D 目标检测和 BEV 地图分割。发表于 ICRA 2023。

## 构建与安装

```bash
# 开发模式安装（需要 CUDA + PyTorch <= 1.10.2）
python setup.py develop

# Docker 构建
cd docker && docker build . -t bevfusion
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

`setup.py` 会编译自定义 CUDA 扩展：`bev_pool_ext`、`sparse_conv_ext`、`voxel_layer`、`iou3d_cuda`、`roiaware_pool3d_ext` 以及若干 PointNet 算子。

## 训练与评估

所有分布式运行使用 `torchpack dist-run -np <num_gpus>`。

```bash
# 训练 BEVFusion 检测（相机+激光雷达）
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth

# 训练纯相机检测
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth

# 训练纯激光雷达检测
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

# 评估检测
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox

# 评估分割
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map

# 下载预训练权重
./tools/download_pretrained.sh

# 可视化预测结果
python tools/visualize.py <config> <checkpoint> --out-dir <dir>

# 准备 nuScenes 数据
python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes
```

注意：训练完成后需单独运行 `tools/test.py` 来计算最终指标。

## 配置系统

配置文件位于 `configs/nuscenes/{det,seg}/`，采用深层 YAML 继承体系，子配置覆盖父配置。命令行覆盖使用点分路径（如 `--model.encoders.camera.backbone.init_cfg.checkpoint`）。

**继承层级示例（BEVFusion 检测）：**
```
configs/nuscenes/det/default.yaml
  └─ transfusion/default.yaml
       └─ secfpn/default.yaml
            └─ camera+lidar/default.yaml
                 └─ swint_v0p075/default.yaml
                      └─ convfuser.yaml   ← 入口
```

根配置中的关键变量：`point_cloud_range`、`voxel_size`、`image_size`、`object_classes`、`augment2d`、`augment3d`、`max_epochs`、`batch_size`。

## 架构概览

### 数据流

```
多传感器输入
├── 相机图像 (B, N_cam, 3, H, W)
├── LiDAR 点云 (List of [N_pts, 5])  ← (x,y,z,intensity,time)
└── 雷达点云 (List of [N_pts, 18])

        ↓ 传感器编码器（各模态独立）

相机分支：
  backbone (SwinTransformer/ResNet)
    → neck (GeneralizedLSSFPN)
      → vtransform → BEV (B, C_cam, H_bev, W_bev)

LiDAR 分支：
  voxelize (Voxelization/DynamicScatter)
    → SparseEncoder (SubMConv3d/SparseConv3d)
      → 稠密 BEV (B, C_lidar, H_bev, W_bev)

雷达分支（可选）：
  voxelize → SparseEncoder → BEV (B, C_radar, H_bev, W_bev)

        ↓ 融合器

ConvFuser: concat([cam_bev, lidar_bev, ...]) → Conv-BN-ReLU → BEV (B, C_fused, H, W)
AddFuser: 加权求和，训练时支持传感器随机丢弃

        ↓ 解码器

Backbone（BEV 上的 2D 卷积）→ Neck（FPN）→ 多尺度 BEV 特征

        ↓ 检测头

TransFusionHead → 3D 目标检测（热图 + Transformer 解码器）
CenterHead      → 备选检测头
SegHead         → BEV 地图分割（可选）
```

### 关键文件

| 功能 | 文件 |
|------|------|
| 主模型 | `mmdet3d/models/fusion_models/bevfusion.py` |
| 相机变换基类 | `mmdet3d/models/vtransforms/base.py` |
| LSS 变换 | `mmdet3d/models/vtransforms/lss.py` |
| 深度感知变换 | `mmdet3d/models/vtransforms/aware_bevdepth.py`, `depth_lss.py` |
| 拼接融合器 | `mmdet3d/models/fusers/conv.py` |
| 加法融合器 | `mmdet3d/models/fusers/add.py` |
| 稀疏 3D 编码器 | `mmdet3d/models/backbones/sparse_encoder.py` |
| TransFusion 头 | `mmdet3d/models/heads/bbox/transfusion.py` |
| BEV pool CUDA 算子 | `mmdet3d/ops/bev_pool/bev_pool.py`, `src/bev_pool_cuda.cu` |
| 体素化算子 | `mmdet3d/ops/voxel/` |
| 数据集 | `mmdet3d/datasets/nuscenes_dataset.py` |
| 训练入口 | `tools/train.py` |
| 推理入口 | `tools/test.py` |

### 相机到 BEV 变换（vtransforms）

`BaseTransform.create_frustum()` 在图像空间构建 3D 相机视锥（D 个深度 bin × fH × fW）。`get_geometry()` 使用标定矩阵将视锥投影到 LiDAR 坐标系。`bev_pool()` 通过自定义 CUDA 核将相机特征累积到 BEV 网格（比朴素实现快 40 倍）。

**LSSTransform**（默认）：`depthnet`（Conv2d）为每个像素预测 D-bin 深度分布，特征按深度加权后通过 CUDA `bev_pool_ext` 核池化到 BEV。

**DepthLSSTransform**：类似，但额外接受显式深度图作为输入。

**AwareBEVDepth / AwareDBEVDepth**：利用相机内参（fx, fy, cx, cy, 外参）的 squeeze-and-excitation 和 ASPP 上下文进行相机内参感知深度估计。

### 自定义 CUDA 算子（`mmdet3d/ops/`）

| 算子 | 用途 |
|----|---------|
| `bev_pool` | 将相机视锥特征求和聚合到 BEV 网格（核心 40× 加速） |
| `voxel` | 点云的硬/动态体素化 |
| `spconv` | LiDAR 编码器的稀疏 3D 卷积 |
| `iou3d` | NMS 的 3D IoU 计算 |
| `roiaware_pool3d` | 数据增强的点在框内查询 |
| `furthest_point_sample`, `ball_query` 等 | PointNet++ 算子 |

### BEVFusion 模型类

`fusion_models/bevfusion.py` 中的 `BEVFusion.forward_single()` 是核心前向传播：
1. `extract_camera_features()` → 相机 BEV
2. `extract_features()` → 激光雷达/雷达 BEV（调用 `voxelize()` 再经稀疏编码器）
3. 拼接已启用传感器的 BEV 特征
4. `fuser(features)` → 融合 BEV
5. `decoder.backbone(fused_bev)` + `decoder.neck(...)` → 多尺度 BEV
6. `self.heads` 中每个头计算预测和损失

### TransFusionHead

`mmdet3d/models/heads/bbox/transfusion.py`：通过以下步骤预测目标查询：
1. BEV 上的共享 2D 卷积
2. 热图头（GaussianFocalLoss）→ Top-K 初始候选
3. Transformer 解码器（自注意力 + 对 BEV 的交叉注意力）× N 层
4. 每层回归头：center(2D)、height、size、yaw、velocity、IoU score

### 数据流水线

典型训练流水线阶段（在配置中定义）：`LoadMultiViewImageFromFiles` → `LoadPointsFromFile` → `LoadPointsFromMultiSweeps` → `LoadRadarPointsMultiSweeps` → `LoadAnnotations3D` → `ObjectPaste` → `ImageAug3D` → `GlobalRotScaleTrans` → `RandomFlip3D` → `PointsRangeFilter` → `ObjectRangeFilter` → `ImageNormalize` → `GridMask` → `PointShuffle` → `DefaultFormatBundle3D` → `Collect3D`。

每个样本的 `metas` 字典包含标定矩阵：`camera_intrinsics [N,3,3]`、`camera2ego [N,4,4]`、`lidar2ego [4,4]`、`lidar2image [N,4,4]`，以及增强矩阵 `img_aug_matrix` 和 `lidar_aug_matrix`。

## 数据准备

参考 [mmdetection3d nuScenes 文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md)。由于坐标系差异，需使用本仓库（而非上游 mmdetection3d）重新生成 info 文件。

预期目录结构：
```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_dbinfos_train.pkl
```

## 依赖版本

Python 3.8（不支持 3.9+）、PyTorch 1.9–1.10.2、mmcv==1.4.0、mmdetection==2.20.0、torchpack、mpi4py==3.0.3、Pillow==8.4.0（版本需严格匹配）。也支持通过 `FORCE_ROCM=1` 使用 ROCm/HIP。
