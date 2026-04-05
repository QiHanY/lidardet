# BEVFusion 数据流详解

> 详细描述从输入到输出的完整数据流，精确到函数级别。

---

## 1. 整体数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     输入数据 (每帧)                                   │
│  img: [B, N=6, 3, H=256, W=704]  (6路相机图像)                      │
│  points: List([N_pts, 5])         (LiDAR点云: x,y,z,intensity,time)  │
│  radar:  List([N_pts, 18])        (雷达点云, 可选)                   │
│  metas: List[dict]                (标定矩阵、增广矩阵等)              │
└──────────────────┬──────────────────────┬───────────────────────────┘
                   │                      │
         ┌─────────▼────────┐   ┌─────────▼────────┐
         │   相机分支        │   │   LiDAR分支       │
         │  Camera Branch   │   │  LiDAR Branch     │
         └─────────┬────────┘   └─────────┬────────┘
                   │                      │
         ┌─────────▼────────┐   ┌─────────▼────────┐
         │ BEV [B,C,H,W]    │   │ BEV [B,C,H,W]    │
         └─────────┬────────┘   └─────────┬────────┘
                   └──────────┬───────────┘
                         ┌────▼────┐
                         │  Fuser  │
                         └────┬────┘
                         ┌────▼────┐
                         │ Decoder │  backbone + neck
                         └────┬────┘
                    ┌─────────┴──────────┐
               ┌────▼────┐         ┌─────▼────┐
               │ Det Head│         │ Seg Head │
               └────┬────┘         └─────┬────┘
               3D BBoxes             BEV Masks
```

---

## 2. 入口：BEVFusion.forward()

**文件:** `mmdet3d/models/fusion_models/bevfusion.py`

```python
class BEVFusion(Base3DFusionModel):

    def forward(self, return_loss=True, **data):
        # 分发到 train 或 test 模式
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self, ...):
        # 调用 forward_single 得到 losses dict
        losses = self.forward_single(...)
        return losses

    def forward_single(
        self,
        img,              # [B, N, 3, H, W]
        points,           # List of [N_pts, 5] per sample
        radar,            # List of [N_pts, 18] per sample (optional)
        metas,            # List of dict
        gt_masks_bev,     # Ground truth BEV masks (optional)
        gt_bboxes_3d,     # List of LiDARInstance3DBoxes
        gt_labels_3d,     # List of Tensors
    ):
        # ① 提取各传感器 BEV 特征
        features = []

        if self.encoders.get("camera"):
            x = self.extract_camera_features(img, points, radar, metas)
            features.append(x)  # [B, C_cam, H_bev, W_bev]

        if self.encoders.get("lidar"):
            x = self.extract_features(points, self.encoders["lidar"], metas)
            features.append(x)  # [B, C_lidar, H_bev, W_bev]

        if self.encoders.get("radar"):
            x = self.extract_features(radar, self.encoders["radar"], metas)
            features.append(x)  # [B, C_radar, H_bev, W_bev]

        # ② 融合
        if self.fuser is not None:
            x = self.fuser(features)     # [B, C_fused, H_bev, W_bev]
        else:
            x = features[0]

        # ③ Decoder
        x = self.decoder["backbone"](x)  # List of feature maps
        x = self.decoder["neck"](x)      # List of multi-scale features

        # ④ Heads
        losses = {}
        for name, head in self.heads.items():
            results = head.forward_train(x, metas, gt_bboxes_3d, gt_labels_3d, gt_masks_bev)
            for k, v in results.items():
                losses[f"{name}/{k}"] = v * self.loss_scale.get(name, 1.0)

        return losses
```

---

## 3. 相机分支：extract_camera_features()

**文件:** `mmdet3d/models/fusion_models/bevfusion.py`

```python
def extract_camera_features(self, img, points, radar, metas):
    """
    输入: img [B, N, 3, H, W]
    输出: BEV特征 [B, C_cam, H_bev, W_bev]
    """
    B, N, C, H, W = img.shape
    img = img.view(B * N, C, H, W)

    # ① Backbone (SwinTransformer)
    # 输出: List of feature maps at different scales
    img_feats = self.encoders["camera"]["backbone"](img)
    # 例: [(B*N, 192, H/8, W/8), (B*N, 384, H/16, W/16), (B*N, 768, H/32, W/32)]

    # ② Neck (GeneralizedLSSFPN)
    # 融合多尺度特征
    img_feats = self.encoders["camera"]["neck"](img_feats)
    # 输出: (B*N, C_neck=256, H_feat, W_feat)  单尺度

    # ③ View Transform (vtransform)
    # 将图像特征 Lift 到 3D，再 Splat 到 BEV
    bev_feats = self.encoders["camera"]["vtransform"](
        img_feats,    # (B*N, C, H_feat, W_feat)
        points,       # 用于深度监督
        radar,
        metas,        # 包含标定矩阵
    )
    # 输出: [B, C_cam, H_bev, W_bev]

    return bev_feats
```

---

## 4. 相机 View Transform (LSS方法)

### 4.1 初始化：创建视锥体 (frustum)

**文件:** `mmdet3d/models/vtransforms/base.py`

```python
class BaseTransform(nn.Module):

    def create_frustum(self):
        """
        创建图像空间视锥体坐标
        输出: [D, fH, fW, 3]  其中3为(u, v, d)
        """
        # depth bins: dbound = [min_d, max_d, step]
        d_coords = torch.arange(*self.dbound, dtype=torch.float)  # [D]
        D = d_coords.shape[0]

        # 图像坐标网格
        x_coords = torch.linspace(0, self.image_size[1]-1, self.feature_size[1])
        y_coords = torch.linspace(0, self.image_size[0]-1, self.feature_size[0])
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)

        # 扩展到 [D, fH, fW]
        d_grid = d_coords.view(D,1,1).expand(D, *self.feature_size)
        x_grid = x_grid.view(1, *self.feature_size).expand(D, *self.feature_size)
        y_grid = y_grid.view(1, *self.feature_size).expand(D, *self.feature_size)

        # 拼合: [D, fH, fW, 3]
        frustum = torch.stack([x_grid, y_grid, d_grid], -1)
        return nn.Parameter(frustum, requires_grad=False)
```

### 4.2 计算 3D 几何坐标

```python
def get_geometry(self, metas):
    """
    将视锥体坐标从相机系投影到 LiDAR 坐标系
    输出: geom [B, N, D, fH, fW, 3]
    """
    frustum = self.frustum  # [D, fH, fW, 3]
    D, fH, fW = frustum.shape[:3]

    # 扩展到 batch×camera 维度
    points = frustum.unsqueeze(0).unsqueeze(0)  # [1,1,D,fH,fW,3]
    points = points.expand(B, N, D, fH, fW, 3)
    points = points.clone()

    # 像素坐标 → 归一化相机坐标 (除以深度d)
    points[..., :2] = points[..., :2] * points[..., 2:3]
    # 现在 points 为 [u*d, v*d, d]

    # 齐次坐标: [B, N, D, fH, fW, 4]
    points_hom = torch.cat([points, torch.ones_like(points[..., :1])], -1)

    # 应用相机内参: camera coords = K^-1 @ [u*d, v*d, d, 1]^T
    # 应用外参: lidar coords = lidar2camera^-1 @ camera coords
    # metas 中包含 lidar2image [B, N, 4, 4]
    lidar2img = metas["lidar2image"]         # [B, N, 4, 4]
    img2lidar = torch.inverse(lidar2img)     # [B, N, 4, 4]

    points_hom = points_hom.view(B, N, D*fH*fW, 4, 1)
    img2lidar = img2lidar.view(B, N, 1, 4, 4)
    points_lidar = torch.matmul(img2lidar, points_hom)  # [B, N, D*fH*fW, 4, 1]

    # 取 xyz
    geom = points_lidar[..., :3, 0].view(B, N, D, fH, fW, 3)
    return geom
```

### 4.3 BEV Pooling (核心加速)

```python
def bev_pool(self, geom, feats):
    """
    将相机特征从 3D 空间投影到 BEV 网格
    输入:
        geom:  [B, N, D, fH, fW, 3]  LiDAR坐标系中的3D点坐标
        feats: [B, N, D, fH, fW, C]  对应的特征向量
    输出:
        bev:   [B, C, H_bev, W_bev]
    """
    B, N, D, H, W, C = feats.shape

    # ① 将 LiDAR 坐标转换为 BEV 网格索引
    coords = geom  # [B, N, D, H, W, 3]
    # 转换公式: idx = (coord - range_min) / voxel_size
    # xbound = [x_min, x_max, x_step]
    x_idx = (coords[..., 0] - self.xbound[0]) / self.xbound[2]
    y_idx = (coords[..., 1] - self.ybound[0]) / self.ybound[2]
    z_idx = (coords[..., 2] - self.zbound[0]) / self.zbound[2]

    # ② 拼合索引 [B*N*D*H*W, 4] 其中最后一维为 (z_idx, x_idx, y_idx, batch_idx)
    coords_flat = torch.stack([z_idx, x_idx, y_idx], -1)  # [B,N,D,H,W,3]
    # ... flatten and add batch index

    # ③ 过滤越界点
    mask = (coords_flat[...,0] >= 0) & (coords_flat[...,0] < D_bev) & ...

    # ④ 调用 CUDA kernel
    from mmdet3d.ops.bev_pool import bev_pool as bev_pool_op
    feats_flat = feats.reshape(-1, C)
    out = bev_pool_op(feats_flat, coords_flat_int, B, D_bev, H_bev, W_bev)
    # out: [B, D_bev, H_bev, W_bev, C]

    # ⑤ 整理输出
    out = out.permute(0, 4, 1, 2, 3)  # [B, C, D_bev, H_bev, W_bev]
    out = out.flatten(1, 2)           # [B, C*D_bev, H_bev, W_bev]
    return out
```

### 4.4 LSSTransform.get_cam_feats()

**文件:** `mmdet3d/models/vtransforms/lss.py`

```python
class LSSTransform(BaseTransform):

    def get_cam_feats(self, x):
        """
        输入: x [B, N, C_in, fH, fW]  backbone+neck 输出
        输出: feats [B, N, D, fH, fW, C_out]  深度加权后的特征
        """
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)

        # depthnet: Conv2d(C_in, D + C_out)
        x = self.depthnet(x)
        # depth_logits: [B*N, D, H, W]
        # feat:         [B*N, C_out, H, W]
        depth = x[:, :self.D].softmax(dim=1)  # depth 分布 [B*N, D, H, W]
        feat  = x[:, self.D:]                 # context 特征 [B*N, C_out, H, W]

        # Outer product: weight features by depth distribution
        # depth: [B*N, D, 1, H, W]
        # feat:  [B*N, 1, C, H, W]
        # result: [B*N, D, C, H, W]
        feat = depth.unsqueeze(2) * feat.unsqueeze(1)

        # Reshape to [B, N, D, H, W, C]
        feat = feat.view(B, N, self.D, C, H, W)
        feat = feat.permute(0, 1, 2, 4, 5, 3)  # [B, N, D, H, W, C]
        return feat
```

---

## 5. LiDAR 分支：extract_features()

**文件:** `mmdet3d/models/fusion_models/bevfusion.py`

```python
def extract_features(self, points, encoder, metas):
    """
    LiDAR / Radar 点云 → 稀疏3D特征 → BEV
    """
    # ① 体素化
    voxels, coors, num_points = self.voxelize(points, encoder)
    # voxels: [total_voxels, max_pts, 5]
    # coors:  [total_voxels, 4]  (batch_idx, z, y, x)

    # ② VoxelNet / PillarNet encoder
    voxel_feats = encoder["backbone"](voxels, coors, len(points))
    # 输出: [B, C_lidar, H_bev, W_bev]  (已经是 2D BEV 图)
    return voxel_feats

def voxelize(self, points, encoder):
    """
    将点云转换为体素表示
    """
    voxel_layer = encoder["voxelize"]
    voxels, coors, num_points = [], [], []
    for pts in points:
        # Voxelization: 对每个样本做体素化
        v, c, n = voxel_layer(pts)
        voxels.append(v)
        coors.append(c)
        num_points.append(n)

    # batch 拼合，coors 中加入 batch 索引
    voxels = torch.cat(voxels, 0)       # [total_voxels, max_pts, 5]
    num_points = torch.cat(num_points)  # [total_voxels]
    # 为每个体素的 coors 加上 batch 索引
    coors_batch = []
    for i, c in enumerate(coors):
        pad = c.new_full((c.shape[0], 1), i)
        coors_batch.append(torch.cat([pad, c], dim=1))
    coors = torch.cat(coors_batch, 0)   # [total_voxels, 4]

    return voxels, coors, num_points
```

### 5.1 SparseEncoder 稀疏3D卷积

**文件:** `mmdet3d/models/backbones/sparse_encoder.py`

```python
class SparseEncoder(nn.Module):
    """
    稀疏3D卷积编码器，将体素转换为 BEV 特征图
    """
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        # 输入: voxel_features [total_voxels, C_in]  (mean-pooled per voxel)
        #       coors: [total_voxels, 4]  (batch, z, y, x)

        # 构建稀疏张量
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=coors.int(),
            spatial_shape=self.sparse_shape,  # [z_dim, y_dim, x_dim]
            batch_size=batch_size,
        )

        # 多阶段稀疏卷积
        x = self.conv_input(input_sp_tensor)  # SubMConv3d

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # 最后阶段输出转为 dense
        out = self.conv_out(encode_features[-1])
        out = out.dense()  # [B, C, z, y, x]

        # 沿 z 轴 flatten 得到 BEV
        N, C, D, H, W = out.shape
        out = out.view(N, C * D, H, W)  # [B, C*z, H_bev, W_bev]

        return out
```

---

## 6. Fuser（融合模块）

### 6.1 ConvFuser

**文件:** `mmdet3d/models/fusers/conv.py`

```python
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        # in_channels: list, e.g. [80, 256]
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs: [cam_bev[B,80,H,W], lidar_bev[B,256,H,W]]
        x = torch.cat(inputs, dim=1)  # [B, 336, H, W]
        return super().forward(x)     # [B, 256, H, W]
```

### 6.2 AddFuser

**文件:** `mmdet3d/models/fusers/add.py`

```python
class AddFuser(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        # 每个传感器分支一个投影层
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.dropout = dropout

    def forward(self, inputs):
        # 训练时随机丢弃某些传感器 (sensor dropout)
        if self.training and self.dropout > 0:
            mask = torch.rand(len(inputs)) > self.dropout
            if not mask.any():
                mask[0] = True  # 至少保留一个

        out = None
        for i, (proj, feat) in enumerate(zip(self.projections, inputs)):
            if self.training and self.dropout > 0 and not mask[i]:
                continue
            projected = proj(feat)  # [B, out_channels, H, W]
            out = projected if out is None else out + projected

        return out
```

---

## 7. Decoder 模块

Decoder 由配置中的 `decoder.backbone` 和 `decoder.neck` 组成。

典型配置:
- **backbone**: `GeneralizedResNet` 或简单的 `nn.Sequential` of 2D convolutions
- **neck**: `SECONDFPN` 或 `GeneralizedLSSFPN`

```python
# decoder forward
x = self.decoder["backbone"](fused_bev)   # List[Tensor]
x = self.decoder["neck"](x)               # List[Tensor] (multi-scale)
```

---

## 8. TransFusionHead（检测头）

**文件:** `mmdet3d/models/heads/bbox/transfusion.py`

```python
class TransFusionHead(nn.Module):

    def forward_single(self, inputs, metas):
        """
        输入: inputs - List of BEV feature maps
        """
        # ① 合并多尺度特征 (if needed)
        feats = inputs[0]  # [B, C, H, W]

        # ② Shared Conv
        feats = self.shared_conv(feats)  # [B, hidden_C, H, W]

        # ③ 热图预测 → 初始化查询位置
        heatmap = self.heatmap_head(feats)  # [B, num_classes, H, W]
        # 取 top-K 位置作为初始 proposals
        top_proposals = heatmap.view(B, -1).topk(self.num_proposals)
        # top_proposals_index: [B, num_proposals]  展平的 BEV 索引

        # ④ 从 BEV 特征提取 proposal 特征
        query_feat = feats.view(B, C, H*W)
        query_feat = query_feat.gather(2, top_proposals_index.unsqueeze(1).expand(-1, C, -1))
        # query_feat: [B, C, num_proposals]

        # ⑤ 位置编码
        query_pos = self.bev_embedding(top_proposals)  # [B, num_proposals, C]

        # ⑥ Transformer Decoder (多层)
        predictions = []
        for layer in self.decoder_layers:
            query_feat = layer(
                query_feat,    # [B, C, num_proposals]  self-attention
                feats,         # BEV feature map  cross-attention key/value
                query_pos,
            )
            # 每层都预测一次结果
            pred = self.prediction_heads(query_feat)
            predictions.append(pred)

        # ⑦ 每层预测内容:
        # center:    [B, 2, num_proposals]  (dx, dy) BEV center offset
        # height:    [B, 1, num_proposals]  z
        # dim:       [B, 3, num_proposals]  (l, w, h)
        # rot:       [B, 2, num_proposals]  (sin, cos) yaw
        # vel:       [B, 2, num_proposals]  (vx, vy)
        # iou:       [B, 1, num_proposals]

        return predictions

    def forward_train(self, inputs, metas, gt_bboxes, gt_labels, gt_masks=None):
        """计算损失"""
        preds = self.forward_single(inputs, metas)

        losses = {}
        # 最终层 prediction 用于 assignment + loss
        final_pred = preds[-1]

        # Hungarian matching
        assignments = self.matcher(final_pred, gt_bboxes, gt_labels)

        # 分类损失 (GaussianFocalLoss on heatmap)
        losses["loss_heatmap"] = self.loss_cls(heatmap, heatmap_target)

        # Bbox 回归损失 (L1)
        losses["loss_bbox"] = self.loss_bbox(pred_bboxes[pos_mask], gt_bboxes_encoded)

        # IoU 损失
        losses["loss_iou"] = self.loss_iou(pred_iou[pos_mask], iou_targets)

        return losses
```

---

## 9. CUDA BEV Pool 算子

**文件:** `mmdet3d/ops/bev_pool/bev_pool.py` + `src/bev_pool_cuda.cu`

### Python 接口

```python
class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, B, D, H, W):
        """
        x:          [N, C]  N个点的特征
        geom_feats: [N, 4]  (z_idx, x_idx, y_idx, batch_idx)
        """
        # ① 计算每个点在 BEV 体积中的线性索引
        ranks = geom_feats[:, 0] * (W * D * B) + \
                geom_feats[:, 1] * (D * B) + \
                geom_feats[:, 2] * B + \
                geom_feats[:, 3]

        # ② 按 rank 排序 (将同一 BEV 格子的点放在一起)
        indices = ranks.argsort()
        x = x[indices]
        geom_feats = geom_feats[indices]
        ranks = ranks[indices]

        # ③ 找到每个区间的起始位置和长度
        # 相邻 rank 不同时标记区间边界
        interval_starts, interval_lengths = find_intervals(ranks)

        # ④ 调用 CUDA kernel
        out = bev_pool_ext.bev_pool_forward(
            x, geom_feats, interval_starts, interval_lengths, B, D, H, W
        )
        # out: [B, D, H, W, C]

        ctx.save_for_backward(x, geom_feats, interval_starts, interval_lengths)
        ctx.shape = (B, D, H, W)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        # 将梯度分发回每个输入点
        x_grad = bev_pool_ext.bev_pool_backward(out_grad, ...)
        return x_grad, None, None, None, None, None
```

### CUDA Kernel (简化)

```cuda
// bev_pool_cuda.cu
__global__ void bev_pool_forward_kernel(
    const float* x,               // [N, C]
    const int*   geom_feats,      // [N, 4]
    const int*   interval_starts, // [n_intervals]
    const int*   interval_lengths,// [n_intervals]
    float*       out,             // [B, D, H, W, C]
    int n_intervals, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_intervals * C) return;

    int interval_idx = idx / C;
    int c = idx % C;

    int start  = interval_starts[interval_idx];
    int length = interval_lengths[interval_idx];

    // 读取该区间对应的 BEV 格子坐标
    int z = geom_feats[start * 4 + 0];
    int x_idx = geom_feats[start * 4 + 1];
    int y_idx = geom_feats[start * 4 + 2];
    int b = geom_feats[start * 4 + 3];

    // 求和聚合
    float psum = 0;
    for (int i = 0; i < length; i++) {
        psum += x[(start + i) * C + c];
    }

    // 写入输出
    out[((b * D + z) * H + x_idx) * W * C + y_idx * C + c] = psum;
}
```

---

## 10. 数据集与数据管道

**文件:** `mmdet3d/datasets/nuscenes_dataset.py`

### 数据预处理 Pipeline

训练时数据字典的变化过程:

```
原始文件
  ↓ LoadMultiViewImageFromFiles
  img: ndarray [N, H, W, 3]

  ↓ LoadPointsFromFile + LoadPointsFromMultiSweeps
  points: LiDARPoints object ([N_pts, 5])

  ↓ LoadRadarPointsMultiSweeps (optional)
  radar: RadarPoints object

  ↓ LoadAnnotations3D
  gt_bboxes_3d: LiDARInstance3DBoxes ([N_box, 7])
  gt_labels_3d: ndarray [N_box]

  ↓ ObjectPaste (GT-Paste 数据增强)
  points, gt_bboxes_3d, gt_labels_3d 更新

  ↓ ImageAug3D
  img: 增广后图像
  img_aug_matrix: [N, 3, 3] 增广变换矩阵

  ↓ GlobalRotScaleTrans
  points, gt_bboxes_3d 变换
  lidar_aug_matrix: [4, 4]

  ↓ RandomFlip3D
  以50%概率翻转点云和bbox

  ↓ PointsRangeFilter → ObjectRangeFilter → ObjectNameFilter
  过滤超出范围的点和目标

  ↓ ImageNormalize
  img 标准化 (ImageNet均值/方差)

  ↓ GridMask (prob=0.7)
  img 随机格子遮挡增强

  ↓ PointShuffle
  打乱点云顺序

  ↓ DefaultFormatBundle3D
  img: Tensor [N, 3, H, W]
  points: Tensor

  ↓ Collect3D
  最终输入字典: {img, points, radar, gt_bboxes_3d, gt_labels_3d, metas}
```

### metas 字典内容

```python
metas = {
    # 标定矩阵
    "camera_intrinsics": np.ndarray [N, 3, 3],   # 内参
    "camera2ego":        np.ndarray [N, 4, 4],   # 相机到车体
    "lidar2ego":         np.ndarray [4, 4],       # LiDAR到车体
    "lidar2camera":      np.ndarray [N, 4, 4],   # LiDAR到相机
    "lidar2image":       np.ndarray [N, 4, 4],   # LiDAR到图像(含内参)
    "camera2lidar":      np.ndarray [N, 4, 4],   # 相机到LiDAR
    # 增广矩阵
    "img_aug_matrix":    np.ndarray [N, 3, 3],   # 图像增广变换
    "lidar_aug_matrix":  np.ndarray [4, 4],       # 点云增广变换
    # 图像参数
    "img_shape":         List[(H, W)],            # 各相机图像尺寸
    "ori_shape":         List[(H, W)],
    # 场景信息
    "sample_idx":        str,
    "timestamp":         float,
}
```

---

## 11. 完整训练流程

```python
# tools/train.py

def main():
    # 1. 初始化分布式训练
    dist.init(launcher='pytorch')

    # 2. 加载配置
    cfg = Config.fromfile(args.config)
    # 命令行参数覆盖: --model.encoders.camera.backbone.init_cfg.checkpoint path

    # 3. 构建 Dataset
    dataset = build_dataset(cfg.data.train)

    # 4. 构建 DataLoader
    loader = build_dataloader(dataset, samples_per_gpu=cfg.batch_size, ...)

    # 5. 构建模型
    model = build_model(cfg.model)
    model.init_weights()

    # 6. (可选) 加载预训练权重
    if args.load_from:
        load_checkpoint(model, args.load_from)

    # 7. 开始训练 (torchpack Runner)
    runner = Runner(model, optimizer=..., scheduler=...)
    runner.run(loader, max_epochs=cfg.max_epochs)
```

训练的每步:
```python
# Base3DFusionModel.train_step()
def train_step(self, data, optimizer):
    losses = self(**data)           # BEVFusion.forward(return_loss=True, **data)
    loss, log_vars = self._parse_losses(losses)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))
    return outputs
```

---

## 12. 测试/推理流程

```python
# tools/test.py

# 1. 构建模型 + 加载 checkpoint
model = build_model(cfg.model)
load_checkpoint(model, checkpoint_path)
model = fuse_conv_bn(model)   # 加速推理

# 2. 推理
for data in dataloader:
    with torch.no_grad():
        result = model(return_loss=False, **data)
    # result: List of {
    #   'boxes_3d': LiDARInstance3DBoxes,
    #   'scores_3d': Tensor,
    #   'labels_3d': Tensor,
    # }

# 3. 评估
dataset.evaluate(results, metric='bbox')
# 输出: mAP, NDS (nuScenes Detection Score)
```

推理时 `BEVFusion.forward_test()` 调用:
```python
def forward_test(self, img, points, metas, ...):
    result = self.forward_single(img, points, ...)
    # 经过 head.get_bboxes() 完成 NMS 后处理
    return result
```

---

## 13. 模型参数规模 (参考)

| 组件 | 典型参数量 |
|------|-----------|
| Camera Backbone (SwinT-Small) | ~50M |
| Camera Neck (LSSFPN) | ~5M |
| LSSTransform (depthnet) | ~1M |
| LiDAR SparseEncoder | ~15M |
| ConvFuser | <0.5M |
| Decoder | ~10M |
| TransFusionHead | ~15M |
| **总计** | **~100M** |

---

## 14. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `point_cloud_range` | `[-54, -54, -5, 54, 54, 3]` | 点云范围 (m) |
| `voxel_size` | `[0.075, 0.075, 0.2]` | 体素大小 (m) |
| `image_size` | `[256, 704]` | 相机输入分辨率 |
| `dbound` | `[1.0, 60.0, 0.5]` | 深度箱范围 |
| `num_proposals` | 200 | TransFusion 查询数 |
| `max_epochs` | 6 / 12 | 训练轮数 |
| `batch_size` | 4 (per GPU) | 每GPU批大小 |
| Camera sweeps | 1 | 相机帧数 |
| LiDAR sweeps | 10 | LiDAR 累积帧数 |
| Radar sweeps | 6 | 雷达累积帧数 |
