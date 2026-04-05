"""
07_annotate.py
轻量 BEV 点云标注工具（纯 Python，浏览器操作）

功能：
  - 在浏览器中显示 BEV 点云俯视图
  - 鼠标点击+拖拽画矩形框
  - 支持旋转（键盘 Q/E 调整 yaw）
  - 支持上一帧/下一帧导航
  - 支持多类别：vehicle / pedestrian（可扩展）
  - BEV 图上显示模型有效检测边界（路径A范围）
  - 保存为 data/annos/XXXXXX.txt（与 04_generate_pkl.py 格式一致）

用法：
  conda run -n superpoint python data/07_annotate.py
  # 然后在浏览器打开 http://localhost:8765

格式说明（保存的 .txt）：
  cx  cy  cz  length  width  height  yaw  class_name
  示例: 5.20 -3.10 0.30 4.50 1.80 1.60 0.78 vehicle
"""

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# ── 路径配置 ─────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent
LIDAR_DIR  = DATA_DIR / "samples" / "LIDAR_TOP"
ANNO_DIR   = DATA_DIR / "annos"

# BEV 显示范围（用你自己数据的完整范围，方便看全局）
PCR        = [-100.0, -100.0, -3.0, 100.0, 100.0, 10.0]

# 路径A：公开模型实际有效检测范围（会在 BEV 上画虚线边界提示）
MODEL_RANGE = [-54.0, -54.0, 54.0, 54.0]  # xmin, ymin, xmax, ymax

# 支持的类别及默认尺寸 {cls: (default_cz, default_height, default_l, default_w)}
CLASSES = {
    "vehicle":     (0.3, 1.6, 4.5, 1.8),
    "pedestrian":  (0.5, 1.7, 0.7, 0.7),
}
CLASS_LIST = list(CLASSES.keys())

PORT = 8765


# ── BEV 图生成（返回 PNG bytes）──────────────────────────────

def make_bev_png(bin_path: Path, boxes: list,
                 x_range=None, y_range=None) -> bytes:
    """PIL 快速渲染 BEV 点云图（~10ms vs matplotlib ~500ms）"""
    import io
    from PIL import Image, ImageDraw

    OUT_W, OUT_H = 700, 700  # 输出图像尺寸（像素）

    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)
    if x_range is None: x_range = (PCR[0], PCR[3])
    if y_range is None: y_range = (PCR[1], PCR[4])
    xr0, xr1 = x_range
    yr0, yr1 = y_range
    xspan = xr1 - xr0
    yspan = yr1 - yr0

    # ── 点云密度图 ──────────────────────────────────────────
    xf, yf = pts[:, 0], pts[:, 1]
    mask = (xf >= xr0) & (xf <= xr1) & (yf >= yr0) & (yf <= yr1)
    xf, yf = xf[mask], yf[mask]

    xi = ((xf - xr0) / xspan * OUT_W).astype(np.int32).clip(0, OUT_W - 1)
    # y 轴翻转：yr0 在底部，yr1 在顶部
    yi = ((1.0 - (yf - yr0) / yspan) * OUT_H).astype(np.int32).clip(0, OUT_H - 1)

    density = np.zeros((OUT_H, OUT_W), dtype=np.float32)
    np.add.at(density, (yi, xi), 1)
    density = np.log1p(density)
    if density.max() > 0:
        density /= density.max()

    # hot colormap: 黑→红→黄→白
    r = np.clip(density * 3.0,       0, 1)
    g = np.clip(density * 3.0 - 1.0, 0, 1)
    b = np.clip(density * 3.0 - 2.0, 0, 1)
    rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(pil_img)

    # ── 辅助：世界坐标 → 像素 ──────────────────────────────
    def w2p(wx, wy):
        px = int((wx - xr0) / xspan * OUT_W)
        py = int((1.0 - (wy - yr0) / yspan) * OUT_H)
        return px, py

    # ── 传感器原点标记 ──────────────────────────────────────
    if xr0 <= 0 <= xr1 and yr0 <= 0 <= yr1:
        ox, oy = w2p(0, 0)
        draw.line([(ox-10, oy), (ox+10, oy)], fill=(0,255,255), width=2)
        draw.line([(ox, oy-10), (ox, oy+10)], fill=(0,255,255), width=2)

    # ── 模型有效范围边界（±54m 虚线框）────────────────────
    mx0, my0, mx1, my1 = MODEL_RANGE
    if not (mx1 < xr0 or mx0 > xr1 or my1 < yr0 or my0 > yr1):
        bx0, by0 = w2p(max(mx0, xr0), min(my1, yr1))
        bx1, by1 = w2p(min(mx1, xr1), max(my0, yr0))
        # 画虚线矩形
        for i in range(bx0, bx1, 8):
            draw.line([(i, by0), (min(i+4, bx1), by0)], fill=(0,191,255), width=2)
            draw.line([(i, by1), (min(i+4, bx1), by1)], fill=(0,191,255), width=2)
        for i in range(by0, by1, 8):
            draw.line([(bx0, i), (bx0, min(i+4, by1))], fill=(0,191,255), width=2)
            draw.line([(bx1, i), (bx1, min(i+4, by1))], fill=(0,191,255), width=2)

    # ── 标注框 ──────────────────────────────────────────────
    cls_colors = {"vehicle": (57,255,20), "pedestrian": (255,165,0)}
    for b in boxes:
        cx, cy, cz, l, w, h, yaw, cls = b
        color = cls_colors.get(cls, (255,255,0))
        ca, sa = float(np.cos(yaw)), float(np.sin(yaw))
        corners = []
        for lx, ly in [(l/2,w/2),(-l/2,w/2),(-l/2,-w/2),(l/2,-w/2)]:
            wx = ca*lx - sa*ly + cx
            wy = sa*lx + ca*ly + cy
            corners.append(w2p(wx, wy))
        for i in range(4):
            draw.line([corners[i], corners[(i+1)%4]], fill=color, width=2)

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


# ── 标注文件读写 ──────────────────────────────────────────────

def load_anno(stem: str) -> list:
    """加载现有标注，返回 list of [cx,cy,cz,l,w,h,yaw,cls]"""
    path = ANNO_DIR / f"{stem}.txt"
    boxes = []
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cx, cy, cz, l, w, h, yaw = [float(x) for x in parts[:7]]
                cls = parts[7] if len(parts) > 7 else "vehicle"
                boxes.append([cx, cy, cz, l, w, h, yaw, cls])
    return boxes


def save_anno(stem: str, boxes: list):
    """保存标注到 annos/XXXXXX.txt"""
    ANNO_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNO_DIR / f"{stem}.txt"
    with open(path, "w") as f:
        for b in boxes:
            cx, cy, cz, l, w, h, yaw, cls = b
            f.write(f"{cx:.3f} {cy:.3f} {cz:.3f} "
                    f"{l:.3f} {w:.3f} {h:.3f} "
                    f"{yaw:.4f} {cls}\n")


# ── HTTP 服务器 ───────────────────────────────────────────────

bin_files = sorted(LIDAR_DIR.glob("*.bin"))
state = {"idx": 0}   # 当前帧索引


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BEV 点云标注工具</title>
<style>
body { font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 10px; }
h2 { margin: 0; color: #a0cfff; }
.toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
button { padding: 6px 14px; background: #2a4080; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
button:hover { background: #3a5090; }
button.danger { background: #803030; }
button.cls-btn { padding: 6px 18px; font-size: 14px; border: 2px solid transparent; }
button.cls-btn.active-vehicle  { border-color: #39ff14; color: #39ff14; }
button.cls-btn.active-pedestrian { border-color: orange; color: orange; }
input[type=number], input[type=text] { width: 70px; padding: 4px; background: #222; color: #eee; border: 1px solid #555; border-radius: 4px; }
label { font-size: 13px; }
select { background:#222; color:#eee; border:1px solid #555; border-radius:4px; padding:2px; }
#canvas-wrap { position: relative; display: inline-block; cursor: crosshair; }
canvas { display: block; }
#overlay { position: absolute; top: 0; left: 0; cursor: crosshair; }
#info { margin: 6px 0; font-size: 13px; color: #aaa; }
#boxlist { margin-top: 8px; }
.box-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 12px;
            background: #1e3060; padding: 4px 8px; border-radius: 4px; }
.box-item button { padding: 2px 8px; font-size: 11px; }
.box-vehicle    { border-left: 3px solid #39ff14; }
.box-pedestrian { border-left: 3px solid orange; }
#status { color: #6f6; font-size: 13px; margin: 4px 0; }
</style>
</head>
<body>
<h2>BEV 点云标注工具</h2>

<div class="toolbar">
  <button onclick="prevFrame()">上一帧</button>
  <span id="frame-info">-</span>
  <button onclick="nextFrame()">下一帧</button>
  &nbsp;
  <span style="font-size:13px;color:#aaa">类别：</span>
  <button id="btn-vehicle"    class="cls-btn active-vehicle"
          onclick="setClass('vehicle')">vehicle</button>
  <button id="btn-pedestrian" class="cls-btn"
          onclick="setClass('pedestrian')">pedestrian</button>
  &nbsp;
  <label>高度(m): <input type="number" id="h-input"  value="1.6" step="0.1" style="width:55px"></label>
  <label>cz(m):   <input type="number" id="cz-input" value="0.3" step="0.1" style="width:55px"></label>
  &nbsp;
  <button onclick="saveAnno()" style="background:#2a7040">保存标注</button>
  <button onclick="clearAll()" class="danger">清空</button>
</div>

<div id="info">选类别后拖拽画框 | Q/E 旋转选中框 | Delete 删除 | 蓝色虚线=模型有效范围(+-54m)</div>
<div id="status"></div>

<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div>
    <div id="canvas-wrap">
      <img id="bev-img" src="" style="max-width:700px; display:block;">
      <canvas id="overlay" width="700" height="700"></canvas>
    </div>
    <div style="font-size:11px; color:#888; margin-top:4px">
      坐标系：X前方，Y左方，原点=传感器
    </div>
  </div>
  <div style="min-width:280px;">
    <b>当前帧标注框：</b>
    <div id="boxlist"></div>
    <div style="margin-top:12px; font-size:12px; color:#aaa;">
      <b>操作说明：</b><br>
      - 选类别后拖拽：画新框<br>
      - 点击框中心：选中（高亮）<br>
      - Q / E：旋转选中框 +-5 度<br>
      - Delete：删除选中框<br>
      - 右侧列表数值实时生效<br>
      <br>
      <span style="color:#39ff14">绿色</span> = vehicle<br>
      <span style="color:orange">橙色</span> = pedestrian<br>
      <span style="color:deepskyblue">蓝色虚线</span> = 模型有效范围
    </div>
  </div>
</div>

<script>
let frameIdx = 0;
let totalFrames = 0;
let boxes = [];
let selectedIdx = -1;
let dragging = false;
let dragStart = null;
let dragCur = [0, 0];

// 固定世界坐标范围（不随缩放改变）
const xRange = [-100, 100];
const yRange = [-100, 100];
const CANVAS_W = 700;
const CANVAS_H = 700;

// CSS transform 状态
let xfScale = 1.0;
let xfTx = 0, xfTy = 0;

const clsDefaults = {
  vehicle:    {h: 1.6, cz: 0.3},
  pedestrian: {h: 1.7, cz: 0.5},
};
let currentCls = 'vehicle';

const img    = document.getElementById('bev-img');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('canvas-wrap');

// ── CSS transform ─────────────────────────────────────────
function applyCSS() {
  wrap.style.transformOrigin = '0 0';
  wrap.style.transform = `translate(${xfTx}px,${xfTy}px) scale(${xfScale})`;
}

// 把鼠标事件坐标转为 canvas 本地坐标（700x700 空间）
function getLocal(e) {
  const r = wrap.getBoundingClientRect();
  return [(e.clientX - r.left) / xfScale,
          (e.clientY - r.top)  / xfScale];
}

// ── 世界坐标 ↔ canvas 像素（始终用固定范围）────────────────
function imgToWorld(lx, ly) {
  const wx = xRange[0] + (lx / CANVAS_W) * (xRange[1] - xRange[0]);
  const wy = yRange[0] + ((CANVAS_H - ly) / CANVAS_H) * (yRange[1] - yRange[0]);
  return [wx, wy];
}
function worldToImg(wx, wy) {
  const lx = (wx - xRange[0]) / (xRange[1] - xRange[0]) * CANVAS_W;
  const ly = CANVAS_H - (wy - yRange[0]) / (yRange[1] - yRange[0]) * CANVAS_H;
  return [lx, ly];
}

function setClass(cls) {
  currentCls = cls;
  document.getElementById('btn-vehicle').className    = 'cls-btn' + (cls==='vehicle'?    ' active-vehicle'   : '');
  document.getElementById('btn-pedestrian').className = 'cls-btn' + (cls==='pedestrian'? ' active-pedestrian': '');
  const d = clsDefaults[cls];
  document.getElementById('h-input').value  = d.h;
  document.getElementById('cz-input').value = d.cz;
}

function loadFrame(idx) {
  frameIdx = idx;
  // 切帧重置缩放
  xfScale = 1.0; xfTx = 0; xfTy = 0; applyCSS();
  fetch('/api/frame?idx=' + idx).then(r => r.json()).then(d => {
    totalFrames = d.total;
    boxes = d.boxes;
    selectedIdx = -1;
    document.getElementById('frame-info').textContent =
      'Frame ' + idx + ' / ' + (totalFrames-1) + '  [' + d.stem + ']';
    img.src = '/api/bev?idx=' + idx + '&t=' + Date.now();
    img.onload = () => { canvas.width = CANVAS_W; canvas.height = CANVAS_H; drawOverlay(); };
    renderBoxList();
  });
}

// ── 绘制覆盖层 ────────────────────────────────────────────
function clsColor(cls) { return cls === 'pedestrian' ? 'orange' : '#39ff14'; }

function drawOverlay() {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  if (selectedIdx >= 0 && selectedIdx < boxes.length)
    drawBox(boxes[selectedIdx], 'yellow', 3);
  if (dragging && dragStart) {
    ctx.strokeStyle = 'cyan'; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    const [x1,y1] = dragStart, [x2,y2] = dragCur;
    ctx.strokeRect(Math.min(x1,x2), Math.min(y1,y2), Math.abs(x2-x1), Math.abs(y2-y1));
    ctx.setLineDash([]);
  }
}

function drawBox(b, color, lw) {
  const ca = Math.cos(b.yaw), sa = Math.sin(b.yaw);
  const corners = [[ b.l/2, b.w/2],[-b.l/2, b.w/2],[-b.l/2,-b.w/2],[ b.l/2,-b.w/2]]
    .map(([lx,ly]) => worldToImg(ca*lx - sa*ly + b.cx, sa*lx + ca*ly + b.cy));
  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  corners.forEach(c => ctx.lineTo(c[0], c[1]));
  ctx.closePath();
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
}

// ── 滚轮缩放（CSS transform，无需重渲图）────────────────
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const r = wrap.getBoundingClientRect();
  const dx = e.clientX - r.left;  // 鼠标相对 wrap 视觉左上角的偏移
  const dy = e.clientY - r.top;
  const factor = e.deltaY > 0 ? 0.8 : 1.25;
  // 保持鼠标所在点不动
  xfTx += dx * (1 - factor);
  xfTy += dy * (1 - factor);
  xfScale *= factor;
  // 限制缩放范围
  if (xfScale < 0.5) { const f = 0.5/xfScale; xfTx += dx*(1-f); xfTy += dy*(1-f); xfScale = 0.5; }
  if (xfScale > 20)  { const f = 20/xfScale;  xfTx += dx*(1-f); xfTy += dy*(1-f); xfScale = 20;  }
  applyCSS();
}, {passive: false});

// ── 右键拖动平移 ─────────────────────────────────────────
let panStart = null, panTx0, panTy0;
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => {
  if (e.button !== 2) return;
  e.preventDefault();
  const r = wrap.getBoundingClientRect();
  panStart = [e.clientX - r.left, e.clientY - r.top];
  panTx0 = xfTx; panTy0 = xfTy;
}, true);
window.addEventListener('mousemove', e => {
  if (!panStart || !(e.buttons & 2)) return;
  const r = wrap.getBoundingClientRect();
  xfTx = panTx0 + (e.clientX - r.left - panStart[0]);
  xfTy = panTy0 + (e.clientY - r.top  - panStart[1]);
  applyCSS();
});
window.addEventListener('mouseup', e => { if (e.button === 2) panStart = null; });

// ── 左键画框 / 选框 ───────────────────────────────────────
// 按下时只记录起点，不立即做 hitTest
// 松开时根据拖拽距离区分"点击选框"还是"画新框"
const DRAG_THRESHOLD = 5;  // 像素，低于此视为点击
canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const [lx, ly] = getLocal(e);
  dragging = true;
  dragStart = [lx, ly]; dragCur = [lx, ly];
});
canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  dragCur = getLocal(e); drawOverlay();
});
canvas.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  dragCur = getLocal(e);
  const [x1,y1] = dragStart, [x2,y2] = dragCur;
  const dist = Math.max(Math.abs(x2-x1), Math.abs(y2-y1));

  if (dist < DRAG_THRESHOLD) {
    // 点击：做 hitTest 选框
    const hit = hitTest(x1, y1);
    selectedIdx = hit;
    renderBoxList(); drawOverlay();
    return;
  }

  // 拖拽：画新框，不管有没有已有框在这里
  const [wx1,wy1] = imgToWorld(Math.min(x1,x2), Math.min(y1,y2));
  const [wx2,wy2] = imgToWorld(Math.max(x1,x2), Math.max(y1,y2));
  const h  = parseFloat(document.getElementById('h-input').value)  || clsDefaults[currentCls].h;
  const cz = parseFloat(document.getElementById('cz-input').value) || clsDefaults[currentCls].cz;
  boxes.push({cx:(wx1+wx2)/2, cy:(wy1+wy2)/2, cz, l:Math.abs(wx2-wx1), w:Math.abs(wy2-wy1), h, yaw:0, cls:currentCls});
  selectedIdx = boxes.length - 1;
  dragStart = null; renderBoxList(); drawOverlay();
});

// ── 键盘 ──────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (selectedIdx < 0) return;
  const step = 5 * Math.PI / 180;
  if (e.key==='q'||e.key==='Q') { boxes[selectedIdx].yaw += step; renderBoxList(); drawOverlay(); }
  else if (e.key==='e'||e.key==='E') { boxes[selectedIdx].yaw -= step; renderBoxList(); drawOverlay(); }
  else if (e.key==='Delete'||e.key==='Backspace') {
    boxes.splice(selectedIdx, 1); selectedIdx = -1;
    renderBoxList(); drawOverlay();
    img.src = '/api/bev?idx=' + frameIdx + '&t=' + Date.now();
  }
});

function hitTest(lx, ly) {
  for (let i = boxes.length-1; i >= 0; i--) {
    const [ipx,ipy] = worldToImg(boxes[i].cx, boxes[i].cy);
    if (Math.sqrt((lx-ipx)**2+(ly-ipy)**2) < 20) return i;
  }
  return -1;
}

function renderBoxList() {
  const div = document.getElementById('boxlist');
  div.innerHTML = '';
  boxes.forEach((b, i) => {
    const item = document.createElement('div');
    item.className = 'box-item box-' + b.cls;
    if (i === selectedIdx) item.style.outline = '1px solid yellow';
    item.innerHTML =
      '<span style="color:' + clsColor(b.cls) + '">#' + i + '</span>' +
      '<select onchange="boxes[' + i + '].cls=this.value;renderBoxList();drawOverlay()">' +
        '<option value="vehicle"'    + (b.cls==='vehicle'?    ' selected':'') + '>vehicle</option>' +
        '<option value="pedestrian"' + (b.cls==='pedestrian'? ' selected':'') + '>pedestrian</option>' +
      '</select>' +
      ' cx:<input type="number" value="' + b.cx.toFixed(1) + '" step="0.5" style="width:52px" onchange="boxes[' + i + '].cx=parseFloat(this.value);drawOverlay()">' +
      ' cy:<input type="number" value="' + b.cy.toFixed(1) + '" step="0.5" style="width:52px" onchange="boxes[' + i + '].cy=parseFloat(this.value);drawOverlay()">' +
      ' l:<input  type="number" value="' + b.l.toFixed(1)  + '" step="0.2" style="width:46px" onchange="boxes[' + i + '].l=parseFloat(this.value);drawOverlay()">' +
      ' w:<input  type="number" value="' + b.w.toFixed(1)  + '" step="0.2" style="width:46px" onchange="boxes[' + i + '].w=parseFloat(this.value);drawOverlay()">' +
      ' h:<input  type="number" value="' + b.h.toFixed(1)  + '" step="0.1" style="width:46px" onchange="boxes[' + i + '].h=parseFloat(this.value)">' +
      ' yaw:<input type="number" value="' + b.yaw.toFixed(2) + '" step="0.05" style="width:52px" onchange="boxes[' + i + '].yaw=parseFloat(this.value);drawOverlay()">' +
      ' <button onclick="selectedIdx=' + i + ';renderBoxList();drawOverlay()">选</button>' +
      ' <button class="danger" onclick="boxes.splice(' + i + ',1);selectedIdx=-1;renderBoxList();drawOverlay()">x</button>';
    div.appendChild(item);
  });
}

function saveAnno() {
  fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx: frameIdx, boxes: boxes})
  }).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = 'Saved ' + d.path + ' (' + boxes.length + ' boxes)';
    img.src = '/api/bev?idx=' + frameIdx + '&t=' + Date.now();
  });
}

function clearAll() {
  if (!confirm('清空当前帧所有框？')) return;
  boxes = []; selectedIdx = -1; renderBoxList(); drawOverlay();
}

function prevFrame() { if (frameIdx > 0) { saveQuiet(); loadFrame(frameIdx-1); } }
function nextFrame() { if (frameIdx < totalFrames-1) { saveQuiet(); loadFrame(frameIdx+1); } }
function saveQuiet() {
  if (boxes.length > 0)
    fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({idx: frameIdx, boxes: boxes})});
}

window.addEventListener('resize', () => drawOverlay());
loadFrame(0);
</script>
</body>
</html>
"""



class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())

        elif path == "/api/frame":
            idx  = int(params.get("idx", [0])[0])
            stem = bin_files[idx].stem
            boxes_data = load_anno(stem)
            data = json.dumps({
                "idx": idx,
                "stem": stem,
                "total": len(bin_files),
                "boxes": [{"cx": b[0], "cy": b[1], "cz": b[2],
                           "l": b[3], "w": b[4], "h": b[5],
                           "yaw": b[6], "cls": b[7]}
                          for b in boxes_data],
            }).encode()
            self._send(200, "application/json", data)

        elif path == "/api/bev":
            idx  = int(params.get("idx", [0])[0])
            xmin = float(params.get("xmin", [PCR[0]])[0])
            xmax = float(params.get("xmax", [PCR[3]])[0])
            ymin = float(params.get("ymin", [PCR[1]])[0])
            ymax = float(params.get("ymax", [PCR[4]])[0])
            stem = bin_files[idx].stem
            boxes_data = load_anno(stem)
            png = make_bev_png(bin_files[idx],
                               [[b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]
                                for b in boxes_data],
                               x_range=(xmin, xmax), y_range=(ymin, ymax))
            self._send(200, "image/png", png)

        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            idx    = body["idx"]
            raw    = body["boxes"]
            stem   = bin_files[idx].stem
            boxes  = [[b["cx"], b["cy"], b["cz"],
                       b["l"],  b["w"],  b["h"],
                       b["yaw"], b["cls"]]
                      for b in raw]
            save_anno(stem, boxes)
            anno_path = ANNO_DIR / f"{stem}.txt"
            resp = json.dumps({"ok": True, "path": str(anno_path)}).encode()
            self._send(200, "application/json", resp)
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    if not bin_files:
        print(f"未找到 .bin 文件：{LIDAR_DIR}")
        print("请先运行 02_bag_to_bin.py 提取点云。")
        return

    ANNO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"找到 {len(bin_files)} 帧点云")
    print(f"标注保存目录：{ANNO_DIR}")
    print(f"\n启动标注服务器...")
    print(f"  → 请在浏览器打开：http://localhost:{PORT}")
    print(f"  → 按 Ctrl+C 退出\n")

    # 延迟自动打开浏览器
    def open_browser():
        import time
        time.sleep(1.0)
        webbrowser.open(f"http://localhost:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止。")


if __name__ == "__main__":
    main()
