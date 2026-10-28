# 轨迹投影工具

将一段带时间顺序的相机位姿投影到当前帧图像上，用于把未来运动轨迹叠加到图像中。工具支持逐帧输出图片、合成视频，以及导出投影点的 CSV。代码只依赖 Python、NumPy 和 OpenCV，不依赖 ORB-SLAM3；轨迹文件可以来自 ORB-SLAM3 或其他能导出兼容位姿格式的系统。

## 设计动机

本工具最初是为大规模图像数据中的**可通行区域自动标注**而设计的，目标是降低人工逐帧标注成本，并为后续模型训练准备带有几何依据的监督信息。它把相机位姿序列中的未来轨迹投影回当前图像，提供与运动方向和空间位置对应的图像线索。

当前实现负责位姿变换、相机模型投影，并导出轨迹叠加图和投影点 CSV。它提供的是可通行区域标注的轨迹线索；若训练任务需要完整的像素级区域掩码，还需在下游标注流程中根据任务定义将这些线索扩展为区域标签。

## 算法概述

输入是按时间排序的相机位姿 `T_w_c[k]`（相机坐标到世界坐标的变换）、相机标定和当前图像。对于当前帧 `i` 与未来帧 `j`，先将未来帧中的采样点变换到当前相机坐标系：

```text
X_c_i = inverse(T_w_c[i]) * T_w_c[j] * X_c_j
```

`--point center` 时，`X_c_j` 是未来帧的相机中心；`--point ground` 时，在相机局部坐标系中由相机中心指向地面参考点的 `--ground-axis` 方向上偏移 `--camera-height` 米，形成一个地面参考点。之后根据相机模型将三维点投影到当前图像：KITTI 使用标定文件中的投影矩阵，鱼眼相机使用 OpenCV fisheye（Kannala–Brandt 8 参数）模型。位于相机后方或图像范围外的点不会绘制。

默认地面方向沿用当前算法使用的 `(0.037309105, 0.998795769, 0.031859748)`，并会归一化。它是**相机局部坐标系中由相机中心指向参考点的方向**，需要按数据集的坐标约定确认；`--camera-height` 是沿该方向的偏移量，不是由程序估计的路面高度。

## 使用的相机

本项目当前的采集与标定配置使用 **FaysTech FS-VI-S80M** 双目视觉惯性相机。对应配置中的相机参数为：

- 左右双目相机，并配有 IMU。
- ORB-SLAM3 配置使用 Kannala–Brandt 8 参数鱼眼模型（`KannalaBrandt8`）。
- 当前配置分辨率为每目 `1280 × 800`，帧率为 `44 FPS`。
- 左右相机各自有内参和畸变参数，另有双目及相机到 IMU 的外参。

以上分辨率和帧率描述的是本项目使用的配置模式。仓库不附带设备专用标定文件；运行时应提供与实际分辨率、镜头和传感器坐标系相匹配的标定参数。

## 演示视频

在浏览器中直接播放：[打开 GitHub Pages 演示页](https://xuanwang0.github.io/trajectory-projection/)。

以下视频取自数据序列 `20260609_03` 的同一段 cam0 关键帧片段，用于对比原始画面、未来轨迹投影和可通行区域掩码：

- [原始画面](media/20260609_03_raw_excerpt.mp4)
- [未来轨迹投影](media/20260609_03_trajectory_excerpt.mp4)
- [轨迹与可通行区域掩码](media/20260609_03_trajectory_mask_excerpt.mp4)：紫色为掩码，黄橙色为投影轨迹。

每段约 24 秒，包含 120 个对齐关键帧，编码尺寸为 960 × 600、5 FPS。原始数据为 1280 × 800、44 FPS；这些仓库视频是便于查看和下载的展示片段，不是完整数据序列。掩码取自该序列已有的 `labeled/masks` 文件。

## 目录

```text
trajectory_projection/
├── README.md
├── index.html                              # GitHub Pages 视频播放页
├── requirements.txt
├── .gitignore
├── project_future_trajectory.py          # 单张 KITTI 图像投影
└── project_future_trajectory_sequence.py # KITTI / TUM 双目序列投影
```

## 部署

需要 Python 3.9 或更新版本。建议使用虚拟环境：

```bash
cd trajectory_projection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。如果环境没有图形界面，可将依赖替换为 `opencv-python-headless`。

## 输入数据格式

### KITTI Odometry

```text
KITTI_ROOT/
├── sequences/00/
│   ├── calib.txt
│   ├── image_0/000000.png ...
│   └── image_1/000000.png ...
└── poses/00.txt
```

位姿文件每行 12 个数，按行优先表示一个 `3x4` 变换矩阵；标定文件需包含对应图像的 `P0` 或 `P1` 投影矩阵。单帧脚本目前只支持这种 KITTI 输入。

### TUM 风格双目鱼眼序列

```text
SEQUENCE/
├── mav0/cam0/data/<timestamp>.jpg   # 也接受 .png
├── mav0/cam1/data/<timestamp>.jpg   # 可选
├── CameraTrajectory.txt             # 或用 --traj 指定其他文件
└── stereo_inertial.yaml             # 或用 --calib-yaml 指定其他文件
```

轨迹文件每行至少 8 列：`timestamp tx ty tz qx qy qz qw`，四元数顺序为 `qx qy qz qw`。图片文件名（去掉扩展名）应是纳秒时间戳；程序将每个轨迹时间戳匹配到时间最近的图像。标定 YAML 需有 `Camera.width`、`Camera.height` 和 `Camera1`（cam0）或 `Camera2`（cam1）的 `fx/fy/cx/cy/k1/k2/k3/k4` 参数。若轨迹是 IMU/body 位姿且 YAML 包含 `IMU.T_b_c1`，默认 `--pose-frame auto` 会应用该外参；若轨迹已经是相机位姿，使用 `--pose-frame camera`。

当前 `cam1` 路径没有独立的 body-to-cam1 外参转换，脚本会提示；需要严格投影到右相机时，应补充右相机外参。建议先使用 `cam0`。

## 使用

### TUM 双目鱼眼：整段序列

```bash
python project_future_trajectory_sequence.py \
  --format tum-stereo \
  --dataset /data/my_sequence \
  --traj /data/my_sequence/KeyFrameTrajectory.txt \
  --calib-yaml /data/my_sequence/stereo_inertial.yaml \
  --image cam0 \
  --point ground --camera-height 1.7 \
  --future 24 --stride 3 \
  --fps 4 \
  --output-dir outputs/frames \
  --video outputs/future_trajectory.mp4 \
  --traj-out outputs/projected_points.csv
```

`--future 24 --stride 3` 表示每个当前帧最多取后续 24 帧，并每隔 3 帧采样一个点。`--frame-step 2` 可每两帧生成一张叠加图；`--start-frame`、`--end-frame` 可限制处理区间。若只需要视频或 CSV，可分别加 `--no-save-frames` 或 `--no-video`。

### KITTI：整段序列

```bash
python project_future_trajectory_sequence.py \
  --format kitti --dataset /data/KITTI \
  --sequence 00 --image image_0 \
  --point ground --camera-height 1.7 --ground-axis 0 1 0 \
  --future 60 --stride 2 \
  --output-dir outputs/kitti00_frames \
  --video outputs/kitti00.mp4
```

KITTI 输入为针孔投影矩阵，`--ground-axis` 默认沿用本项目已有值；上例显式使用 KITTI 相机坐标的 `+Y`（向下）作为指向地面的方向。应根据具体位姿定义和相机坐标约定确认方向与符号。

### KITTI：单张图像

```bash
python project_future_trajectory.py \
  --dataset /data/KITTI --sequence 00 --frame 100 \
  --image image_0 --point ground --camera-height 1.7 \
  --ground-axis 0 1 0 --future 120 --stride 2 \
  --output outputs/kitti00_frame100.png
```

不指定 `--output` 时，单帧结果写入 `outputs/future_trajectory/`。整段序列脚本未指定输出路径时，会在 `outputs/future_trajectory_sequence/` 下创建图片、视频和 CSV。

## 输出

- **叠加图像/视频**：当前图像上绘制未来采样点和连线，颜色从近处偏黄渐变到远处偏红。
- **CSV**：包含当前帧、未来帧、图像文件名、世界坐标、当前相机坐标、像素坐标和是否落在图像内等字段。

## 限制

- 本工具只对输入位姿做几何投影，不估计未来轨迹，也不做障碍物检测或路径规划。
- KITTI 序列图像目前按 PNG 文件读取；TUM 风格序列接受 JPG 和 PNG。
- 鱼眼投影依赖 OpenCV fisheye 模型与 YAML 中的 Kannala–Brandt 参数。
- 轨迹与图像时间戳、相机坐标系、单位和标定必须一致；时间匹配只取最近时间戳，不执行插值。
- 默认输出视频使用 OpenCV 可用的视频编码器；若系统缺少 MP4 编码器，可尝试 `.avi` 输出或仅保存图片。
