#!/usr/bin/env python
"""Render future-trajectory overlays for a full image sequence.

Supports two dataset layouts:
  - kitti        : KITTI Odometry (poses as 12-value rows, calib.txt with P0/P1).
  - tum-stereo   : a TUM-style stereo sequence with
                     mav0/cam0/data/<timestamp>.jpg images,
                     CameraTrajectory.txt (`timestamp tx ty tz qx qy qz qw`),
                     stereo_inertial.yaml (KannalaBrandt8 fisheye intrinsics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from project_future_trajectory import (  # noqa: E402
    draw_trajectory,
    future_points_in_current_camera,
    project_points,
    read_calib,
    read_poses,
)


# ---------- shared helpers ----------

def count_visible(image_shape, pixels, valid):
    height, width = image_shape[:2]
    in_image = (
        valid
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return int(in_image.sum())


def open_video_writer(video_path, fps, frame_size):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix.lower()
    codec_candidates = ["mp4v", "avc1", "XVID"] if suffix == ".mp4" else ["XVID", "MJPG", "mp4v"]
    for codec in codec_candidates:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to open video writer: {video_path}")


# ---------- TUM-style stereo loaders ----------

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


def read_poses_tum(traj_path: Path):
    """Read trajectory as 4x4 transforms T_w_c.

    Format: timestamp tx ty tz qx qy qz qw (camera-from-world poses written by ORB-SLAM3
    are stored as world-from-camera here; we treat them consistently as T_w_c so the
    inverse used downstream maps a world point into the camera frame).
    """
    raw = np.loadtxt(str(traj_path), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 8:
        raise ValueError(
            f"Expected 8 columns (ts tx ty tz qx qy qz qw) in {traj_path}, got {raw.shape[1]}"
        )
    timestamps = raw[:, 0]
    trans = raw[:, 1:4]
    quats = raw[:, 4:8]  # qx qy qz qw

    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], raw.shape[0], axis=0)
    for i in range(raw.shape[0]):
        R = _quat_to_rot(*quats[i])
        transforms[i, :3, :3] = R
        transforms[i, :3, 3] = trans[i]
    return transforms, timestamps


def _yaml_get_float(text: str, key: str) -> float:
    import re
    m = re.search(rf"^{re.escape(key)}\s*:\s*([-\d.eE+]+)", text, flags=re.MULTILINE)
    if not m:
        raise KeyError(f"Key {key} not found in YAML")
    return float(m.group(1))


def _yaml_get_matrix(text: str, key: str):
    """Parse an !!opencv-matrix block under `key` and return (rows, cols, np.ndarray)."""
    import re
    pattern = re.compile(
        rf"^{re.escape(key)}\s*:\s*!!opencv-matrix\s*\n"
        r"\s*rows:\s*(\d+)\s*\n"
        r"\s*cols:\s*(\d+)\s*\n"
        r"\s*dt:\s*[a-zA-Z0-9_]+\s*\n"
        r"\s*data:\s*\[([^\]]*)\]",
        flags=re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return None
    rows = int(m.group(1))
    cols = int(m.group(2))
    data = np.fromstring(m.group(3), sep=",", dtype=np.float64)
    if data.size != rows * cols:
        raise ValueError(f"Matrix {key} expected {rows*cols} values, got {data.size}")
    return rows, cols, data.reshape(rows, cols)


def load_T_b_c(yaml_path: Path):
    """Return T_b_c1 (body-from-camera0) if present in the YAML, else None."""
    text = yaml_path.read_text(encoding="utf-8")
    parsed = _yaml_get_matrix(text, "IMU.T_b_c1")
    if parsed is None:
        return None
    rows, cols, M = parsed
    if (rows, cols) != (4, 4):
        raise ValueError(f"IMU.T_b_c1 must be 4x4, got {rows}x{cols}")
    return M


def load_intrinsics_kb8(yaml_path: Path, cam: str = "cam0"):
    """Load KannalaBrandt8 fisheye intrinsics from stereo_inertial.yaml.

    Returns dict with K (3x3), D (4,) (k1..k4), width, height.
    """
    text = yaml_path.read_text(encoding="utf-8")
    prefix = "Camera1" if cam == "cam0" else "Camera2"
    fx = _yaml_get_float(text, f"{prefix}.fx")
    fy = _yaml_get_float(text, f"{prefix}.fy")
    cx = _yaml_get_float(text, f"{prefix}.cx")
    cy = _yaml_get_float(text, f"{prefix}.cy")
    k1 = _yaml_get_float(text, f"{prefix}.k1")
    k2 = _yaml_get_float(text, f"{prefix}.k2")
    k3 = _yaml_get_float(text, f"{prefix}.k3")
    k4 = _yaml_get_float(text, f"{prefix}.k4")
    width = int(_yaml_get_float(text, "Camera.width"))
    height = int(_yaml_get_float(text, "Camera.height"))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    return {"K": K, "D": D, "width": width, "height": height}


def project_points_fisheye(intr, points_cam):
    """Project 3D points (camera frame) through KannalaBrandt8 / OpenCV fisheye.

    Returns pixels (N,2) and valid (N,) boolean mask.
    """
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3)
    valid = points_cam[:, 2] > 1e-6
    pixels = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        proj, _ = cv2.fisheye.projectPoints(pts[valid], rvec, tvec, intr["K"], intr["D"])
        pixels[valid] = proj.reshape(-1, 2)
    return pixels, valid


def match_image_paths_to_poses(image_dir: Path, timestamps_ns: np.ndarray, exts=(".jpg", ".png")):
    """For each pose timestamp (ns), find the image file with the closest timestamp."""
    files = []
    for ext in exts:
        files.extend(image_dir.glob(f"*{ext}"))
    if not files:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    file_ts = np.array([float(p.stem) for p in files], dtype=np.float64)
    order = np.argsort(file_ts)
    file_ts = file_ts[order]
    files = [files[i] for i in order]

    matched = []
    matched_ts = []
    for ts in timestamps_ns:
        idx = int(np.argmin(np.abs(file_ts - ts)))
        matched.append(files[idx])
        matched_ts.append(file_ts[idx])
    return matched, np.asarray(matched_ts, dtype=np.float64)


# ---------- argparse ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project future trajectory for every frame and save an overlay video."
    )
    parser.add_argument(
        "--format",
        choices=["kitti", "tum-stereo"],
        default="tum-stereo",
        help="Dataset layout: KITTI Odometry or TUM-style stereo sequence.",
    )
    # KITTI options
    parser.add_argument("--dataset", type=Path, required=True,
                        help="KITTI: dataset dir; tum-stereo: sequence dir (containing mav0/, CameraTrajectory.txt).")
    parser.add_argument("--sequence", default="00", help="KITTI sequence id, for example 00.")
    parser.add_argument("--image", default="image_0",
                        help="KITTI: image_0|image_1; tum-stereo: cam0|cam1 (default cam0).")
    # tum-stereo options
    parser.add_argument("--traj", type=Path, default=None,
                        help="tum-stereo: trajectory file. Default: <dataset>/CameraTrajectory.txt")
    parser.add_argument("--calib-yaml", type=Path, default=None,
                        help="tum-stereo: calibration YAML path. Default: <dataset>/stereo_inertial.yaml")
    parser.add_argument("--pose-frame", choices=["auto", "body", "camera"], default="auto",
                        help="Frame of poses in the trajectory file. ORB-SLAM3 IMU runs save body (IMU) poses; "
                             "'auto' applies T_b_c1 from the YAML if present, otherwise treats them as camera poses.")
    # Common projection options
    parser.add_argument("--future", type=int, default=60, help="Number of future frames to draw per current frame.")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth future frame.")
    parser.add_argument("--point", choices=["ground", "center"], default="ground",
                        help="Project ground footpoints or camera centers.")
    parser.add_argument("--camera-height", type=float, default=1.65,
                        help="Camera height in meters for --point ground.")
    parser.add_argument("--ground-axis", type=float, nargs=3, default=(0.037309105, 0.998795769, 0.031859748),
                        metavar=("X", "Y", "Z"),
                        help="Direction from camera center toward ground reference point in local camera frame.")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--label-every", type=int, default=0,
                        help="Label every Nth visible point; 0 disables labels.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frame rate.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for overlay images. Default: outputs/future_trajectory_sequence/<seq>/<run>")
    parser.add_argument("--video", type=Path, default=None,
                        help="Output video path. Default: outputs/future_trajectory_sequence/<seq>/<run>.mp4")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--traj-out", type=Path, default=None,
                        help="Path of the trajectory CSV. Default: outputs/future_trajectory_sequence/<seq>/<run>_trajectory.csv. "
                             "Set to '-' or pass --no-traj to disable.")
    parser.add_argument("--no-traj", action="store_true",
                        help="Skip writing the trajectory CSV file.")
    return parser.parse_args()


def _setup_kitti(args):
    seq_dir = args.dataset / "sequences" / args.sequence
    image_dir = seq_dir / args.image
    poses_path = args.dataset / "poses" / f"{args.sequence}.txt"
    calib_path = seq_dir / "calib.txt"
    for p, name in [(image_dir, "Image directory"), (poses_path, "Pose file"), (calib_path, "Calibration file")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {image_dir}")
    poses = read_poses(poses_path)
    calib = read_calib(calib_path)
    projection_key = "P0" if args.image == "image_0" else "P1"
    P = calib[projection_key]
    seq_label = args.sequence

    def projector(points_cam):
        return project_points(P, points_cam)

    return image_paths, poses, projector, seq_label


def _setup_tum_stereo(args):
    base = args.dataset
    cam = args.image if args.image in ("cam0", "cam1") else "cam0"
    image_dir = base / "mav0" / cam / "data"
    traj_path = args.traj or (base / "CameraTrajectory.txt")
    yaml_path = args.calib_yaml or (base / "stereo_inertial.yaml")
    for p, name in [(image_dir, "Image directory"), (traj_path, "Trajectory file"), (yaml_path, "YAML calibration")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    poses, timestamps_ns = read_poses_tum(traj_path)
    image_paths, _ = match_image_paths_to_poses(image_dir, timestamps_ns)
    intr = load_intrinsics_kb8(yaml_path, cam=cam)

    # Convert body (IMU) poses to camera poses if needed: T_w_c = T_w_b * T_b_c
    T_b_c = load_T_b_c(yaml_path)
    if args.pose_frame == "camera":
        T_b_c = None  # don't apply
    elif args.pose_frame == "body" and T_b_c is None:
        raise RuntimeError(
            f"--pose-frame=body but IMU.T_b_c1 not found in {yaml_path}; cannot transform body poses to camera frame."
        )
    if T_b_c is not None:
        if cam == "cam1":
            print("[warn] --image cam1 is not directly supported by IMU.T_b_c1; "
                  "trajectory will be projected on cam0 frame and rendered on cam1 image.", flush=True)
        poses = poses @ T_b_c[None, :, :]  # broadcast over N

    seq_label = base.name

    def projector(points_cam):
        return project_points_fisheye(intr, points_cam)

    return image_paths, poses, projector, seq_label


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.future < 1:
        raise ValueError("--future must be >= 1")
    if args.no_save_frames and args.no_video and args.no_traj:
        raise ValueError(
            "All outputs disabled (--no-save-frames + --no-video + --no-traj); nothing would be written."
        )

    if args.format == "kitti":
        image_paths, poses, projector, seq_label = _setup_kitti(args)
    else:
        image_paths, poses, projector, seq_label = _setup_tum_stereo(args)

    frame_count = min(len(image_paths), len(poses))
    if args.start_frame < 0 or args.start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}], got {args.start_frame}")
    end_frame = frame_count - 1 if args.end_frame is None else min(args.end_frame, frame_count - 1)
    if end_frame < args.start_frame:
        raise ValueError("--end-frame must be >= --start-frame")

    run_name = f"{args.image}_{args.point}_future{args.future}_stride{args.stride}"
    output_dir = args.output_dir or Path("outputs") / "future_trajectory_sequence" / seq_label / run_name
    video_path = args.video or Path("outputs") / "future_trajectory_sequence" / seq_label / f"{run_name}.mp4"
    save_traj = not args.no_traj
    traj_path = None
    if save_traj:
        traj_path = args.traj_out or (
            Path("outputs") / "future_trajectory_sequence" / seq_label / f"{run_name}_trajectory.csv"
        )

    sample = cv2.imread(str(image_paths[args.start_frame]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_paths[args.start_frame]}")
    height, width = sample.shape[:2]
    writer = None if args.no_video else open_video_writer(video_path, args.fps, (width, height))

    if not args.no_save_frames:
        output_dir.mkdir(parents=True, exist_ok=True)

    current_frames = list(range(args.start_frame, end_frame + 1, args.frame_step))
    total_visible = 0
    written_frames = 0

    traj_file = None
    if traj_path is not None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_file = traj_path.open("w", encoding="utf-8")
        traj_file.write(
            "current_frame,future_frame,current_image,future_image,world_x,world_y,world_z,"
            "cam_x,cam_y,cam_z,pixel_u,pixel_v,in_image\n"
        )

    try:
        for n, frame in enumerate(current_frames, start=1):
            image_path = image_paths[frame]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV failed to read image: {image_path}")
            if image.shape[:2] != (height, width):
                raise RuntimeError(f"Frame size changed at {image_path}: expected {(width, height)}")

            future_frames = list(range(frame + 1, min(frame + args.future + 1, frame_count), args.stride))
            if future_frames:
                points_cam, frame_ids = future_points_in_current_camera(
                    poses=poses,
                    current_frame=frame,
                    future_frames=future_frames,
                    point_mode=args.point,
                    camera_height=args.camera_height,
                    ground_axis=args.ground_axis,
                )
                pixels, valid = projector(points_cam)
                overlay = draw_trajectory(
                    image=image,
                    pixels=pixels,
                    valid=valid,
                    frame_ids=frame_ids,
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius,
                    label_every=args.label_every,
                )
                total_visible += count_visible(overlay.shape, pixels, valid)

                if traj_file is not None:
                    h_img, w_img = overlay.shape[:2]
                    T_w_c = poses[frame]
                    for k, fid in enumerate(frame_ids):
                        cam_xyz = points_cam[k]
                        # World point corresponding to the projected sample (camera center or ground footprint)
                        world_xyz = (T_w_c @ np.r_[cam_xyz, 1.0])[:3]
                        u, v = pixels[k]
                        in_img = bool(
                            valid[k]
                            and not np.isnan(u) and not np.isnan(v)
                            and 0 <= u < w_img and 0 <= v < h_img
                        )
                        u_str = "" if np.isnan(u) else f"{u:.4f}"
                        v_str = "" if np.isnan(v) else f"{v:.4f}"
                        traj_file.write(
                            f"{frame},{int(fid)},"
                            f"{image_paths[frame].name},{image_paths[fid].name},"
                            f"{world_xyz[0]:.6f},{world_xyz[1]:.6f},{world_xyz[2]:.6f},"
                            f"{cam_xyz[0]:.6f},{cam_xyz[1]:.6f},{cam_xyz[2]:.6f},"
                            f"{u_str},{v_str},{int(in_img)}\n"
                        )
            else:
                overlay = image

            if not args.no_save_frames:
                out_path = output_dir / f"{frame:06d}.png"
                if not cv2.imwrite(str(out_path), overlay):
                    raise RuntimeError(f"OpenCV failed to write frame: {out_path}")

            if writer is not None:
                writer.write(overlay)

            written_frames += 1
            if n == 1 or n == len(current_frames) or n % 100 == 0:
                print(f"[{n}/{len(current_frames)}] frame={frame:06d} -> {image_path.name}")
    finally:
        if writer is not None:
            writer.release()
        if traj_file is not None:
            traj_file.close()

    if not args.no_save_frames:
        print(f"Saved overlay frames: {output_dir}")
    if not args.no_video:
        print(f"Saved video: {video_path}")
    if traj_path is not None:
        print(f"Saved trajectory CSV: {traj_path}")
    print(f"Processed frames: {written_frames}")
    print(f"Total visible projected points: {total_visible}")


if __name__ == "__main__":
    main()
