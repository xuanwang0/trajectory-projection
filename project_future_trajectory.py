#!/usr/bin/env python
"""Project future KITTI Odometry poses onto the current image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_calib(calib_path: Path) -> dict[str, np.ndarray]:
    calib: dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, values = line.split(":", 1)
            calib[key] = np.fromstring(values, sep=" ", dtype=np.float64).reshape(3, 4)
    return calib


def read_poses(poses_path: Path) -> np.ndarray:
    raw = np.loadtxt(str(poses_path), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], raw.shape[0], axis=0)
    transforms[:, :3, :] = raw.reshape(-1, 3, 4)
    return transforms


def project_points(P: np.ndarray, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.c_[points_cam, np.ones(len(points_cam), dtype=np.float64)]
    pixels_h = (P @ homogeneous.T).T
    valid = (points_cam[:, 2] > 1e-6) & (pixels_h[:, 2] > 1e-6)
    pixels = np.empty((len(points_cam), 2), dtype=np.float64)
    pixels[:] = np.nan
    pixels[valid] = pixels_h[valid, :2] / pixels_h[valid, 2:3]
    return pixels, valid


def future_points_in_current_camera(
    poses: np.ndarray,
    current_frame: int,
    future_frames: Iterable[int],
    point_mode: str,
    camera_height: float,
    ground_axis: Iterable[float] = (0.037309105, 0.998795769, 0.031859748),
) -> tuple[np.ndarray, np.ndarray]:
    T_0_i = poses[current_frame]
    T_i_0 = np.linalg.inv(T_0_i)
    if point_mode == "ground":
        axis = np.asarray(ground_axis, dtype=np.float64).reshape(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-12:
            raise ValueError("ground_axis must be a non-zero 3D vector")
        offset = camera_height * axis / axis_norm
    elif point_mode == "center":
        offset = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported point_mode: {point_mode}")
    local_point = np.r_[offset, 1.0]

    points = []
    frame_ids = []
    for j in future_frames:
        T_i_j = T_i_0 @ poses[j]
        X_i = T_i_j @ local_point
        points.append(X_i[:3])
        frame_ids.append(j)
    return np.asarray(points, dtype=np.float64), np.asarray(frame_ids, dtype=np.int32)


def color_for_index(index: int, count: int) -> tuple[int, int, int]:
    if count <= 1:
        return (0, 255, 255)
    alpha = index / (count - 1)
    # BGR: near future yellow, far future red.
    return (0, int(round(255 * (1.0 - 0.6 * alpha))), 255)


def draw_trajectory(
    image: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray,
    frame_ids: np.ndarray,
    line_thickness: int,
    point_radius: int,
    label_every: int,
) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    in_image = (
        valid
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    drawable = [(idx, tuple(np.rint(pixels[idx]).astype(int))) for idx in np.where(in_image)[0]]

    for pair_idx, ((idx_a, pt_a), (idx_b, pt_b)) in enumerate(zip(drawable[:-1], drawable[1:])):
        color = color_for_index(pair_idx, max(len(drawable) - 1, 1))
        cv2.line(out, pt_a, pt_b, color, line_thickness, cv2.LINE_AA)

    for draw_idx, (idx, pt) in enumerate(drawable):
        color = color_for_index(draw_idx, max(len(drawable), 1))
        cv2.circle(out, pt, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, pt, point_radius, color, -1, cv2.LINE_AA)
        if label_every > 0 and (draw_idx == 0 or draw_idx % label_every == 0):
            cv2.putText(
                out,
                str(int(frame_ids[idx])),
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                str(int(frame_ids[idx])),
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project future KITTI Odometry ground-truth trajectory onto a current frame."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="KITTI odometry dataset directory.")
    parser.add_argument("--sequence", default="00", help="Sequence id, for example 00.")
    parser.add_argument("--frame", type=int, default=100, help="Current frame index.")
    parser.add_argument("--future", type=int, default=120, help="Number of future frames to draw.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth future frame.")
    parser.add_argument("--image", choices=["image_0", "image_1"], default="image_0", help="Image folder to draw on.")
    parser.add_argument("--point", choices=["ground", "center"], default="ground", help="Project ground footpoints or camera centers.")
    parser.add_argument("--camera-height", type=float, default=1.65, help="Camera height in meters for --point ground.")
    parser.add_argument("--ground-axis", type=float, nargs=3, default=(0.037309105, 0.998795769, 0.031859748),
                        metavar=("X", "Y", "Z"),
                        help="Direction from camera center toward ground reference point in local camera frame.")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--label-every", type=int, default=20, help="Label every Nth visible point; 0 disables labels.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_dir = args.dataset / "sequences" / args.sequence
    poses_path = args.dataset / "poses" / f"{args.sequence}.txt"
    image_path = seq_dir / args.image / f"{args.frame:06d}.png"
    calib_path = seq_dir / "calib.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Pose file not found: {poses_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")

    poses = read_poses(poses_path)
    if args.frame < 0 or args.frame >= len(poses):
        raise ValueError(f"--frame must be in [0, {len(poses) - 1}], got {args.frame}")

    start = args.frame + 1
    stop = min(args.frame + args.future + 1, len(poses))
    future_frames = range(start, stop, max(args.stride, 1))
    future_frames = list(future_frames)
    if not future_frames:
        raise ValueError("No future frames are available for the requested --frame/--future settings.")

    calib = read_calib(calib_path)
    projection_key = "P0" if args.image == "image_0" else "P1"
    P = calib[projection_key]

    points_cam, frame_ids = future_points_in_current_camera(
        poses=poses,
        current_frame=args.frame,
        future_frames=future_frames,
        point_mode=args.point,
        camera_height=args.camera_height,
        ground_axis=args.ground_axis,
    )
    pixels, valid = project_points(P, points_cam)
    out = draw_trajectory(
        image=image,
        pixels=pixels,
        valid=valid,
        frame_ids=frame_ids,
        line_thickness=args.line_thickness,
        point_radius=args.point_radius,
        label_every=args.label_every,
    )

    output = args.output
    if output is None:
        output = (
            Path("outputs")
            / "future_trajectory"
            / args.sequence
            / f"{args.image}_{args.frame:06d}_{args.point}_future{args.future}.png"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), out):
        raise RuntimeError(f"OpenCV failed to write output: {output}")

    height, width = image.shape[:2]
    in_image = (
        valid
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    print(f"Saved: {output}")
    print(f"Projected future frames: {len(frame_ids)}")
    print(f"Visible points in current image: {int(in_image.sum())}")


if __name__ == "__main__":
    main()
