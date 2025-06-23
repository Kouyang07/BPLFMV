#!/usr/bin/env python3
"""
Pose Estimation with Court CSV Input

Reads court.csv directly and outputs pose.json.
This is the bridge script that converts from CSV to JSON format.
Now includes court boundary enlargement for more inclusive pose detection.
"""

import cv2
import torch
import numpy as np
import json
import sys
import os
import csv
import logging
from tqdm import tqdm
from ultralytics import YOLO
import warnings

# Suppress warnings and YOLO output
warnings.filterwarnings("ignore")
os.environ['YOLO_VERBOSE'] = 'False'

# Set ultralytics logging level to ERROR to suppress info messages
logging.getLogger('ultralytics').setLevel(logging.ERROR)


def read_court_csv(csv_path):
    """Read court coordinates from CSV file."""
    court_points = {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Court CSV not found: {csv_path}")

    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            point_name = row['Point']
            x_coord = float(row['X'])
            y_coord = float(row['Y'])
            court_points[point_name] = [x_coord, y_coord]

    print(f"Loaded {len(court_points)} court points from CSV")
    return court_points


def extract_corner_points(all_court_points):
    """Extract the 4 main corner points (P1-P4) for court boundary."""
    corner_points = {}

    # Try to get P1-P4 first
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_court_points:
            corner_points[point_name] = all_court_points[point_name]

    if len(corner_points) == 4:
        return corner_points

    # Fallback: use first 4 points if P1-P4 not available
    print("Warning: P1-P4 not found, using first 4 points as corners")
    point_items = list(all_court_points.items())[:4]
    return {f"P{i+1}": coords for i, (_, coords) in enumerate(point_items)}


def enlarge_court_boundary(corner_points, enlargement_factor=0.4):
    """Enlarge court boundary by a given factor (default 40%)."""
    if len(corner_points) != 4:
        return corner_points

    # Get original corner points
    points = [corner_points[f"P{i}"] for i in range(1, 5) if f"P{i}" in corner_points]
    if len(points) != 4:
        return corner_points

    # Calculate center of the court
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4

    # Enlarge each point by moving it away from the center
    enlarged_points = {}
    for i, point in enumerate(points):
        # Vector from center to point
        dx = point[0] - center_x
        dy = point[1] - center_y

        # Enlarge by moving further from center
        new_x = center_x + dx * (1 + enlargement_factor)
        new_y = center_y + dy * (1 + enlargement_factor)

        enlarged_points[f"P{i+1}"] = [new_x, new_y]

    return enlarged_points


def is_point_in_court(point, corner_points, enlarged_boundary=None):
    """Check if a point is inside the court polygon using corner points."""
    x, y = point

    # Use enlarged boundary if provided, otherwise use original
    boundary_points = enlarged_boundary if enlarged_boundary is not None else corner_points

    # Create polygon from corner points
    polygon = [boundary_points[f"P{i}"] for i in range(1, 5) if f"P{i}" in boundary_points]

    if len(polygon) != 4:
        print(f"Warning: Only {len(polygon)} corner points available for court boundary")
        return True  # Allow all points if boundary detection failed

    # Point-in-polygon algorithm
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_video_info(video_path):
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count
    }


class PoseDetect:
    def __init__(self):
        self.device = self.select_device()
        self.setup_YOLO()

    def select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def setup_YOLO(self):
        logging.info(f"Loading YOLO model on device: {self.device}")

        # Look for YOLO model in common locations
        model_paths = [
            'samples/yolo11x-pose.pt',
            'yolo11x-pose.pt',
            './yolo11x-pose.pt'
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found YOLO model: {path}")
                break

        if model_path is None:
            print("YOLO model not found locally, downloading...")
            model_path = 'yolo11x-pose.pt'

        # Suppress YOLO output during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Redirect stdout temporarily to suppress YOLO prints
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.__pose_model = YOLO(model_path, verbose=False)
                self.__pose_model.to(self.device)

    def get_human_joints(self, frame):
        # Suppress output during inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.__pose_model(frame, verbose=False)
        return results

    def process_video(self, video_path, court_csv_path, output_json_path):
        """Process video for pose estimation with enlarged court boundary."""
        logging.info(f"Processing video: {video_path}")

        # Load court data from CSV
        all_court_points = read_court_csv(court_csv_path)
        corner_points = extract_corner_points(all_court_points)

        # Create enlarged court boundary (40% bigger)
        enlarged_court = enlarge_court_boundary(corner_points, enlargement_factor=0.4)

        logging.info("Court boundary enlarged by 40% for more inclusive pose detection")

        # Get video info
        video_info = get_video_info(video_path)

        # Assume all frames are rally frames for now
        # You could implement more sophisticated rally detection here
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"Processing {len(rally_frames)} frames...")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        pose_data = []

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in rally_frames:
                    results = self.get_human_joints(frame)

                    if results[0].keypoints is not None:
                        keypoints = results[0].keypoints
                        for human_idx, person_keypoints in enumerate(keypoints):
                            if person_keypoints.xy is None or person_keypoints.conf is None:
                                continue

                            xy = person_keypoints.xy[0]
                            conf = person_keypoints.conf[0]

                            # Check if any body part is within the enlarged court boundary
                            is_in_court = False
                            court_joints = []

                            for joint_idx in range(xy.shape[0]):
                                if conf[joint_idx].item() > 0.5:
                                    x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
                                    # Use enlarged court boundary for filtering
                                    if is_point_in_court((x, y), corner_points, enlarged_court):
                                        is_in_court = True
                                        court_joints.append(joint_idx)

                            if is_in_court:
                                # Store pose data
                                person_joints = []
                                for joint_idx in range(xy.shape[0]):
                                    # Check if this specific joint is in the enlarged court area
                                    joint_in_court = False
                                    if conf[joint_idx].item() > 0.5:
                                        x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
                                        joint_in_court = is_point_in_court((x, y), corner_points, enlarged_court)

                                    person_joints.append({
                                        "joint_index": joint_idx,
                                        "x": float(xy[joint_idx, 0].item()),
                                        "y": float(xy[joint_idx, 1].item()),
                                        "confidence": float(conf[joint_idx].item()),
                                        "in_court": joint_in_court
                                    })

                                pose_data.append({
                                    "frame_index": frame_index,
                                    "human_index": human_idx,
                                    "joints": person_joints,
                                    "joints_in_court": len(court_joints)
                                })

                frame_index += 1
                pbar.update(1)

        cap.release()

        # Create output JSON
        output_data = {
            "court_points": corner_points,           # P1-P4 for homography (original)
            "enlarged_court_points": enlarged_court, # P1-P4 enlarged by 40%
            "all_court_points": all_court_points,    # All detected points
            "video_info": video_info,
            "rally_frames": list(rally_frames),
            "pose_data": pose_data,
            "processing_info": {
                "total_poses_detected": len(pose_data),
                "total_court_points": len(all_court_points),
                "corner_points_used": list(corner_points.keys()),
                "court_enlargement_factor": 0.4,
                "inclusion_criteria": "At least one joint within enlarged court boundary",
                "model_device": self.device,
                "model_type": "YOLO11x-pose",
                "pose_estimator": "YOLO11x Pose Estimation",
                "accuracy_note": "Standard YOLO accuracy with enlarged court boundary"
            }
        }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info(f"Pose estimation completed!")
        logging.info(f"✓ Total poses detected: {len(pose_data)}")
        logging.info(f"✓ Court boundary enlarged by 40% for inclusive detection")
        logging.info(f"✓ Data saved to: {output_json_path}")

        return output_json_path


def main(video_path):
    """Main pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*60)
    print("YOLO Pose Estimation with Enlarged Court Boundary")
    print("="*60)
    print("Using YOLO11x-pose with 40% enlarged court boundary")
    print("More inclusive pose detection for better coverage")
    print("="*60)

    # Determine paths based on your structure
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    court_csv_path = os.path.join(result_dir, "court.csv")
    output_json_path = os.path.join(result_dir, "pose.json")

    # Check if court CSV exists
    if not os.path.exists(court_csv_path):
        logging.error(f"Court CSV not found: {court_csv_path}")
        logging.error("Please run preprocessing first: python3 preprocess.py <video_path>")
        return None

    # Process poses
    pose_detect = PoseDetect()
    try:
        output_path = pose_detect.process_video(video_path, court_csv_path, output_json_path)
        return output_path
    finally:
        # Clean up YOLO model
        del pose_detect


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_pose.py <input_video_path>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    main(input_video_path)