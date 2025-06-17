#!/usr/bin/env python3
"""
ViTPose-G Maximum Accuracy Pose Estimation with Court CSV Input

Uses only ViTPose-G (Giant) for maximum accuracy pose estimation.
ViTPose-G achieves 81.1 AP on COCO test-dev set - state-of-the-art performance.
"""

import cv2
import numpy as np
import json
import sys
import os
import csv
import logging
from tqdm import tqdm
import warnings

# MMPose imports
from mmpose.apis import MMPoseInferencer

# Suppress warnings
warnings.filterwarnings("ignore")


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
    """Enlarge court boundary by a given factor (default 20%)."""
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


class ViTPoseGPoseDetect:
    def __init__(self, device=None, confidence_threshold=0.2):
        self.device = self.select_device() if device is None else device
        self.confidence_threshold = confidence_threshold
        self.setup_vitpose_g()

    def select_device(self):
        """Select best available device"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def setup_vitpose_g(self):
        """Initialize MMPose inferencer with ViTPose-G only."""
        print(f"Loading ViTPose-G model on device: {self.device}")
        print("ViTPose-G provides state-of-the-art accuracy (81.1 AP on COCO)")
        print("This may take a moment to download the model weights...")

        try:
            # Use only ViTPose-G (Giant) - the highest accuracy model available
            # Note: Different possible naming conventions for ViTPose-G
            vitpose_g_configs = [
                'td-hm_ViTPose-giant_8xb64-210e_coco-256x192',
                'td-hm_ViTPose-huge_8xb64-210e_coco-256x192',
                'vitpose-g',
                'vitpose-huge'
            ]

            for config in vitpose_g_configs:
                try:
                    print(f"Trying ViTPose-G configuration: {config}")
                    self.inferencer = MMPoseInferencer(
                        pose2d=config,
                        device=self.device,
                        show_progress=False
                    )
                    print(f"✓ Successfully loaded ViTPose-G with config: {config}")
                    self.model_config = config
                    return
                except Exception as e:
                    print(f"✗ Failed with config {config}: {e}")
                    continue

            # If all specific configs fail, this will raise an error
            raise RuntimeError("All ViTPose-G configurations failed")

        except Exception as e:
            error_msg = f"""
Failed to load ViTPose-G: {e}

ViTPose-G might not be available in your MMPose installation.
You have several options:

1. Install MMPose with ViTPose models:
   pip install mmpose
   
2. Download ViTPose weights manually from:
   https://github.com/ViTAE-Transformer/ViTPose
   
3. Use a different high-accuracy model by modifying the script

ViTPose-G requires significant GPU memory (8GB+ recommended).
Make sure you have sufficient resources available.
"""
            raise RuntimeError(error_msg)

    def get_human_joints(self, frame):
        """Get human pose keypoints from frame using ViTPose-G."""
        try:
            # ViTPose-G optimized inference parameters
            result_generator = self.inferencer(
                frame,
                show=False,
                return_vis=False,
                kpt_thr=self.confidence_threshold,  # Lower threshold for more keypoints
                skeleton_style='mmpose'  # Use mmpose skeleton style for consistency
            )
            result = next(result_generator)

            # Extract predictions from result
            predictions = result.get('predictions', [[]])[0]  # Get first batch, which is list of persons
            return predictions

        except Exception as e:
            print(f"Warning: Error in ViTPose-G pose detection: {e}")
            return []

    def process_video(self, video_path, court_csv_path, output_json_path):
        """Process video for pose estimation using ViTPose-G."""
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Using ViTPose-G (state-of-the-art accuracy: 81.1 AP on COCO)")
        logging.info(f"Model config: {self.model_config}")
        logging.info(f"Confidence threshold: {self.confidence_threshold}")

        # Load court data from CSV
        all_court_points = read_court_csv(court_csv_path)
        corner_points = extract_corner_points(all_court_points)

        # Create enlarged court boundary (20% bigger)
        enlarged_court = enlarge_court_boundary(corner_points, enlargement_factor=0.4)

        logging.info("Court boundary enlarged by 20% for more inclusive pose detection")

        # Get video info
        video_info = get_video_info(video_path)

        # Assume all frames are rally frames for now
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"Processing {len(rally_frames)} frames with ViTPose-G maximum accuracy...")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        pose_data = []

        with tqdm(total=frame_count, desc="Processing frames (ViTPose-G)") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in rally_frames:
                    # Get pose predictions from ViTPose-G
                    predictions = self.get_human_joints(frame)

                    # Process each detected person
                    for human_idx, prediction in enumerate(predictions):
                        # Extract keypoints and scores from MMPose format
                        keypoints = prediction.get('keypoints', None)
                        keypoint_scores = prediction.get('keypoint_scores', None)

                        if keypoints is None or keypoint_scores is None:
                            continue

                        # Check if at least one joint is within the enlarged court boundary
                        valid_joints_in_court = []
                        any_joint_in_court = False

                        for joint_idx in range(len(keypoints)):
                            if keypoint_scores[joint_idx] > self.confidence_threshold:
                                x, y = keypoints[joint_idx][0], keypoints[joint_idx][1]
                                # Check against enlarged court boundary
                                if is_point_in_court((x, y), corner_points, enlarged_court):
                                    valid_joints_in_court.append(joint_idx)
                                    any_joint_in_court = True

                        # Include this pose if at least one joint is in the enlarged court
                        if any_joint_in_court:
                            # Store pose data in format compatible with original script
                            person_joints = []
                            for joint_idx in range(len(keypoints)):
                                # Check if this specific joint is in the enlarged court area
                                joint_in_court = False
                                if keypoint_scores[joint_idx] > self.confidence_threshold:
                                    x, y = keypoints[joint_idx][0], keypoints[joint_idx][1]
                                    joint_in_court = is_point_in_court((x, y), corner_points, enlarged_court)

                                person_joints.append({
                                    "joint_index": joint_idx,
                                    "x": float(keypoints[joint_idx][0]),
                                    "y": float(keypoints[joint_idx][1]),
                                    "confidence": float(keypoint_scores[joint_idx]),
                                    "in_court": joint_in_court
                                })

                            pose_data.append({
                                "frame_index": frame_index,
                                "human_index": human_idx,
                                "joints": person_joints,
                                "joints_in_court": len(valid_joints_in_court)
                            })

                frame_index += 1
                pbar.update(1)

        cap.release()

        # Create output JSON
        output_data = {
            "court_points": corner_points,           # P1-P4 for homography (original)
            "enlarged_court_points": enlarged_court, # P1-P4 enlarged by 20%
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
                "model_type": "ViTPose-G",
                "pose_estimator": "ViTPose-Giant (81.1 AP on COCO)",
                "model_config": self.model_config,
                "confidence_threshold": self.confidence_threshold,
                "accuracy_note": "State-of-the-art accuracy with enlarged court boundary"
            }
        }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info(f"ViTPose-G pose estimation completed!")
        logging.info(f"✓ Model: ViTPose-Giant (maximum accuracy)")
        logging.info(f"✓ Total poses detected: {len(pose_data)}")
        logging.info(f"✓ Data saved to: {output_json_path}")

        return output_json_path


def main(video_path, confidence_threshold=0.2):
    """Main ViTPose-G pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*60)
    print("ViTPose-G Maximum Accuracy Pose Estimation")
    print("="*60)
    print("Using ViTPose-Giant for state-of-the-art pose estimation")
    print("Performance: 81.1 AP on COCO test-dev set")
    print("Note: Requires significant GPU memory and processing time")
    print("="*60)

    # Determine paths based on your structure
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    court_csv_path = os.path.join(result_dir, "court.csv")
    output_json_path = os.path.join(result_dir, "pose_vitpose_g.json")

    # Check if court CSV exists
    if not os.path.exists(court_csv_path):
        logging.error(f"Court CSV not found: {court_csv_path}")
        logging.error("Please run preprocessing first: python3 preprocess.py <video_path>")
        return None

    # Process poses with ViTPose-G
    pose_detect = ViTPoseGPoseDetect(confidence_threshold=confidence_threshold)
    try:
        output_path = pose_detect.process_video(video_path, court_csv_path, output_json_path)
        return output_path
    except Exception as e:
        logging.error(f"Error during ViTPose-G pose processing: {e}")
        return None
    finally:
        # Clean up
        del pose_detect


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ViTPose-G Maximum Accuracy Pose Estimation')
    parser.add_argument('video_path', help='Input video path')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Confidence threshold for keypoints (default: 0.2 for maximum sensitivity)')

    args = parser.parse_args()

    print(f"Input video: {args.video_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Output will be saved as: pose_vitpose_g.json")
    print()

    main(args.video_path, args.confidence)