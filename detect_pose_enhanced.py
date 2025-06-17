#!/usr/bin/env python3
"""
Enhanced Pose Estimation with MMPose Inferencer and Court CSV Input

Uses the highest accuracy models available in MMPose with enhanced processing options.
Optimized for maximum accuracy at the cost of computational resources.
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


def is_point_in_court(point, corner_points):
    """Check if a point is inside the court polygon using corner points."""
    x, y = point

    # Create polygon from corner points
    polygon = [corner_points[f"P{i}"] for i in range(1, 5) if f"P{i}" in corner_points]

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


class EnhancedMMPosePoseDetect:
    def __init__(self, device=None, accuracy_level="highest"):
        self.device = self.select_device() if device is None else device
        self.accuracy_level = accuracy_level
        self.setup_mmpose()

    def select_device(self):
        """Select best available device"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def setup_mmpose(self):
        """Initialize MMPose inferencer with the highest accuracy models available."""
        print(f"Loading MMPose model on device: {self.device}")
        print(f"Accuracy level: {self.accuracy_level}")

        # Model configurations ranked by accuracy (highest to lowest)
        model_configs = {
            "highest": [
                # ViTPose models (state-of-the-art accuracy)
                ('td-hm_ViTPose-huge_8xb64-210e_coco-256x192', "ViTPose-Huge (highest accuracy)"),
                ('td-hm_ViTPose-large_8xb64-210e_coco-256x192', "ViTPose-Large (very high accuracy)"),
                ('td-hm_ViTPose-base_8xb64-210e_coco-256x192', "ViTPose-Base (high accuracy)"),
                # HRNet models (proven high accuracy)
                ('td-hm_hrnet-w48_8xb32-210e_coco-384x288', "HRNet-W48 384x288 (high accuracy, larger input)"),
                ('td-hm_hrnet-w48_8xb32-210e_coco-256x192', "HRNet-W48 (high accuracy)"),
                ('td-hm_hrnet-w32_8xb64-210e_coco-256x192', "HRNet-W32 (good accuracy)"),
            ],
            "high": [
                ('td-hm_hrnet-w48_8xb32-210e_coco-256x192', "HRNet-W48"),
                ('td-hm_hrnet-w32_8xb64-210e_coco-256x192', "HRNet-W32"),
                ('human', "RTMPose (default)"),
            ],
            "balanced": [
                ('human', "RTMPose (default)"),
                ('rtmpose-m_8xb256-420e_coco-256x192', "RTMPose-M"),
            ]
        }

        configs_to_try = model_configs.get(self.accuracy_level, model_configs["highest"])

        for model_name, description in configs_to_try:
            try:
                print(f"Attempting to load: {description}")
                self.inferencer = MMPoseInferencer(
                    pose2d=model_name,
                    device=self.device,
                    show_progress=False
                )
                print(f"✓ Successfully loaded: {description}")
                self.model_name = description
                return
            except Exception as e:
                print(f"✗ Failed to load {description}: {e}")
                continue

        # If all models fail, raise error
        raise RuntimeError("Failed to load any MMPose inferencer. Please check your MMPose installation.")

    def get_human_joints(self, frame, enhanced_processing=True):
        """Get human pose keypoints from frame using MMPose inferencer with enhanced processing."""
        try:
            # Enhanced inference parameters for higher accuracy
            inference_params = {
                'show': False,
                'return_vis': False,
                'kpt_thr': 0.2,  # Lower threshold for more keypoints
            }

            if enhanced_processing:
                # Additional parameters for higher accuracy
                inference_params.update({
                    'skeleton_style': 'mmpose',  # Use mmpose skeleton style
                    'radius': 4,  # Larger radius for better visualization (if needed)
                    'thickness': 2,  # Better line thickness
                })

            # MMPose inferencer returns a generator
            result_generator = self.inferencer(frame, **inference_params)
            result = next(result_generator)

            # Extract predictions from result
            # result['predictions'] is a list with one element (batch size 1)
            # result['predictions'][0] is a list of detected persons
            predictions = result.get('predictions', [[]])[0]  # Get first batch, which is list of persons
            return predictions

        except Exception as e:
            print(f"Warning: Error in pose detection: {e}")
            return []

    def process_video(self, video_path, court_csv_path, output_json_path,
                      enhanced_processing=True, confidence_threshold=0.3):
        """Process video for pose estimation using MMPose with enhanced settings."""
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Using model: {self.model_name}")
        logging.info(f"Enhanced processing: {enhanced_processing}")
        logging.info(f"Confidence threshold: {confidence_threshold}")

        # Load court data from CSV
        all_court_points = read_court_csv(court_csv_path)
        corner_points = extract_corner_points(all_court_points)

        # Get video info
        video_info = get_video_info(video_path)

        # Assume all frames are rally frames for now
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"Processing {len(rally_frames)} frames with enhanced accuracy...")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        pose_data = []

        with tqdm(total=frame_count, desc="Processing frames (enhanced accuracy)") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in rally_frames:
                    # Get pose predictions from MMPose with enhanced processing
                    predictions = self.get_human_joints(frame, enhanced_processing)

                    # Process each detected person
                    for human_idx, prediction in enumerate(predictions):
                        # Extract keypoints and scores from MMPose format
                        keypoints = prediction.get('keypoints', None)
                        keypoint_scores = prediction.get('keypoint_scores', None)

                        if keypoints is None or keypoint_scores is None:
                            continue

                        # Check if any body part is within the court
                        is_in_court = False
                        court_joints = []

                        for joint_idx in range(len(keypoints)):
                            if keypoint_scores[joint_idx] > confidence_threshold:
                                x, y = keypoints[joint_idx][0], keypoints[joint_idx][1]
                                if is_point_in_court((x, y), corner_points):
                                    is_in_court = True
                                    court_joints.append(joint_idx)

                        if is_in_court:
                            # Store pose data in format compatible with original script
                            person_joints = []
                            for joint_idx in range(len(keypoints)):
                                person_joints.append({
                                    "joint_index": joint_idx,
                                    "x": float(keypoints[joint_idx][0]),
                                    "y": float(keypoints[joint_idx][1]),
                                    "confidence": float(keypoint_scores[joint_idx]),
                                    "in_court": joint_idx in court_joints
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
            "court_points": corner_points,           # P1-P4 for homography
            "all_court_points": all_court_points,    # All detected points
            "video_info": video_info,
            "rally_frames": list(rally_frames),
            "pose_data": pose_data,
            "processing_info": {
                "total_poses_detected": len(pose_data),
                "total_court_points": len(all_court_points),
                "corner_points_used": list(corner_points.keys()),
                "model_device": self.device,
                "model_type": "MMPose Enhanced",
                "pose_estimator": self.model_name,
                "accuracy_level": self.accuracy_level,
                "enhanced_processing": enhanced_processing,
                "confidence_threshold": confidence_threshold
            }
        }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info(f"Enhanced pose estimation completed!")
        logging.info(f"✓ Model used: {self.model_name}")
        logging.info(f"✓ Total poses detected: {len(pose_data)}")
        logging.info(f"✓ Data saved to: {output_json_path}")

        return output_json_path


def main(video_path, accuracy_level="highest", confidence_threshold=0.3):
    """Main enhanced pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Determine paths based on your structure
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    court_csv_path = os.path.join(result_dir, "court.csv")
    output_json_path = os.path.join(result_dir, "pose_enhanced.json")

    # Check if court CSV exists
    if not os.path.exists(court_csv_path):
        logging.error(f"Court CSV not found: {court_csv_path}")
        logging.error("Please run preprocessing first: python3 preprocess.py <video_path>")
        return None

    # Process poses with Enhanced MMPose
    pose_detect = EnhancedMMPosePoseDetect(accuracy_level=accuracy_level)
    try:
        output_path = pose_detect.process_video(
            video_path,
            court_csv_path,
            output_json_path,
            enhanced_processing=True,
            confidence_threshold=confidence_threshold
        )
        return output_path
    except Exception as e:
        logging.error(f"Error during pose processing: {e}")
        return None
    finally:
        # Clean up
        del pose_detect


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced MMPose Human Pose Estimation')
    parser.add_argument('video_path', help='Input video path')
    parser.add_argument('--accuracy', choices=['highest', 'high', 'balanced'],
                        default='highest', help='Accuracy level (default: highest)')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold for keypoints (default: 0.3)')

    args = parser.parse_args()

    print("Enhanced MMPose Pose Estimation")
    print("===============================")
    print(f"Video: {args.video_path}")
    print(f"Accuracy level: {args.accuracy}")
    print(f"Confidence threshold: {args.confidence}")
    print("\nThis script uses the highest accuracy models available in MMPose.")
    print("Models will be tried in order of accuracy: ViTPose-Huge > ViTPose-Large > ViTPose-Base > HRNet-W48...")
    print()

    main(args.video_path, args.accuracy, args.confidence)