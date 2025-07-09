#!/usr/bin/env python3
"""
Pose Estimation with Perspective-Aware Court Enlargement

Reads court.csv directly and outputs pose.json with sophisticated court boundary
enlargement based on camera position and perspective. Uses solvePnP to determine
camera position and extends court boundary intelligently for jump detection.
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

# Badminton court dimensions (in meters)
COURT_LENGTH = 13.4  # meters
COURT_WIDTH = 6.1    # meters
JUMP_HEIGHT = 1.25   # meters (maximum expected jump height)

# YOLO pose keypoint indices for validation (COCO format)
ANKLE_KNEE_INDICES = [13, 14, 15, 16]  # Left knee, Right knee, Left ankle, Right ankle
MIN_VALID_JOINTS = 2  # At least 2 out of 4 ankle/knee joints needed


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


def get_3d_court_points():
    """
    Get 3D world coordinates of badminton court corners.
    Origin at P1 (top-left), court lies on ground plane (z=0).
    Clockwise order: P1(TL) -> P4(TR) -> P3(BR) -> P2(BL)
    """
    return {
        'P1': np.array([0.0, 0.0, 0.0]),        # Top-left
        'P2': np.array([0.0, COURT_LENGTH, 0.0]), # Bottom-left
        'P3': np.array([COURT_WIDTH, COURT_LENGTH, 0.0]), # Bottom-right
        'P4': np.array([COURT_WIDTH, 0.0, 0.0])  # Top-right
    }


def solve_camera_pose(corner_points_2d, camera_matrix=None):
    """
    Solve camera pose using solvePnP with known court dimensions.

    Args:
        corner_points_2d: Dictionary with P1-P4 image coordinates
        camera_matrix: Camera intrinsic matrix (if None, will estimate)

    Returns:
        (success, rvec, tvec, camera_matrix) or (False, None, None, None)
    """
    try:
        # Get 3D world points
        world_points_3d = get_3d_court_points()

        # Prepare 3D and 2D point arrays in order P1, P2, P3, P4
        object_points = np.array([
            world_points_3d['P1'],
            world_points_3d['P2'],
            world_points_3d['P3'],
            world_points_3d['P4']
        ], dtype=np.float32)

        image_points = np.array([
            corner_points_2d['P1'],
            corner_points_2d['P2'],
            corner_points_2d['P3'],
            corner_points_2d['P4']
        ], dtype=np.float32)

        # Estimate camera matrix if not provided
        if camera_matrix is None:
            # Simple estimation based on image points
            image_center_x = np.mean(image_points[:, 0])
            image_center_y = np.mean(image_points[:, 1])

            # Estimate focal length from court perspective
            court_width_pixels = np.linalg.norm(image_points[0] - image_points[3])  # P1 to P4
            court_length_pixels = np.linalg.norm(image_points[0] - image_points[1])  # P1 to P2

            # Rough focal length estimation
            focal_length = max(court_width_pixels, court_length_pixels) * 1.2

            camera_matrix = np.array([
                [focal_length, 0, image_center_x],
                [0, focal_length, image_center_y],
                [0, 0, 1]
            ], dtype=np.float32)

        # Solve PnP
        dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            print("✓ Camera pose solved successfully")
            # Convert rotation vector to matrix for easier interpretation
            rmat, _ = cv2.Rodrigues(rvec)
            camera_position = -rmat.T @ tvec
            print(f"  Camera position: ({camera_position[0][0]:.2f}, {camera_position[1][0]:.2f}, {camera_position[2][0]:.2f}) meters")
            return True, rvec, tvec, camera_matrix
        else:
            print("✗ solvePnP failed")
            return False, None, None, None

    except Exception as e:
        print(f"✗ Camera pose solving failed: {e}")
        return False, None, None, None


def determine_far_baseline(corner_points_2d, rvec, tvec):
    """
    Determine which baseline (top or bottom) is farther from the camera.

    Returns:
        'top' if P1-P4 baseline is farther, 'bottom' if P2-P3 baseline is farther
    """
    try:
        # Get camera position in world coordinates
        rmat, _ = cv2.Rodrigues(rvec)
        camera_position = -rmat.T @ tvec
        cam_x, cam_y, cam_z = camera_position.flatten()

        # Calculate distances to baseline midpoints
        top_baseline_midpoint = np.array([COURT_WIDTH/2, 0.0, 0.0])  # Midpoint of P1-P4
        bottom_baseline_midpoint = np.array([COURT_WIDTH/2, COURT_LENGTH, 0.0])  # Midpoint of P2-P3

        dist_to_top = np.linalg.norm(camera_position.flatten() - top_baseline_midpoint)
        dist_to_bottom = np.linalg.norm(camera_position.flatten() - bottom_baseline_midpoint)

        far_baseline = 'bottom' if dist_to_bottom > dist_to_top else 'top'
        print(f"  Far baseline: {far_baseline} (dist_top: {dist_to_top:.2f}m, dist_bottom: {dist_to_bottom:.2f}m)")

        return far_baseline

    except Exception as e:
        print(f"Warning: Could not determine far baseline: {e}")
        return 'bottom'  # Default assumption


def create_perspective_enlarged_court(corner_points_2d, rvec, tvec, camera_matrix):
    """
    Create perspective-aware enlarged court boundary.
    Simpler approach: extend the far baseline parallel to itself by a calculated distance.
    """
    try:
        # Determine which baseline is farther from camera
        far_baseline = determine_far_baseline(corner_points_2d, rvec, tvec)

        # Get camera position and height for extension calculation
        rmat, _ = cv2.Rodrigues(rvec)
        camera_position = -rmat.T @ tvec
        camera_height = abs(camera_position[2][0])  # Height above ground

        # Calculate extension distance based on camera height and jump height
        # Using similar triangles: extension_distance = (jump_height / camera_height) * baseline_distance
        if far_baseline == 'bottom':
            # Distance from camera to bottom baseline
            baseline_center_3d = np.array([COURT_WIDTH/2, COURT_LENGTH, 0])
            baseline_distance = np.linalg.norm(camera_position.flatten() - baseline_center_3d)
        else:
            # Distance from camera to top baseline
            baseline_center_3d = np.array([COURT_WIDTH/2, 0, 0])
            baseline_distance = np.linalg.norm(camera_position.flatten() - baseline_center_3d)

        # Calculate extension factor based on perspective geometry
        if camera_height > 0.1:  # Avoid division by zero
            extension_factor = JUMP_HEIGHT / camera_height
            # Limit extension to reasonable bounds
            extension_factor = min(extension_factor, 0.5)  # Max 50% extension
            extension_factor = max(extension_factor, 0.1)  # Min 10% extension
        else:
            extension_factor = 0.3  # Default fallback

        print(f"  Camera height: {camera_height:.2f}m, Extension factor: {extension_factor:.3f}")

        # Simple geometric extension: move far baseline away from near baseline
        enlarged_corners = {}

        if far_baseline == 'bottom':
            # Keep top baseline (P1, P4), extend bottom baseline (P2, P3)
            enlarged_corners['P1'] = [float(corner_points_2d['P1'][0]), float(corner_points_2d['P1'][1])]
            enlarged_corners['P4'] = [float(corner_points_2d['P4'][0]), float(corner_points_2d['P4'][1])]

            # Calculate direction vector from top to bottom baseline
            top_center = [(corner_points_2d['P1'][0] + corner_points_2d['P4'][0])/2,
                          (corner_points_2d['P1'][1] + corner_points_2d['P4'][1])/2]
            bottom_center = [(corner_points_2d['P2'][0] + corner_points_2d['P3'][0])/2,
                             (corner_points_2d['P2'][1] + corner_points_2d['P3'][1])/2]

            # Direction from top to bottom
            direction = [bottom_center[0] - top_center[0], bottom_center[1] - top_center[1]]
            direction_length = np.sqrt(direction[0]**2 + direction[1]**2)
            if direction_length > 0:
                direction = [direction[0]/direction_length, direction[1]/direction_length]
            else:
                direction = [0, 1]  # Default downward

            # Extension distance in pixels
            court_height_pixels = direction_length
            extension_pixels = court_height_pixels * extension_factor

            # Move P2 and P3 further in the direction
            enlarged_corners['P2'] = [
                float(corner_points_2d['P2'][0] + direction[0] * extension_pixels),
                float(corner_points_2d['P2'][1] + direction[1] * extension_pixels)
            ]
            enlarged_corners['P3'] = [
                float(corner_points_2d['P3'][0] + direction[0] * extension_pixels),
                float(corner_points_2d['P3'][1] + direction[1] * extension_pixels)
            ]

        else:
            # Keep bottom baseline (P2, P3), extend top baseline (P1, P4)
            enlarged_corners['P2'] = [float(corner_points_2d['P2'][0]), float(corner_points_2d['P2'][1])]
            enlarged_corners['P3'] = [float(corner_points_2d['P3'][0]), float(corner_points_2d['P3'][1])]

            # Calculate direction vector from bottom to top baseline
            bottom_center = [(corner_points_2d['P2'][0] + corner_points_2d['P3'][0])/2,
                             (corner_points_2d['P2'][1] + corner_points_2d['P3'][1])/2]
            top_center = [(corner_points_2d['P1'][0] + corner_points_2d['P4'][0])/2,
                          (corner_points_2d['P1'][1] + corner_points_2d['P4'][1])/2]

            # Direction from bottom to top
            direction = [top_center[0] - bottom_center[0], top_center[1] - bottom_center[1]]
            direction_length = np.sqrt(direction[0]**2 + direction[1]**2)
            if direction_length > 0:
                direction = [direction[0]/direction_length, direction[1]/direction_length]
            else:
                direction = [0, -1]  # Default upward

            # Extension distance in pixels
            court_height_pixels = direction_length
            extension_pixels = court_height_pixels * extension_factor

            # Move P1 and P4 further in the direction
            enlarged_corners['P1'] = [
                float(corner_points_2d['P1'][0] + direction[0] * extension_pixels),
                float(corner_points_2d['P1'][1] + direction[1] * extension_pixels)
            ]
            enlarged_corners['P4'] = [
                float(corner_points_2d['P4'][0] + direction[0] * extension_pixels),
                float(corner_points_2d['P4'][1] + direction[1] * extension_pixels)
            ]

        print(f"    {far_baseline} baseline extended by {extension_pixels:.1f} pixels ({extension_factor*100:.1f}%)")
        print("✓ Perspective-aware court enlargement completed")
        return enlarged_corners

    except Exception as e:
        print(f"✗ Perspective enlargement failed: {e}")
        return None


def enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.4):
    """Fallback: Uniform court boundary enlargement."""
    print(f"Using fallback uniform enlargement ({enlargement_factor*100}%)")

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

        # Ensure Python float types
        enlarged_points[f"P{i+1}"] = [float(new_x), float(new_y)]

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


def is_valid_pose(person_keypoints, enlarged_boundary, original_boundary):
    """
    Check if pose is valid based on ankle and knee joint positions.
    Requires at least 2 out of 4 ankle/knee joints to be within enlarged boundary.
    """
    if person_keypoints.xy is None or person_keypoints.conf is None:
        return False, []

    xy = person_keypoints.xy[0]
    conf = person_keypoints.conf[0]

    valid_joints_in_court = []

    # Check ankle and knee joints specifically
    for joint_idx in ANKLE_KNEE_INDICES:
        if joint_idx < len(conf) and conf[joint_idx].item() > 0.5:
            x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
            if is_point_in_court((x, y), original_boundary, enlarged_boundary):
                valid_joints_in_court.append(joint_idx)

    # Require at least 2 valid ankle/knee joints
    is_valid = len(valid_joints_in_court) >= MIN_VALID_JOINTS
    return is_valid, valid_joints_in_court


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
        """Process video for pose estimation with perspective-aware court boundary."""
        logging.info(f"Processing video: {video_path}")

        # Load court data from CSV
        all_court_points = read_court_csv(court_csv_path)
        corner_points = extract_corner_points(all_court_points)

        print("\n=== Court Boundary Enlargement ===")
        print("Attempting perspective-aware enlargement using camera pose...")

        # Try perspective-aware enlargement using solvePnP
        success, rvec, tvec, camera_matrix = solve_camera_pose(corner_points)

        enlarged_court = None
        enlargement_method = "uniform_fallback"

        if success:
            enlarged_court = create_perspective_enlarged_court(
                corner_points, rvec, tvec, camera_matrix
            )
            if enlarged_court is not None:
                enlargement_method = "perspective_aware"

        # Fallback to uniform enlargement if perspective method failed
        if enlarged_court is None:
            print("Falling back to uniform enlargement...")
            enlarged_court = enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.4)
            enlargement_method = "uniform_fallback"

        print(f"Court enlargement method: {enlargement_method}")
        print("=====================================\n")

        # Get video info
        video_info = get_video_info(video_path)

        # Assume all frames are rally frames for now
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"Processing {len(rally_frames)} frames...")
        print(f"Validation criteria: At least {MIN_VALID_JOINTS} ankle/knee joints within enlarged boundary")

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
                            # Check if pose is valid using ankle/knee criteria
                            is_valid, valid_joints = is_valid_pose(
                                person_keypoints, enlarged_court, corner_points
                            )

                            if is_valid:
                                xy = person_keypoints.xy[0]
                                conf = person_keypoints.conf[0]

                                # Store pose data for all joints
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
                                        "in_court": joint_in_court,
                                        "is_ankle_knee": joint_idx in ANKLE_KNEE_INDICES
                                    })

                                pose_data.append({
                                    "frame_index": frame_index,
                                    "human_index": human_idx,
                                    "joints": person_joints,
                                    "valid_ankle_knee_joints": len(valid_joints),
                                    "validation_joints": valid_joints
                                })

                frame_index += 1
                pbar.update(1)

        cap.release()

        # Create output JSON
        output_data = {
            "court_points": corner_points,           # P1-P4 original corners
            "enlarged_court_points": enlarged_court, # Enlarged boundary
            "all_court_points": all_court_points,    # All detected points
            "video_info": video_info,
            "rally_frames": list(rally_frames),
            "pose_data": pose_data,
            "processing_info": {
                "total_poses_detected": len(pose_data),
                "total_court_points": len(all_court_points),
                "corner_points_used": list(corner_points.keys()),
                "enlargement_method": enlargement_method,
                "jump_height_considered": f"{JUMP_HEIGHT}m",
                "validation_criteria": f"At least {MIN_VALID_JOINTS} ankle/knee joints in enlarged boundary",
                "ankle_knee_joint_indices": ANKLE_KNEE_INDICES,
                "model_device": self.device,
                "model_type": "YOLO11x-pose",
                "pose_estimator": "YOLO11x Pose Estimation with Perspective-Aware Boundary",
                "badminton_court_dimensions": f"{COURT_LENGTH}m x {COURT_WIDTH}m"
            }
        }

        # Add camera pose info if available
        if success and enlargement_method == "perspective_aware":
            rmat, _ = cv2.Rodrigues(rvec)
            camera_position = -rmat.T @ tvec
            output_data["camera_info"] = {
                "pose_solved": True,
                "camera_position": [float(x) for x in camera_position.flatten()],
                "rotation_vector": [float(x) for x in rvec.flatten()],
                "translation_vector": [float(x) for x in tvec.flatten()],
                "camera_matrix": [[float(x) for x in row] for row in camera_matrix]
            }
        else:
            output_data["camera_info"] = {
                "pose_solved": False,
                "fallback_method": "uniform_enlargement_40_percent"
            }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info(f"Pose estimation completed!")
        logging.info(f"✓ Total poses detected: {len(pose_data)}")
        logging.info(f"✓ Enlargement method: {enlargement_method}")
        logging.info(f"✓ Validation: {MIN_VALID_JOINTS}+ ankle/knee joints required")
        logging.info(f"✓ Data saved to: {output_json_path}")

        return output_json_path


def main(video_path):
    """Main pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*70)
    print("YOLO Pose Estimation with Perspective-Aware Court Enlargement")
    print("="*70)
    print("Enhanced for badminton with intelligent court boundary extension")
    print(f"Court dimensions: {COURT_LENGTH}m x {COURT_WIDTH}m")
    print(f"Jump height consideration: {JUMP_HEIGHT}m")
    print(f"Validation: {MIN_VALID_JOINTS}+ ankle/knee joints required")
    print("="*70)

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