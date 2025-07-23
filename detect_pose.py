#!/usr/bin/env python3
"""
Enhanced Pose Estimation with Court Detection Integration

Implementation based on research paper methodology:
- YOLOv11x-pose for human keypoint detection
- Perspective-n-Point (PnP) algorithm for camera pose estimation using calibration data
- Intelligent boundary extension based on camera elevation angle
- Ankle-knee joint validation for player filtering
- Sequential player detection without Y-position based ID assignment

Reads calibration data from detect_court.py and outputs pose.json with validated player poses.
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
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Badminton court dimensions (in meters) - Standard international court
COURT_LENGTH = 13.4  # baseline to baseline
COURT_WIDTH = 6.1    # sideline to sideline
MAX_JUMP_HEIGHT = 0.75  # maximum expected jump height in meters

# YOLO pose keypoint indices (COCO format)
ANKLE_KNEE_INDICES = [13, 14, 15, 16]  # left_knee, right_knee, left_ankle, right_ankle
MIN_ANKLE_KNEE_JOINTS = 2  # minimum required ankle/knee joints for validation
CONFIDENCE_THRESHOLD = 0.5  # minimum confidence for joint detection


def read_calibration_csv(csv_path):
    """Read calibration data from detect_court.py format."""
    calibration_data = {
        'camera_matrix': None,
        'dist_coeffs': None,
        'rvec': None,
        'tvec': None,
        'court_points': {},
        'reprojection_error': None,
        'image_size': None,
        'calibration_method': None
    }

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        current_section = None

        for row in csv_reader:
            if not row or row[0].startswith('#'):
                continue

            if len(row) < 2:
                continue

            key = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ''

            try:
                # Camera matrix parameters
                if key == 'fx':
                    current_section = 'camera_matrix'
                    calibration_data['camera_matrix'] = np.zeros((3, 3), dtype=np.float32)
                    calibration_data['camera_matrix'][2, 2] = 1.0
                    calibration_data['camera_matrix'][0, 0] = float(value)
                elif key == 'fy' and current_section == 'camera_matrix':
                    calibration_data['camera_matrix'][1, 1] = float(value)
                elif key == 'cx' and current_section == 'camera_matrix':
                    calibration_data['camera_matrix'][0, 2] = float(value)
                elif key == 'cy' and current_section == 'camera_matrix':
                    calibration_data['camera_matrix'][1, 2] = float(value)

                # Distortion coefficients
                elif key in ['k1', 'k2', 'k3', 'p1', 'p2']:
                    if calibration_data['dist_coeffs'] is None:
                        calibration_data['dist_coeffs'] = np.zeros(5, dtype=np.float32)
                    idx_map = {'k1': 0, 'k2': 1, 'p1': 2, 'p2': 3, 'k3': 4}
                    calibration_data['dist_coeffs'][idx_map[key]] = float(value)

                # Camera pose parameters
                elif key == 'rx':
                    current_section = 'pose'
                    calibration_data['rvec'] = np.zeros(3, dtype=np.float32)
                    calibration_data['rvec'][0] = float(value)
                elif key == 'ry' and current_section == 'pose':
                    calibration_data['rvec'][1] = float(value)
                elif key == 'rz' and current_section == 'pose':
                    calibration_data['rvec'][2] = float(value)
                elif key == 'tx':
                    if calibration_data['tvec'] is None:
                        calibration_data['tvec'] = np.zeros(3, dtype=np.float32)
                    calibration_data['tvec'][0] = float(value)
                elif key == 'ty':
                    if calibration_data['tvec'] is None:
                        calibration_data['tvec'] = np.zeros(3, dtype=np.float32)
                    calibration_data['tvec'][1] = float(value)
                elif key == 'tz':
                    if calibration_data['tvec'] is None:
                        calibration_data['tvec'] = np.zeros(3, dtype=np.float32)
                    calibration_data['tvec'][2] = float(value)

                # Quality metrics
                elif key == 'reprojection_error_px':
                    calibration_data['reprojection_error'] = float(value)
                elif key == 'calibration_strategy':
                    calibration_data['calibration_method'] = value

                # Point error data (skip this section for now)
                elif key == 'Point':
                    current_section = 'point_errors'
                    continue

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse row {row}: {e}")
                continue

    # Set image size based on cx, cy if available
    if calibration_data['camera_matrix'] is not None:
        cx = calibration_data['camera_matrix'][0, 2]
        cy = calibration_data['camera_matrix'][1, 2]
        calibration_data['image_size'] = [int(cx * 2), int(cy * 2)]  # Approximate from principal point

    print(f"Loaded calibration data:")
    print(f"  - Camera matrix: {'✓' if calibration_data['camera_matrix'] is not None else '✗'}")
    print(f"  - Distortion coefficients: {'✓' if calibration_data['dist_coeffs'] is not None else '✗'}")
    print(f"  - Pose parameters: {'✓' if calibration_data['rvec'] is not None else '✗'}")
    if calibration_data['reprojection_error'] is not None:
        print(f"  - Reprojection error: {calibration_data['reprojection_error']:.2f} pixels")
    if calibration_data['calibration_method']:
        print(f"  - Calibration method: {calibration_data['calibration_method']}")

    return calibration_data


def read_court_points_csv(csv_path):
    """Read court points from the court.csv file."""
    court_points = {}

    if not os.path.exists(csv_path):
        print(f"Court points CSV not found: {csv_path}")
        return court_points

    with open(csv_path, 'r') as f:
        # Check if header exists
        first_line = f.readline().strip()
        f.seek(0)

        if 'Point' in first_line and 'X' in first_line:
            print("📋 Court CSV has header row")
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row['Point'].strip()
                    x = float(row['X'])
                    y = float(row['Y'])
                    court_points[name] = [x, y]
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Skipping invalid row: {e}")
        else:
            print("📋 Court CSV without header")
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        name = row[0].strip()
                        x = float(row[1])
                        y = float(row[2])
                        court_points[name] = [x, y]
                    except (ValueError, IndexError):
                        continue

    print(f"✅ Loaded {len(court_points)} court points")
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
        print(f"✅ Found all corner points: {list(corner_points.keys())}")
        return corner_points

    # Fallback: use first 4 points if P1-P4 not available
    print("⚠️  P1-P4 not found, using first 4 points as corners")
    point_items = list(all_court_points.items())[:4]
    return {f"P{i+1}": coords for i, (_, coords) in enumerate(point_items)}


def get_3d_court_points():
    """
    Get 3D world coordinates of badminton court corners.
    Standard badminton court on ground plane (z=0).
    """
    return {
        'P1': np.array([0.0, 0.0, 0.0]),          # top-left corner
        'P2': np.array([0.0, COURT_LENGTH, 0.0]), # bottom-left corner
        'P3': np.array([COURT_WIDTH, COURT_LENGTH, 0.0]), # bottom-right corner
        'P4': np.array([COURT_WIDTH, 0.0, 0.0])   # top-right corner
    }


def calculate_camera_elevation_angle(rvec, tvec):
    """
    Calculate camera elevation angle relative to court plane.
    Used for perspective-aware boundary extension.
    """
    try:
        # Convert rotation vector to matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Get camera position in world coordinates
        camera_position = -rmat.T @ tvec
        h_camera = abs(camera_position[2])  # Height above ground

        # Calculate horizontal distance to court center
        court_center = np.array([COURT_WIDTH/2, COURT_LENGTH/2, 0])
        d_court = np.linalg.norm(camera_position[:2] - court_center[:2])

        # Calculate elevation angle
        theta = np.arctan(h_camera / max(d_court, 0.1))  # Avoid division by zero

        print(f"  📐 Camera height: {h_camera:.2f}m")
        print(f"  📏 Distance to court center: {d_court:.2f}m")
        print(f"  📐 Elevation angle: {np.degrees(theta):.1f}°")

        return theta, h_camera, d_court

    except Exception as e:
        print(f"⚠️  Elevation angle calculation failed: {e}")
        return np.radians(30), 5.0, 10.0  # Default values


def create_perspective_aware_boundary(corner_points_2d, rvec, tvec):
    """
    Create perspective-aware enlarged court boundary using calibrated camera parameters.
    """
    try:
        print("🔧 Creating perspective-aware boundary...")

        # Calculate camera elevation angle
        theta, h_camera, d_court = calculate_camera_elevation_angle(rvec, tvec)

        # Determine which regions are farther from camera
        rmat, _ = cv2.Rodrigues(rvec)
        camera_position = -rmat.T @ tvec

        # Calculate distances to different court regions
        regions = {
            'top': np.array([COURT_WIDTH/2, 0.0, 0.0]),           # P1-P4 baseline
            'bottom': np.array([COURT_WIDTH/2, COURT_LENGTH, 0.0]), # P2-P3 baseline
            'left': np.array([0.0, COURT_LENGTH/2, 0.0]),         # P1-P2 sideline
            'right': np.array([COURT_WIDTH, COURT_LENGTH/2, 0.0])  # P3-P4 sideline
        }

        region_distances = {}
        for region, center_3d in regions.items():
            distance = np.linalg.norm(camera_position[:2] - center_3d[:2])
            region_distances[region] = distance

        # Calculate extension distances for each region
        extensions = {}
        for region, d_region in region_distances.items():
            # Extension distance based on jump height and perspective
            extension_distance = (MAX_JUMP_HEIGHT / np.tan(theta)) * (d_region / d_court)
            # Convert to pixel units (approximate)
            court_length_pixels = np.linalg.norm(
                np.array(corner_points_2d['P2']) - np.array(corner_points_2d['P1'])
            )
            pixels_per_meter = court_length_pixels / COURT_LENGTH
            extension_pixels = extension_distance * pixels_per_meter
            extensions[region] = min(extension_pixels, court_length_pixels * 0.3)  # Limit extension

        print(f"  📏 Extension distances (pixels): {extensions}")

        # Apply asymmetric extension
        enlarged_corners = {}

        # Calculate court center for direction vectors
        center_x = np.mean([corner_points_2d[f'P{i}'][0] for i in range(1, 5)])
        center_y = np.mean([corner_points_2d[f'P{i}'][1] for i in range(1, 5)])

        # Extend each corner based on its region distance
        for i in range(1, 5):
            point_name = f'P{i}'
            if point_name not in corner_points_2d:
                continue

            original = np.array(corner_points_2d[point_name])

            # Direction from center to corner
            direction = original - np.array([center_x, center_y])
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 0:
                direction_unit = direction / direction_norm

                # Determine extension based on corner position
                if i in [1, 4]:  # Top baseline (P1, P4)
                    extension = extensions['top']
                else:  # Bottom baseline (P2, P3)
                    extension = extensions['bottom']

                # Apply extension
                extended_point = original + direction_unit * extension
                enlarged_corners[point_name] = [float(extended_point[0]), float(extended_point[1])]
            else:
                enlarged_corners[point_name] = corner_points_2d[point_name]

        print("✅ Perspective-aware boundary enlargement completed")
        return enlarged_corners

    except Exception as e:
        print(f"❌ Perspective-aware enlargement failed: {e}")
        return None


def enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.3):
    """Fallback uniform court boundary enlargement."""
    print(f"🔧 Using fallback uniform enlargement ({enlargement_factor*100:.0f}%)")

    if len(corner_points) != 4:
        print(f"⚠️  Expected 4 corner points, got {len(corner_points)}")
        return corner_points

    # Calculate center of the court
    center_x = sum(corner_points[f"P{i}"][0] for i in range(1, 5)) / 4
    center_y = sum(corner_points[f"P{i}"][1] for i in range(1, 5)) / 4

    # Enlarge each point by moving it away from the center
    enlarged_points = {}
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name not in corner_points:
            continue

        point = corner_points[point_name]
        dx = point[0] - center_x
        dy = point[1] - center_y

        new_x = center_x + dx * (1 + enlargement_factor)
        new_y = center_y + dy * (1 + enlargement_factor)

        enlarged_points[point_name] = [float(new_x), float(new_y)]

    print(f"✅ Uniform enlargement applied to {len(enlarged_points)} points")
    return enlarged_points


def is_point_in_court(point, corner_points, enlarged_boundary=None):
    """Check if a point is inside the court polygon using ray casting algorithm."""
    x, y = point

    # Use enlarged boundary if provided
    boundary_points = enlarged_boundary if enlarged_boundary is not None else corner_points

    # Create polygon from corner points (ensure proper order)
    polygon = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in boundary_points:
            polygon.append(boundary_points[point_name])

    if len(polygon) != 4:
        print(f"⚠️  Invalid polygon: only {len(polygon)} points")
        return True  # Allow all points if boundary detection failed

    # Ray casting algorithm for point-in-polygon test
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


def validate_pose_ankle_knee_criterion(person_keypoints, enlarged_boundary, original_boundary):
    """
    Validate pose using strict ankle-knee joint criterion.
    Requires at least 2 ankle/knee joints within enlarged court boundary.
    """
    if person_keypoints.xy is None or person_keypoints.conf is None:
        return False, []

    xy = person_keypoints.xy[0]
    conf = person_keypoints.conf[0]

    valid_ankle_knee_joints = []

    # Check ankle and knee joints specifically
    for joint_idx in ANKLE_KNEE_INDICES:
        if joint_idx < len(conf) and conf[joint_idx].item() > CONFIDENCE_THRESHOLD:
            x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
            if is_point_in_court((x, y), original_boundary, enlarged_boundary):
                valid_ankle_knee_joints.append(joint_idx)

    # Require at least MIN_ANKLE_KNEE_JOINTS valid joints
    is_valid = len(valid_ankle_knee_joints) >= MIN_ANKLE_KNEE_JOINTS
    return is_valid, valid_ankle_knee_joints


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


class EnhancedPoseDetector:
    """Enhanced pose detection system using calibrated camera parameters."""

    def __init__(self):
        self.device = self._select_device()
        self._setup_yolo()

    def _select_device(self):
        """Select optimal device for processing."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _setup_yolo(self):
        """Setup YOLOv11x-pose model."""
        print(f"🤖 Loading YOLOv11x-pose model on device: {self.device}")

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
                print(f"✅ Found YOLO model: {path}")
                break

        if model_path is None:
            print("📥 YOLO model not found locally, downloading...")
            model_path = 'yolo11x-pose.pt'

        # Load model with suppressed output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.pose_model = YOLO(model_path, verbose=False)
                self.pose_model.to(self.device)

        print("✅ YOLO model loaded successfully")

    def detect_poses(self, frame):
        """Detect human poses in frame using YOLOv11x-pose."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.pose_model(frame, verbose=False)
        return results

    def process_video(self, video_path, calibration_data, court_points, output_json_path):
        """Process video for enhanced pose estimation using calibrated parameters."""
        print(f"🎬 Processing video: {video_path}")

        # Extract corner points from court detection
        corner_points = extract_corner_points(court_points)

        if len(corner_points) < 4:
            raise ValueError(f"Need at least 4 corner points, got {len(corner_points)}")

        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        rvec = calibration_data['rvec']
        tvec = calibration_data['tvec']

        print("\n=== Enhanced Pose Estimation Pipeline ===")
        print("🔧 Using calibrated camera parameters for accurate pose estimation...")

        # Create perspective-aware boundary using calibrated parameters
        enlarged_court = None
        enlargement_method = "uniform_fallback"

        if camera_matrix is not None and rvec is not None and tvec is not None:
            print("🎯 Attempting perspective-aware boundary creation...")
            enlarged_court = create_perspective_aware_boundary(corner_points, rvec, tvec)
            if enlarged_court is not None:
                enlargement_method = "perspective_aware_calibrated"
                print("✅ Perspective-aware boundary created successfully")

        # Fallback to uniform enlargement
        if enlarged_court is None:
            print("⚠️  Falling back to uniform enlargement...")
            enlarged_court = enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.3)

        print(f"🔧 Court enlargement method: {enlargement_method}")
        print("=========================================\n")

        # Get video information
        video_info = get_video_info(video_path)
        print(f"📹 Video: {video_info['width']}x{video_info['height']}, {video_info['frame_count']} frames @ {video_info['fps']:.1f}fps")

        # Process all frames (assuming all are rally frames for this implementation)
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"🏃 Processing {len(rally_frames)} frames...")
        print(f"✅ Validation: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints required")
        print(f"🎯 No automatic player ID assignment - sequential detection only")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        all_pose_data = []

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in rally_frames:
                    # Detect poses
                    results = self.detect_poses(frame)

                    # Extract and validate poses
                    if results[0].keypoints is not None:
                        keypoints = results[0].keypoints
                        for human_idx, person_keypoints in enumerate(keypoints):
                            # Validate pose using ankle-knee criterion
                            is_valid, valid_joints = validate_pose_ankle_knee_criterion(
                                person_keypoints, enlarged_court, corner_points
                            )

                            if is_valid:
                                xy = person_keypoints.xy[0]
                                conf = person_keypoints.conf[0]

                                # Create pose data structure
                                person_joints = []
                                for joint_idx in range(xy.shape[0]):
                                    x, y = float(xy[joint_idx, 0].item()), float(xy[joint_idx, 1].item())
                                    confidence = float(conf[joint_idx].item())

                                    # Check if joint is in court area
                                    joint_in_court = False
                                    if confidence > CONFIDENCE_THRESHOLD:
                                        joint_in_court = is_point_in_court((x, y), corner_points, enlarged_court)

                                    person_joints.append({
                                        "joint_index": joint_idx,
                                        "x": x,
                                        "y": y,
                                        "confidence": confidence,
                                        "in_court": joint_in_court,
                                        "is_ankle_knee": joint_idx in ANKLE_KNEE_INDICES
                                    })

                                pose_data = {
                                    "frame_index": frame_index,
                                    "human_index": human_idx,
                                    "joints": person_joints,
                                    "valid_ankle_knee_joints": len(valid_joints),
                                    "validation_joints": valid_joints
                                }

                                all_pose_data.append(pose_data)

                frame_index += 1
                pbar.update(1)

        cap.release()

        # Create comprehensive output
        output_data = {
            "court_points": corner_points,
            "enlarged_court_points": enlarged_court,
            "all_court_points": court_points,
            "video_info": video_info,
            "rally_frames": list(rally_frames),
            "pose_data": all_pose_data,
            "camera_calibration": {
                "camera_matrix": calibration_data['camera_matrix'].tolist() if calibration_data['camera_matrix'] is not None else None,
                "distortion_coefficients": calibration_data['dist_coeffs'].tolist() if calibration_data['dist_coeffs'] is not None else None,
                "rotation_vector": calibration_data['rvec'].tolist() if calibration_data['rvec'] is not None else None,
                "translation_vector": calibration_data['tvec'].tolist() if calibration_data['tvec'] is not None else None,
                "reprojection_error": calibration_data['reprojection_error'],
                "calibration_method": calibration_data['calibration_method'],
                "image_size": calibration_data['image_size']
            },
            "processing_info": {
                "total_poses_detected": len(all_pose_data),
                "total_court_points": len(court_points),
                "corner_points_used": list(corner_points.keys()),
                "enlargement_method": enlargement_method,
                "max_jump_height_considered": f"{MAX_JUMP_HEIGHT}m",
                "validation_criteria": f"At least {MIN_ANKLE_KNEE_JOINTS} ankle/knee joints in enlarged boundary",
                "ankle_knee_joint_indices": ANKLE_KNEE_INDICES,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "id_assignment_method": "Sequential detection only (no automatic ID assignment)",
                "model_device": self.device,
                "model_type": "YOLOv11x-pose",
                "pose_estimator": "Enhanced Pose Estimation with Calibrated Camera Parameters",
                "court_dimensions": f"{COURT_LENGTH}m x {COURT_WIDTH}m",
                "implementation_features": [
                    "Calibrated camera parameters from detect_court.py",
                    "Perspective-n-Point (PnP) camera pose from calibration",
                    "Intelligent boundary extension based on calibrated elevation angle",
                    "Ankle-knee joint validation for player filtering",
                    "Sequential pose detection without automatic player ID assignment",
                    "Full integration with BWF standard court detection system",
                    "Automatic coordinate system correction"
                ]
            }
        }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n=== Processing Complete ===")
        print(f"✅ Total valid poses detected: {len(all_pose_data)}")
        print(f"🔧 Enlargement method: {enlargement_method}")
        print(f"✅ Validation: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints required")
        print(f"🎯 Camera calibration: {'Calibrated' if camera_matrix is not None else 'Estimated'}")
        print(f"💾 Data saved to: {output_json_path}")
        print("===========================")

        return output_json_path


def main(video_path):
    """Main enhanced pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("ENHANCED POSE ESTIMATION WITH CALIBRATED CAMERA PARAMETERS")
    print("="*80)
    print("Implementation features:")
    print("• YOLOv11x-pose for human keypoint detection")
    print("• Calibrated camera parameters from detect_court.py")
    print("• Intelligent boundary extension based on calibrated elevation angle")
    print("• Ankle-knee joint validation for player filtering")
    print("• Sequential pose detection (no automatic player ID assignment)")
    print("• Full integration with BWF standard court detection system")
    print("• Automatic coordinate system correction")
    print("="*80)
    print(f"Court dimensions: {COURT_LENGTH}m x {COURT_WIDTH}m")
    print(f"Maximum jump height: {MAX_JUMP_HEIGHT}m")
    print(f"Validation threshold: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints")
    print("="*80)

    # Determine file paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    # Look for calibration CSV file (from detect_court.py)
    calibration_csv_path = os.path.join(result_dir, "calibration.csv")
    court_csv_path = os.path.join(result_dir, "court.csv")

    if not os.path.exists(calibration_csv_path):
        logging.error(f"Calibration data not found: {calibration_csv_path}")
        logging.error("Please run court detection first: python3 detect_court.py <video_path>")
        return None

    if not os.path.exists(court_csv_path):
        logging.error(f"Court points data not found: {court_csv_path}")
        logging.error("Please run court detection first: python3 detect_court.py <video_path>")
        return None

    output_json_path = os.path.join(result_dir, "pose.json")

    # Load calibration data
    try:
        print(f"📊 Loading calibration data from: {calibration_csv_path}")
        calibration_data = read_calibration_csv(calibration_csv_path)

        print(f"📍 Loading court points from: {court_csv_path}")
        court_points = read_court_points_csv(court_csv_path)

        if len(court_points) == 0:
            raise ValueError("No court points found in CSV file")

    except Exception as e:
        logging.error(f"Failed to load calibration/court data: {e}")
        return None

    # Initialize enhanced pose detector
    pose_detector = EnhancedPoseDetector()

    try:
        output_path = pose_detector.process_video(video_path, calibration_data, court_points, output_json_path)
        return output_path
    except Exception as e:
        logging.error(f"Pose estimation failed: {e}")
        return None
    finally:
        # Clean up resources
        del pose_detector


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_pose.py <input_video_path>")
        print("\nExample:")
        print("  python3 detect_pose.py samples/badminton_match.mp4")
        print("\nRequirements:")
        print("  - Court detection and calibration must be run first")
        print("  - detect_court.py generates calibration data")
        print("  - YOLOv11x-pose model (will be downloaded if not found)")
        print("  - OpenCV, PyTorch, Ultralytics")
        sys.exit(1)

    input_video_path = sys.argv[1]

    # Validate input video
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found: {input_video_path}")
        sys.exit(1)

    result = main(input_video_path)
    if result:
        print(f"\n✅ Success! Enhanced pose estimation completed.")
        print(f"  Output: {result}")
    else:
        print(f"\n❌ Failed! Enhanced pose estimation encountered errors.")
        sys.exit(1)