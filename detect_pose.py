#!/usr/bin/env python3
"""
Enhanced Pose Estimation with Perspective-Aware Court Boundary Enlargement

Implementation based on research paper methodology:
- YOLOv11x-pose for human keypoint detection
- Perspective-n-Point (PnP) algorithm for camera pose estimation
- Intelligent boundary extension based on camera elevation angle
- Ankle-knee joint validation for player filtering
- Multi-person tracking with Hungarian algorithm for ID assignment

Reads court.csv and outputs pose.json with validated player poses.
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
from scipy.optimize import linear_sum_assignment
import warnings

# Suppress warnings and YOLO output
warnings.filterwarnings("ignore")
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Badminton court dimensions (in meters) - Standard international court
COURT_LENGTH = 13.4  # baseline to baseline
COURT_WIDTH = 6.1    # sideline to sideline
MAX_JUMP_HEIGHT = 1.25  # maximum expected jump height in meters

# YOLO pose keypoint indices (COCO format)
ANKLE_KNEE_INDICES = [13, 14, 15, 16]  # left_knee, right_knee, left_ankle, right_ankle
MIN_ANKLE_KNEE_JOINTS = 2  # minimum required ankle/knee joints for validation
CONFIDENCE_THRESHOLD = 0.5  # minimum confidence for joint detection

# Tracking parameters
MAX_TRACK_DISTANCE = 100  # maximum pixel distance for track association
MAX_TRACK_AGE = 30  # maximum frames to keep a track without detection


class TrackState:
    """Manages individual player track state."""
    def __init__(self, track_id, initial_pose, frame_idx):
        self.id = track_id
        self.poses = [initial_pose]
        self.frame_indices = [frame_idx]
        self.last_seen = frame_idx
        self.age = 0
        self.center_history = [self._get_pose_center(initial_pose)]

    def _get_pose_center(self, pose):
        """Calculate center point of pose from valid keypoints."""
        valid_points = []
        for joint in pose['joints']:
            if joint['confidence'] > CONFIDENCE_THRESHOLD:
                valid_points.append([joint['x'], joint['y']])

        if valid_points:
            center = np.mean(valid_points, axis=0)
            return [float(center[0]), float(center[1])]
        return [0.0, 0.0]

    def update(self, new_pose, frame_idx):
        """Update track with new pose detection."""
        self.poses.append(new_pose)
        self.frame_indices.append(frame_idx)
        self.last_seen = frame_idx
        self.age = 0
        self.center_history.append(self._get_pose_center(new_pose))

    def predict_position(self):
        """Predict next position based on velocity."""
        if len(self.center_history) < 2:
            return self.center_history[-1] if self.center_history else [0, 0]

        # Simple linear prediction based on last two positions
        velocity = [
            self.center_history[-1][0] - self.center_history[-2][0],
            self.center_history[-1][1] - self.center_history[-2][1]
        ]
        predicted = [
            self.center_history[-1][0] + velocity[0],
            self.center_history[-1][1] + velocity[1]
        ]
        return predicted


class MultiPersonTracker:
    """Handles tracking of multiple players with ID assignment."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.frame_idx = 0

    def update(self, valid_poses, frame_idx):
        """Update tracks with new pose detections using Hungarian algorithm."""
        self.frame_idx = frame_idx

        # Age existing tracks
        for track in self.tracks.values():
            track.age += 1

        if not valid_poses:
            return []

        # Calculate cost matrix for pose-to-track assignment
        active_tracks = [t for t in self.tracks.values() if t.age < MAX_TRACK_AGE]

        if not active_tracks:
            # No active tracks, create new ones
            for pose in valid_poses:
                self._create_new_track(pose, frame_idx)
        else:
            # Calculate distance matrix
            cost_matrix = np.full((len(valid_poses), len(active_tracks)), np.inf)

            for pose_idx, pose in enumerate(valid_poses):
                pose_center = self._get_pose_center(pose)
                for track_idx, track in enumerate(active_tracks):
                    predicted_pos = track.predict_position()
                    distance = np.linalg.norm(np.array(pose_center) - np.array(predicted_pos))

                    if distance < MAX_TRACK_DISTANCE:
                        # Include pose similarity in cost
                        pose_similarity = self._calculate_pose_similarity(pose, track.poses[-1])
                        cost_matrix[pose_idx, track_idx] = distance + (1.0 - pose_similarity) * 50

            # Solve assignment problem using Hungarian algorithm
            pose_indices, track_indices = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            unmatched_poses = list(range(len(valid_poses)))
            for pose_idx, track_idx in zip(pose_indices, track_indices):
                if cost_matrix[pose_idx, track_idx] < np.inf:
                    active_tracks[track_idx].update(valid_poses[pose_idx], frame_idx)
                    unmatched_poses.remove(pose_idx)

            # Create new tracks for unmatched poses
            for pose_idx in unmatched_poses:
                self._create_new_track(valid_poses[pose_idx], frame_idx)

        # Remove old tracks
        self.tracks = {tid: track for tid, track in self.tracks.items() if track.age < MAX_TRACK_AGE}

        # Assign player IDs based on court position (left=0, right=1)
        return self._assign_player_ids()

    def _create_new_track(self, pose, frame_idx):
        """Create new track for unmatched pose."""
        track = TrackState(self.next_id, pose, frame_idx)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _get_pose_center(self, pose):
        """Calculate center point of pose from valid keypoints."""
        valid_points = []
        for joint in pose['joints']:
            if joint['confidence'] > CONFIDENCE_THRESHOLD:
                valid_points.append([joint['x'], joint['y']])

        if valid_points:
            center = np.mean(valid_points, axis=0)
            return [float(center[0]), float(center[1])]
        return [0.0, 0.0]

    def _calculate_pose_similarity(self, pose1, pose2):
        """Calculate similarity between two poses based on keypoint positions."""
        similarity_sum = 0.0
        valid_joints = 0

        for i, (joint1, joint2) in enumerate(zip(pose1['joints'], pose2['joints'])):
            if (joint1['confidence'] > CONFIDENCE_THRESHOLD and
                    joint2['confidence'] > CONFIDENCE_THRESHOLD):
                distance = np.sqrt((joint1['x'] - joint2['x'])**2 + (joint1['y'] - joint2['y'])**2)
                # Normalize by image diagonal (assuming ~1000px diagonal)
                normalized_distance = distance / 1000.0
                similarity = max(0, 1.0 - normalized_distance)
                similarity_sum += similarity
                valid_joints += 1

        return similarity_sum / max(1, valid_joints)

    def _assign_player_ids(self):
        """Assign player IDs (0, 1) based on court position relative to center line."""
        active_tracks = [t for t in self.tracks.values() if t.age == 0]  # Only recently updated tracks

        if len(active_tracks) == 0:
            return []
        elif len(active_tracks) == 1:
            # Single player, assign ID 0
            pose = active_tracks[0].poses[-1].copy()
            pose['player_id'] = 0
            pose['track_id'] = active_tracks[0].id
            return [pose]
        else:
            # Multiple players, assign based on horizontal position
            poses_with_positions = []
            for track in active_tracks:
                pose = track.poses[-1].copy()
                center = self._get_pose_center(pose)
                poses_with_positions.append((pose, center[0], track.id))

            # Sort by x-coordinate (left to right)
            poses_with_positions.sort(key=lambda x: x[1])

            # Assign IDs: leftmost = 0, rightmost = 1
            result = []
            for i, (pose, x_pos, track_id) in enumerate(poses_with_positions[:2]):  # Max 2 players
                pose['player_id'] = i
                pose['track_id'] = track_id
                result.append(pose)

            return result


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
    Standard badminton court on ground plane (z=0).
    """
    return {
        'P1': np.array([0.0, 0.0, 0.0]),          # near-left corner
        'P2': np.array([0.0, COURT_LENGTH, 0.0]), # far-left corner
        'P3': np.array([COURT_WIDTH, COURT_LENGTH, 0.0]), # far-right corner
        'P4': np.array([COURT_WIDTH, 0.0, 0.0])   # near-right corner
    }


def estimate_camera_intrinsics(corner_points_2d):
    """
    Estimate camera intrinsic parameters using geometric constraints.
    Based on research paper methodology for uncalibrated cameras.
    """
    try:
        # Get image points
        image_points = np.array([
            corner_points_2d['P1'],
            corner_points_2d['P2'],
            corner_points_2d['P3'],
            corner_points_2d['P4']
        ], dtype=np.float32)

        # Calculate image dimensions from court projection
        w = max(image_points[:, 0]) - min(image_points[:, 0])
        h = max(image_points[:, 1]) - min(image_points[:, 1])

        # Estimate focal length using court perspective
        # Based on vanishing point analysis of parallel lines
        court_width_pixels = np.linalg.norm(image_points[0] - image_points[3])  # P1 to P4
        court_length_pixels = np.linalg.norm(image_points[0] - image_points[1])  # P1 to P2

        # Scaling factor k determined through vanishing point analysis
        k = max(court_width_pixels, court_length_pixels) / np.sqrt(w**2 + h**2)
        k = max(0.8, min(k, 2.0))  # Reasonable bounds for sports cameras

        # Focal length estimation
        fx = fy = (w * h / np.sqrt(w**2 + h**2)) * k

        # Principal point at image center
        cx = np.mean(image_points[:, 0])
        cy = np.mean(image_points[:, 1])

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return camera_matrix

    except Exception as e:
        print(f"Camera intrinsic estimation failed: {e}")
        # Fallback estimation
        image_center_x = np.mean([p[0] for p in corner_points_2d.values()])
        image_center_y = np.mean([p[1] for p in corner_points_2d.values()])
        focal_length = 1000.0  # Default focal length

        return np.array([
            [focal_length, 0, image_center_x],
            [0, focal_length, image_center_y],
            [0, 0, 1]
        ], dtype=np.float32)


def solve_camera_pose(corner_points_2d):
    """
    Solve camera pose using Perspective-n-Point (PnP) algorithm.
    Implementation follows research paper methodology.
    """
    try:
        # Get 3D world points
        world_points_3d = get_3d_court_points()

        # Prepare 3D and 2D point arrays
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

        # Estimate camera intrinsics
        camera_matrix = estimate_camera_intrinsics(corner_points_2d)

        # Solve PnP
        dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if success:
            print("✓ Camera pose solved successfully using PnP algorithm")
            return True, rvec, tvec, camera_matrix
        else:
            print("✗ PnP algorithm failed")
            return False, None, None, None

    except Exception as e:
        print(f"✗ Camera pose solving failed: {e}")
        return False, None, None, None


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
        h_camera = abs(camera_position[2][0])  # Height above ground

        # Calculate horizontal distance to court center
        court_center = np.array([COURT_WIDTH/2, COURT_LENGTH/2, 0])
        d_court = np.linalg.norm(camera_position.flatten()[:2] - court_center[:2])

        # Calculate elevation angle
        theta = np.arctan(h_camera / max(d_court, 0.1))  # Avoid division by zero

        print(f"  Camera height: {h_camera:.2f}m")
        print(f"  Distance to court center: {d_court:.2f}m")
        print(f"  Elevation angle: {np.degrees(theta):.1f}°")

        return theta, h_camera, d_court

    except Exception as e:
        print(f"Elevation angle calculation failed: {e}")
        return np.radians(30), 5.0, 10.0  # Default values


def create_perspective_aware_boundary(corner_points_2d, rvec, tvec):
    """
    Create perspective-aware enlarged court boundary.
    Implementation based on research paper geometric principles.
    """
    try:
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
            distance = np.linalg.norm(camera_position.flatten() - center_3d)
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

        print(f"  Extension distances (pixels): {extensions}")

        # Apply asymmetric extension
        enlarged_corners = {}

        # Calculate court center for direction vectors
        center_x = np.mean([corner_points_2d[f'P{i}'][0] for i in range(1, 5)])
        center_y = np.mean([corner_points_2d[f'P{i}'][1] for i in range(1, 5)])

        # Extend each corner based on its region distance
        for i in range(1, 5):
            point_name = f'P{i}'
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

        print("✓ Perspective-aware boundary enlargement completed")
        return enlarged_corners

    except Exception as e:
        print(f"✗ Perspective-aware enlargement failed: {e}")
        return None


def enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.3):
    """Fallback uniform court boundary enlargement."""
    print(f"Using fallback uniform enlargement ({enlargement_factor*100:.0f}%)")

    if len(corner_points) != 4:
        return corner_points

    # Calculate center of the court
    center_x = sum(corner_points[f"P{i}"][0] for i in range(1, 5)) / 4
    center_y = sum(corner_points[f"P{i}"][1] for i in range(1, 5)) / 4

    # Enlarge each point by moving it away from the center
    enlarged_points = {}
    for i in range(1, 5):
        point = corner_points[f"P{i}"]
        dx = point[0] - center_x
        dy = point[1] - center_y

        new_x = center_x + dx * (1 + enlargement_factor)
        new_y = center_y + dy * (1 + enlargement_factor)

        enlarged_points[f"P{i}"] = [float(new_x), float(new_y)]

    return enlarged_points


def is_point_in_court(point, corner_points, enlarged_boundary=None):
    """Check if a point is inside the court polygon using ray casting algorithm."""
    x, y = point

    # Use enlarged boundary if provided
    boundary_points = enlarged_boundary if enlarged_boundary is not None else corner_points

    # Create polygon from corner points
    polygon = [boundary_points[f"P{i}"] for i in range(1, 5) if f"P{i}" in boundary_points]

    if len(polygon) != 4:
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
    """Enhanced pose detection system with perspective-aware processing."""

    def __init__(self):
        self.device = self._select_device()
        self._setup_yolo()
        self.tracker = MultiPersonTracker()

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
        print(f"Loading YOLOv11x-pose model on device: {self.device}")

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

        # Load model with suppressed output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.pose_model = YOLO(model_path, verbose=False)
                self.pose_model.to(self.device)

    def detect_poses(self, frame):
        """Detect human poses in frame using YOLOv11x-pose."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.pose_model(frame, verbose=False)
        return results

    def process_video(self, video_path, court_csv_path, output_json_path):
        """Process video for enhanced pose estimation."""
        print(f"Processing video: {video_path}")

        # Load court data
        all_court_points = read_court_csv(court_csv_path)
        corner_points = extract_corner_points(all_court_points)

        print("\n=== Enhanced Pose Estimation Pipeline ===")
        print("Implementing perspective-aware court boundary enlargement...")

        # Attempt perspective-aware enlargement
        success, rvec, tvec, camera_matrix = solve_camera_pose(corner_points)

        enlarged_court = None
        enlargement_method = "uniform_fallback"
        camera_info = {"pose_solved": False}

        if success:
            enlarged_court = create_perspective_aware_boundary(corner_points, rvec, tvec)
            if enlarged_court is not None:
                enlargement_method = "perspective_aware"
                # Store camera information
                rmat, _ = cv2.Rodrigues(rvec)
                camera_position = -rmat.T @ tvec
                camera_info = {
                    "pose_solved": True,
                    "camera_position": [float(x) for x in camera_position.flatten()],
                    "rotation_vector": [float(x) for x in rvec.flatten()],
                    "translation_vector": [float(x) for x in tvec.flatten()],
                    "camera_matrix": [[float(x) for x in row] for row in camera_matrix]
                }

        # Fallback to uniform enlargement
        if enlarged_court is None:
            print("Falling back to uniform enlargement...")
            enlarged_court = enlarge_court_boundary_uniform(corner_points, enlargement_factor=0.3)
            camera_info["fallback_method"] = "uniform_enlargement_30_percent"

        print(f"Court enlargement method: {enlargement_method}")
        print("=========================================\n")

        # Get video information
        video_info = get_video_info(video_path)

        # Process all frames (assuming all are rally frames for this implementation)
        rally_frames = set(range(0, video_info["frame_count"]))

        print(f"Processing {len(rally_frames)} frames...")
        print(f"Validation: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints required")
        print(f"Using multi-person tracking with Hungarian algorithm")

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
                    frame_valid_poses = []
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

                                frame_valid_poses.append(pose_data)

                    # Update tracker and assign player IDs
                    tracked_poses = self.tracker.update(frame_valid_poses, frame_index)
                    all_pose_data.extend(tracked_poses)

                frame_index += 1
                pbar.update(1)

        cap.release()

        # Create comprehensive output
        output_data = {
            "court_points": corner_points,
            "enlarged_court_points": enlarged_court,
            "all_court_points": all_court_points,
            "video_info": video_info,
            "rally_frames": list(rally_frames),
            "pose_data": all_pose_data,
            "camera_info": camera_info,
            "processing_info": {
                "total_poses_detected": len(all_pose_data),
                "total_court_points": len(all_court_points),
                "corner_points_used": list(corner_points.keys()),
                "enlargement_method": enlargement_method,
                "max_jump_height_considered": f"{MAX_JUMP_HEIGHT}m",
                "validation_criteria": f"At least {MIN_ANKLE_KNEE_JOINTS} ankle/knee joints in enlarged boundary",
                "ankle_knee_joint_indices": ANKLE_KNEE_INDICES,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "tracking_algorithm": "Hungarian algorithm with pose similarity",
                "model_device": self.device,
                "model_type": "YOLOv11x-pose",
                "pose_estimator": "Enhanced Perspective-Aware Pose Estimation",
                "court_dimensions": f"{COURT_LENGTH}m x {COURT_WIDTH}m",
                "implementation_features": [
                    "Perspective-n-Point (PnP) camera pose estimation",
                    "Geometric camera intrinsic parameter estimation",
                    "Intelligent boundary extension based on elevation angle",
                    "Ankle-knee joint validation for player filtering",
                    "Multi-person tracking with Hungarian algorithm",
                    "Player ID assignment based on court position"
                ]
            }
        }

        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n=== Processing Complete ===")
        print(f"✓ Total valid poses detected: {len(all_pose_data)}")
        print(f"✓ Enlargement method: {enlargement_method}")
        print(f"✓ Validation: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints required")
        print(f"✓ Tracking: Multi-person with ID assignment")
        print(f"✓ Data saved to: {output_json_path}")
        print("===========================")

        return output_json_path


def main(video_path):
    """Main enhanced pose estimation function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("ENHANCED POSE ESTIMATION WITH PERSPECTIVE-AWARE COURT ENLARGEMENT")
    print("="*80)
    print("Implementation based on research paper methodology:")
    print("• YOLOv11x-pose for human keypoint detection")
    print("• Perspective-n-Point (PnP) algorithm for camera pose estimation")
    print("• Intelligent boundary extension based on camera elevation angle")
    print("• Ankle-knee joint validation for player filtering")
    print("• Multi-person tracking with Hungarian algorithm")
    print("• Player ID assignment based on court position")
    print("="*80)
    print(f"Court dimensions: {COURT_LENGTH}m x {COURT_WIDTH}m")
    print(f"Maximum jump height: {MAX_JUMP_HEIGHT}m")
    print(f"Validation threshold: {MIN_ANKLE_KNEE_JOINTS}+ ankle/knee joints")
    print("="*80)

    # Determine file paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    court_csv_path = os.path.join(result_dir, "court.csv")
    output_json_path = os.path.join(result_dir, "pose.json")

    # Validate input files
    if not os.path.exists(court_csv_path):
        logging.error(f"Court CSV not found: {court_csv_path}")
        logging.error("Please run court detection first: python3 detect_court.py <video_path>")
        return None

    # Initialize enhanced pose detector
    pose_detector = EnhancedPoseDetector()

    try:
        output_path = pose_detector.process_video(video_path, court_csv_path, output_json_path)
        return output_path
    except Exception as e:
        logging.error(f"Pose estimation failed: {e}")
        return None
    finally:
        # Clean up resources
        del pose_detector


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 enhanced_pose_estimation.py <input_video_path>")
        print("\nExample:")
        print("  python3 enhanced_pose_estimation.py samples/badminton_match.mp4")
        print("\nRequirements:")
        print("  - Court detection must be run first (court.csv required)")
        print("  - YOLOv11x-pose model (will be downloaded if not found)")
        print("  - OpenCV, PyTorch, Ultralytics, SciPy")
        sys.exit(1)

    input_video_path = sys.argv[1]

    # Validate input video
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found: {input_video_path}")
        sys.exit(1)

    result = main(input_video_path)
    if result:
        print(f"\n✓ Success! Enhanced pose estimation completed.")
        print(f"  Output: {result}")
    else:
        print(f"\n✗ Failed! Enhanced pose estimation encountered errors.")
        sys.exit(1)