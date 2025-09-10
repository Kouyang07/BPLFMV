#!/usr/bin/env python3

import sys
import os
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import math


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class PlayerState:
    """Simple player state for dual tracking."""
    player_id: str
    last_position: Tuple[float, float]
    last_frame: int
    velocity: Tuple[float, float]
    confidence_history: deque
    trajectory: deque
    consecutive_detections: int = 0

    def __post_init__(self):
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            self.trajectory = deque(maxlen=15)


class DualPlayerTracker:
    """Simple tracker for exactly 2 badminton players."""

    def __init__(self, court_width: float = 6.1, court_length: float = 13.4, debug: bool = False):
        self.court_width = court_width
        self.court_length = court_length
        self.debug = debug

        # Simple parameters
        self.max_distance_threshold = 2.0  # Maximum distance for assignment
        self.occlusion_max_frames = 30     # Frames to keep tracking during occlusion

        # Two player states exactly
        self.player_0: Optional[PlayerState] = None
        self.player_1: Optional[PlayerState] = None
        self.frame_rate = 30.0
        self.initialized = False

    def set_frame_rate(self, fps: float):
        """Set video frame rate."""
        self.frame_rate = max(1.0, fps)

    def _calculate_position_confidence(self, ankle_detections: List[Dict]) -> float:
        """Calculate position confidence."""
        if not ankle_detections:
            return 0.0

        joint_confidences = [ankle['joint_confidence'] for ankle in ankle_detections]
        avg_confidence = sum(joint_confidences) / len(joint_confidences)

        if len(ankle_detections) == 2:
            ankle1, ankle2 = ankle_detections[0], ankle_detections[1]
            distance = np.sqrt((ankle1['world_x'] - ankle2['world_x'])**2 +
                               (ankle1['world_y'] - ankle2['world_y'])**2)
            if 0.05 <= distance <= 0.6:
                return min(1.0, avg_confidence * 1.15)
            else:
                return avg_confidence * 0.7
        else:
            return avg_confidence * 0.85

    def _calculate_velocity(self, positions: List[Tuple[float, float, int]]) -> Tuple[float, float]:
        """Calculate velocity from position history."""
        if len(positions) < 2:
            return (0.0, 0.0)

        velocities = []
        weights = []

        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]

            dt = (pos2[2] - pos1[2]) / self.frame_rate
            if dt <= 0:
                continue

            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]

            vx = dx / dt
            vy = dy / dt

            speed = math.sqrt(vx*vx + vy*vy)
            if speed <= 8.0:  # Max reasonable velocity
                velocities.append((vx, vy))
                weights.append(math.exp(-i * 0.1))

        if not velocities:
            return (0.0, 0.0)

        total_weight = sum(weights)
        if total_weight == 0:
            return (0.0, 0.0)

        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights)) / total_weight
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights)) / total_weight

        return (avg_vx, avg_vy)

    def _predict_position(self, player_state: PlayerState, target_frame: int) -> Tuple[float, float]:
        """Simple linear position prediction."""
        if not player_state.trajectory:
            return player_state.last_position

        dt = (target_frame - player_state.last_frame) / self.frame_rate
        if dt <= 0:
            return player_state.last_position

        pos_x, pos_y = player_state.last_position
        vel_x, vel_y = player_state.velocity

        # Simple linear prediction with damping
        damping = 0.95 ** dt
        predicted_x = pos_x + vel_x * dt * damping
        predicted_y = pos_y + vel_y * dt * damping

        return (predicted_x, predicted_y)

    def _initialize_players(self, detections: List[Dict], frame_idx: int):
        """Initialize the two players with first detections."""
        if len(detections) >= 2:
            # Sort detections by Y position (court length) to get consistent assignment
            sorted_dets = sorted(detections, key=lambda d: d['center_position']['y'])

            # Player 0 gets the detection with smaller Y (closer to net)
            det0 = sorted_dets[0]
            pos0 = (det0['center_position']['x'], det0['center_position']['y'])
            confidence0 = self._calculate_position_confidence(det0['ankles'])

            self.player_0 = PlayerState(
                player_id='player_0',
                last_position=pos0,
                last_frame=frame_idx,
                velocity=(0.0, 0.0),
                confidence_history=deque([confidence0], maxlen=10),
                trajectory=deque([(*pos0, frame_idx)], maxlen=15),
                consecutive_detections=1
            )

            # Player 1 gets the detection with larger Y (farther from net)
            det1 = sorted_dets[1]
            pos1 = (det1['center_position']['x'], det1['center_position']['y'])
            confidence1 = self._calculate_position_confidence(det1['ankles'])

            self.player_1 = PlayerState(
                player_id='player_1',
                last_position=pos1,
                last_frame=frame_idx,
                velocity=(0.0, 0.0),
                confidence_history=deque([confidence1], maxlen=10),
                trajectory=deque([(*pos1, frame_idx)], maxlen=15),
                consecutive_detections=1
            )

            self.initialized = True

            if self.debug:
                print(f"Frame {frame_idx}: Initialized player_0 at ({pos0[0]:.1f},{pos0[1]:.1f}) and player_1 at ({pos1[0]:.1f},{pos1[1]:.1f})")

        elif len(detections) == 1:
            # Only one detection - initialize player_0 first
            det0 = detections[0]
            pos0 = (det0['center_position']['x'], det0['center_position']['y'])
            confidence0 = self._calculate_position_confidence(det0['ankles'])

            self.player_0 = PlayerState(
                player_id='player_0',
                last_position=pos0,
                last_frame=frame_idx,
                velocity=(0.0, 0.0),
                confidence_history=deque([confidence0], maxlen=10),
                trajectory=deque([(*pos0, frame_idx)], maxlen=15),
                consecutive_detections=1
            )

            if self.debug:
                print(f"Frame {frame_idx}: Initialized player_0 at ({pos0[0]:.1f},{pos0[1]:.1f}), waiting for player_1")

    def _assign_detections_simple(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Simple assignment for exactly 2 players using distance."""
        assignments = {}

        if not detections:
            return assignments

        # Get current player positions (predicted if occluded)
        player_positions = {}

        if self.player_0:
            frames_gap = frame_idx - self.player_0.last_frame
            if frames_gap <= self.occlusion_max_frames:
                if frames_gap > 0:
                    player_positions['player_0'] = self._predict_position(self.player_0, frame_idx)
                else:
                    player_positions['player_0'] = self.player_0.last_position

        if self.player_1:
            frames_gap = frame_idx - self.player_1.last_frame
            if frames_gap <= self.occlusion_max_frames:
                if frames_gap > 0:
                    player_positions['player_1'] = self._predict_position(self.player_1, frame_idx)
                else:
                    player_positions['player_1'] = self.player_1.last_position

        if not player_positions:
            return assignments

        # For each detection, find closest player
        used_players = set()

        for det_idx, detection in enumerate(detections):
            det_pos = (detection['center_position']['x'], detection['center_position']['y'])

            best_player = None
            best_distance = float('inf')

            for player_id, player_pos in player_positions.items():
                if player_id in used_players:
                    continue

                distance = np.sqrt((det_pos[0] - player_pos[0])**2 + (det_pos[1] - player_pos[1])**2)

                if distance < best_distance and distance <= self.max_distance_threshold:
                    best_distance = distance
                    best_player = player_id

            if best_player:
                assignments[det_idx] = best_player
                used_players.add(best_player)

                if self.debug:
                    print(f"Frame {frame_idx}: Assigned detection {det_idx} at ({det_pos[0]:.1f},{det_pos[1]:.1f}) to {best_player} (dist: {best_distance:.2f}m)")
            else:
                # Handle case where we need to initialize second player
                if not self.player_1 and len(detections) >= 2:
                    assignments[det_idx] = 'player_1'
                    if self.debug:
                        print(f"Frame {frame_idx}: Assigning detection {det_idx} to initialize player_1")
                elif self.debug:
                    print(f"Frame {frame_idx}: No assignment for detection {det_idx} at ({det_pos[0]:.1f},{det_pos[1]:.1f}) - distance too large")

        return assignments

    def _update_player_state(self, player_id: str, detection: Dict, frame_idx: int):
        """Update player state."""
        center_pos = detection['center_position']
        position = (center_pos['x'], center_pos['y'])
        confidence = self._calculate_position_confidence(detection['ankles'])

        if player_id == 'player_0':
            player_state = self.player_0
        elif player_id == 'player_1':
            if not self.player_1:
                # Initialize player_1
                self.player_1 = PlayerState(
                    player_id='player_1',
                    last_position=position,
                    last_frame=frame_idx,
                    velocity=(0.0, 0.0),
                    confidence_history=deque([confidence], maxlen=10),
                    trajectory=deque([(*position, frame_idx)], maxlen=15),
                    consecutive_detections=1
                )
                if self.debug:
                    print(f"Frame {frame_idx}: Initialized player_1 at ({position[0]:.1f},{position[1]:.1f})")
                return
            player_state = self.player_1
        else:
            return  # Unknown player_id

        if player_state:
            # Update trajectory
            player_state.trajectory.append((position[0], position[1], frame_idx))

            # Calculate velocity
            trajectory_list = list(player_state.trajectory)
            player_state.velocity = self._calculate_velocity(trajectory_list)

            # Update confidence
            player_state.confidence_history.append(confidence)

            # Update position and frame
            player_state.last_position = position
            player_state.last_frame = frame_idx
            player_state.consecutive_detections += 1

    def process_frame_detections(self, frame_ankle_data: Dict[int, List[Dict]], frame_idx: int) -> Dict[str, Dict[str, Any]]:
        """Process frame with simple dual player tracking."""
        if not frame_ankle_data:
            return {}

        # Convert to detection format
        detections = []
        for person_id, ankle_detections in frame_ankle_data.items():
            if not ankle_detections:
                continue

            avg_x = sum(ankle['world_x'] for ankle in ankle_detections) / len(ankle_detections)
            avg_y = sum(ankle['world_y'] for ankle in ankle_detections) / len(ankle_detections)

            detections.append({
                'person_id': person_id,
                'ankles': ankle_detections,
                'center_position': {'x': float(avg_x), 'y': float(avg_y)}
            })

        if self.debug:
            active_count = int(bool(self.player_0)) + int(bool(self.player_1))
            print(f"Frame {frame_idx}: {len(detections)} detections, {active_count}/2 players initialized")

        # Initialize players if not done yet
        if not self.initialized:
            self._initialize_players(detections, frame_idx)
            if not self.initialized and len(detections) == 0:
                return {}

        # Simple assignment for 2 players
        assignments = self._assign_detections_simple(detections, frame_idx)

        # Build result and update states
        frame_players = {}

        for det_idx, player_id in assignments.items():
            detection = detections[det_idx]

            frame_players[player_id] = {
                'ankles': detection['ankles'],
                'center_position': detection['center_position']
            }

            self._update_player_state(player_id, detection, frame_idx)

        return frame_players

    def get_final_player_mapping(self) -> Dict[str, str]:
        """Get mapping to standard player_0, player_1 format."""
        # For dual tracking, mapping is direct
        mapping = {}
        if self.player_0:
            mapping['player_0'] = 'player_0'
        if self.player_1:
            mapping['player_1'] = 'player_1'
        return mapping

    def print_tracking_stats(self):
        """Print dual player tracking statistics."""
        print("\\n=== DUAL PLAYER TRACKING STATISTICS ===")
        print(f"Initialized: {self.initialized}")
        print(f"Distance threshold: {self.max_distance_threshold:.2f}m")

        if self.player_0:
            frames_tracked = len(self.player_0.trajectory)
            avg_confidence = (sum(self.player_0.confidence_history) / len(self.player_0.confidence_history)
                              if self.player_0.confidence_history else 0)
            last_pos = self.player_0.last_position
            print(f"Player 0: {frames_tracked} frames, conf={avg_confidence:.2f}, pos=({last_pos[0]:.1f},{last_pos[1]:.1f})")
        else:
            print("Player 0: Not initialized")

        if self.player_1:
            frames_tracked = len(self.player_1.trajectory)
            avg_confidence = (sum(self.player_1.confidence_history) / len(self.player_1.confidence_history)
                              if self.player_1.confidence_history else 0)
            last_pos = self.player_1.last_position
            print(f"Player 1: {frames_tracked} frames, conf={avg_confidence:.2f}, pos=({last_pos[0]:.1f},{last_pos[1]:.1f})")
        else:
            print("Player 1: Not initialized")

        print("=" * 45)


class Enhanced3DAnkleTracker:
    """Enhanced ankle tracker with proper 3D geometric ankle offset calculation."""

    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16
    CONFIDENCE_THRESHOLD = 0.5
    ANKLE_HEIGHT_METERS = 0.04  # 4cm ankle height above ground

    def __init__(self, video_path: str, debug: bool = False, reflect_y: bool = False):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.calibration_file = self.results_dir / "calibration.csv"
        self.output_file = self.results_dir / "positions.json"
        self.debug = debug
        self.reflect_y = reflect_y

        self.pose_data = None
        self.court_points = None
        self.video_info = None
        self.homography_matrix = None

        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_height = None
        self.camera_tilt_angle = None  # θ in radians
        self.fy = None  # Vertical focal length
        self.v0 = None  # Principal point vertical coordinate
        self.calibration_available = False

        self.player_tracker = DualPlayerTracker(
            court_width=self.COURT_WIDTH,
            court_length=self.COURT_LENGTH,
            debug=debug
        )

        self.frame_data_internal = {}

    def load_calibration_data(self) -> None:
        """Load enhanced calibration data with camera tilt angle."""
        if not self.calibration_file.exists():
            if self.debug:
                print("No calibration data - using basic homography")
            return

        try:
            calibration_params = {}
            with open(self.calibration_file, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if not row or row[0].startswith('#') or len(row) < 2:
                        continue

                    key = row[0].strip()
                    value = row[1].strip()

                    try:
                        if key in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']:
                            calibration_params[key] = float(value)
                        elif key == 'camera_height_m':
                            calibration_params['camera_height_m'] = float(value)
                        elif key == 'camera_tilt_deg':
                            calibration_params['camera_tilt_deg'] = float(value)
                        elif key == 'reprojection_error_px':
                            calibration_params['reprojection_error_px'] = float(value)
                    except (ValueError, IndexError):
                        continue

            # Set up camera matrix
            if all(param in calibration_params for param in ['fx', 'fy', 'cx', 'cy']):
                self.camera_matrix = np.array([
                    [calibration_params['fx'], 0, calibration_params['cx']],
                    [0, calibration_params['fy'], calibration_params['cy']],
                    [0, 0, 1]
                ], dtype=np.float32)

                self.fy = calibration_params['fy']
                self.v0 = calibration_params['cy']

            # Set up distortion coefficients
            dist_coeffs = []
            for param in ['k1', 'k2', 'p1', 'p2', 'k3']:
                if param in calibration_params:
                    dist_coeffs.append(calibration_params[param])
            if dist_coeffs:
                self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

            # Camera geometry parameters
            self.camera_height = calibration_params.get('camera_height_m')
            camera_tilt_deg = calibration_params.get('camera_tilt_deg', 0.0)
            self.camera_tilt_angle = math.radians(camera_tilt_deg)  # Convert to radians

            reprojection_error = calibration_params.get('reprojection_error_px')

            # Check if we have sufficient calibration for 3D correction
            self.calibration_available = (
                    self.camera_matrix is not None and
                    self.camera_height is not None and
                    self.fy is not None and
                    self.v0 is not None and
                    (reprojection_error is None or reprojection_error < 30)
            )

            if self.calibration_available:
                if self.debug:
                    print(f"Enhanced 3D calibration loaded:")
                    print(f"  Camera height: {self.camera_height:.2f}m")
                    print(f"  Camera tilt: {camera_tilt_deg:.1f}°")
                    print(f"  Focal length (fy): {self.fy:.1f}px")
                    print(f"  Reprojection error: {reprojection_error:.1f}px")
            else:
                if self.debug:
                    print("Insufficient calibration data for 3D correction")

        except Exception as e:
            if self.debug:
                print(f"Calibration load error: {e}")
            self.calibration_available = False

    def calculate_3d_ankle_offset_pixel_correction(self, ankle_pixel: Tuple[float, float]) -> float:
        """
        Calculate proper 3D ankle offset using the paper's approach.

        This implements the equations from the paper:
        Y_c = Y_w * sin(θ) + h * cos(θ)
        Z_c = Y_w * cos(θ) - h * sin(θ)
        Δv(s) = f_y * [(Y_c - s*cos(θ))/(Z_c + s*sin(θ)) - Y_c/Z_c]

        Returns: pixel offset in v (vertical) direction to subtract from ankle position
        """
        if not self.calibration_available:
            return 12.0  # Fallback to fixed pixel offset

        try:
            # First, transform ankle pixel to rough world coordinates to estimate Y_w
            ankle_point = np.array([[ankle_pixel]], dtype=np.float32)
            rough_world = cv2.perspectiveTransform(ankle_point, self.homography_matrix)
            Y_w = float(rough_world[0][0][1])  # Y-distance along court from camera

            # Camera geometry parameters
            h = self.camera_height
            theta = self.camera_tilt_angle
            s = self.ANKLE_HEIGHT_METERS

            # Calculate camera frame coordinates for ground point (s=0)
            Y_c = Y_w * math.sin(theta) + h * math.cos(theta)
            Z_c = Y_w * math.cos(theta) - h * math.sin(theta)

            # Avoid division by zero
            if abs(Z_c) < 0.1:
                Z_c = 0.1 if Z_c >= 0 else -0.1

            # Calculate the pixel offset using the paper's formula
            # This is the difference between where elevated ankle appears vs ground position
            ground_projection = Y_c / Z_c
            elevated_projection = (Y_c - s * math.cos(theta)) / (Z_c + s * math.sin(theta))

            # Convert to pixel coordinates (note: we want the magnitude of correction)
            delta_v = self.fy * (elevated_projection - ground_projection)

            # Clamp to reasonable values (safety check)
            delta_v = max(-50.0, min(50.0, delta_v))

            if self.debug and abs(delta_v) > 0.1:
                print(f"3D offset calc: Y_w={Y_w:.1f}m, delta_v={delta_v:.1f}px")

            return float(delta_v)

        except Exception as e:
            if self.debug:
                print(f"3D offset calculation failed: {e}")
            return 12.0  # Fallback

    def load_pose_data(self) -> None:
        """Load pose detection data."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.video_info = data.get('video_info', {})
        self.court_points = data.get('court_points', {}) or data.get('all_court_points', {})

        fps = self.video_info.get('fps', 30.0)
        self.player_tracker.set_frame_rate(fps)

        if not self.court_points:
            raise ValueError("No court points found")

        if self.debug:
            print(f"Loaded {len(data.get('pose_data', []))} pose detections")

    def calculate_homography(self) -> None:
        """Calculate homography from court corners."""
        required_corners = ['P1', 'P2', 'P3', 'P4']

        missing_corners = [corner for corner in required_corners if corner not in self.court_points]
        if missing_corners:
            raise ValueError(f"Missing court corners: {missing_corners}")

        image_points = []
        for corner in required_corners:
            coords = self.court_points[corner]
            if isinstance(coords, list) and len(coords) >= 2:
                image_points.append([float(coords[0]), float(coords[1])])
            elif isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                image_points.append([float(coords['x']), float(coords['y'])])
            else:
                raise ValueError(f"Invalid coordinate format for {corner}: {coords}")

        image_points = np.array(image_points, dtype=np.float32)

        world_points = np.array([
            [0, 0],
            [0, self.COURT_LENGTH],
            [self.COURT_WIDTH, self.COURT_LENGTH],
            [self.COURT_WIDTH, 0]
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

    def undistort_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Undistort pixel point if calibration available."""
        if not self.calibration_available or self.dist_coeffs is None:
            return point

        try:
            point_array = np.array([[point]], dtype=np.float32)
            undistorted = cv2.undistortPoints(
                point_array, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
            )
            return float(undistorted[0][0][0]), float(undistorted[0][0][1])
        except Exception:
            return point

    def calculate_ankle_ground_position(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate ankle ground position with proper 3D geometric correction."""
        try:
            # Step 1: Undistort the pixel if calibration is available
            if self.calibration_available:
                undistorted_pixel = self.undistort_point(ankle_pixel)
            else:
                undistorted_pixel = ankle_pixel

            # Step 2: Calculate the proper 3D geometric offset
            pixel_offset = self.calculate_3d_ankle_offset_pixel_correction(undistorted_pixel)

            # Step 3: Apply the offset (positive offset moves point down in image)
            # Since we want to find where the ground contact would be, we need to offset downward
            corrected_pixel = (undistorted_pixel[0], undistorted_pixel[1] + abs(pixel_offset))

            # Step 4: Transform to world coordinates
            point = np.array([[corrected_pixel]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)

            world_x = float(world_point[0][0][0])
            world_y = float(world_point[0][0][1])

            # Step 5: Apply reflection if requested
            if self.reflect_y:
                world_x = self.COURT_WIDTH - world_x

            # Step 6: Soft boundary correction
            if world_x < -1.0:
                world_x = -1.0 + (world_x + 1.0) * 0.3
            elif world_x > self.COURT_WIDTH + 1.0:
                world_x = self.COURT_WIDTH + 1.0 + (world_x - self.COURT_WIDTH - 1.0) * 0.3

            if world_y < -1.0:
                world_y = -1.0 + (world_y + 1.0) * 0.3
            elif world_y > self.COURT_LENGTH + 1.0:
                world_y = self.COURT_LENGTH + 1.0 + (world_y - self.COURT_LENGTH - 1.0) * 0.3

            return world_x, world_y

        except Exception as e:
            if self.debug:
                print(f"Position calculation failed: {e}")
            return 0.0, 0.0

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return float(joint['x']), float(joint['y'])
        return None

    def process_person_ankles(self, joints: List[Dict]) -> List[Dict[str, Any]]:
        """Process ankle positions for a person with enhanced 3D correction."""
        ankle_detections = []

        # Process left ankle
        ankle_left_pixel = self.extract_joint_position(joints, self.ANKLE_LEFT)
        if ankle_left_pixel:
            left_world_x, left_world_y = self.calculate_ankle_ground_position(ankle_left_pixel)

            left_confidence = 0.0
            for joint in joints:
                if joint['joint_index'] == self.ANKLE_LEFT:
                    left_confidence = float(joint['confidence'])
                    break

            ankle_detections.append({
                'ankle_side': 'left',
                'world_x': float(left_world_x),
                'world_y': float(left_world_y),
                'joint_confidence': left_confidence,
                'method': 'enhanced_3d_geometric'
            })

        # Process right ankle
        ankle_right_pixel = self.extract_joint_position(joints, self.ANKLE_RIGHT)
        if ankle_right_pixel:
            right_world_x, right_world_y = self.calculate_ankle_ground_position(ankle_right_pixel)

            right_confidence = 0.0
            for joint in joints:
                if joint['joint_index'] == self.ANKLE_RIGHT:
                    right_confidence = float(joint['confidence'])
                    break

            ankle_detections.append({
                'ankle_side': 'right',
                'world_x': float(right_world_x),
                'world_y': float(right_world_y),
                'joint_confidence': right_confidence,
                'method': 'enhanced_3d_geometric'
            })

        return ankle_detections

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> None:
        """Process all people in a frame."""
        frame_ankle_data = {}

        for person_id, human_data in enumerate(frame_data):
            joints = human_data.get('joints', [])
            person_ankles = self.process_person_ankles(joints)
            if person_ankles:
                frame_ankle_data[person_id] = person_ankles

        if frame_ankle_data:
            player_assignments = self.player_tracker.process_frame_detections(frame_ankle_data, frame_index)
            if player_assignments:
                self.frame_data_internal[frame_index] = player_assignments

    def process_all_frames(self) -> None:
        """Process all frames with enhanced 3D tracking."""
        pose_data = self.pose_data.get('pose_data', [])

        if not pose_data:
            print("No pose data found")
            return

        frames_data = {}
        for entry in pose_data:
            frame_idx = entry['frame_index']
            if frame_idx not in frames_data:
                frames_data[frame_idx] = []
            frames_data[frame_idx].append(entry)

        if self.debug:
            method = "enhanced 3D geometric" if self.calibration_available else "basic homography"
            print(f"Processing {len(frames_data)} frames with {method} correction")

        for frame_idx in sorted(frames_data.keys()):
            frame_data = frames_data[frame_idx]
            self.process_frame(frame_data, frame_idx)

        if self.debug:
            print(f"Processed {len(self.frame_data_internal)} frames with detections")
            self.player_tracker.print_tracking_stats()

    def convert_to_standard_format(self) -> Dict[str, Dict]:
        """Convert to standard player_0, player_1 format."""
        player_mapping = self.player_tracker.get_final_player_mapping()

        if self.debug:
            print(f"Player mapping: {player_mapping}")

        standard_frame_data = {}
        for frame_idx, frame_data in self.frame_data_internal.items():
            standard_frame = {}
            for internal_id, detection in frame_data.items():
                if internal_id in player_mapping:
                    standard_id = player_mapping[internal_id]
                    standard_frame[standard_id] = detection

            if standard_frame:
                standard_frame_data[str(frame_idx)] = standard_frame

        return standard_frame_data

    def save_results(self) -> None:
        """Save results with enhanced tracking metadata."""
        frame_data_dict = self.convert_to_standard_format()

        total_frames_with_data = len(frame_data_dict)
        total_ankle_detections = 0
        player_0_detections = 0
        player_1_detections = 0

        for frame_data in frame_data_dict.values():
            if 'player_0' in frame_data:
                player_0_detections += len(frame_data['player_0']['ankles'])
            if 'player_1' in frame_data:
                player_1_detections += len(frame_data['player_1']['ankles'])

        total_ankle_detections = player_0_detections + player_1_detections

        # Determine method used
        method = "enhanced_3d_geometric" if self.calibration_available else "basic_homography"

        output_data = {
            'video_info': {
                'video_name': self.video_name,
                'frame_count': self.video_info.get('frame_count', 0),
                'fps': self.video_info.get('fps', 0),
                'width': self.video_info.get('width', 0),
                'height': self.video_info.get('height', 0)
            },
            'court_info': {
                'width_meters': float(self.COURT_WIDTH),
                'length_meters': float(self.COURT_LENGTH),
                'coordinate_system': 'Origin at top-left corner (P1), X=width, Y=length'
            },
            'tracking_summary': {
                'frames_with_ankle_data': total_frames_with_data,
                'total_ankle_detections': total_ankle_detections,
                'player_0_detections': player_0_detections,
                'player_1_detections': player_1_detections,
                'method': method,
                'players_created': 2,
                'ankle_offset_method': 'proper_3d_geometric' if self.calibration_available else 'fixed_pixel',
                'ankle_height_meters': self.ANKLE_HEIGHT_METERS
            },
            'calibration_info': {
                'calibration_available': self.calibration_available,
                'camera_height_m': self.camera_height,
                'camera_tilt_deg': math.degrees(self.camera_tilt_angle) if self.camera_tilt_angle else None,
                'vertical_focal_length': self.fy,
                'uses_3d_correction': self.calibration_available
            },
            'frame_data': frame_data_dict
        }

        output_data_clean = convert_numpy_types(output_data)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data_clean, f, indent=2)

        print(f"Results saved: {self.output_file}")
        print(f"Frames with data: {total_frames_with_data}")
        print(f"Player 0: {player_0_detections} detections")
        print(f"Player 1: {player_1_detections} detections")
        print(f"Method: Enhanced 3D geometric ankle correction" if self.calibration_available else "Method: Basic homography with fixed offset")

    def run(self) -> None:
        """Run enhanced dual tracking pipeline."""
        print(f"Starting enhanced dual player tracking: {self.video_name}")

        try:
            self.load_calibration_data()
            self.load_pose_data()
            self.calculate_homography()
            self.process_all_frames()
            self.save_results()

            if self.calibration_available:
                print("Enhanced 3D geometric tracking completed!")
            else:
                print("Basic tracking completed (no 3D calibration available)!")

        except Exception as e:
            print(f"Error during processing: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_location.py <video_file_path> [--debug] [--reflect-y]")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv
    reflect_y = "--reflect-y" in sys.argv

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    tracker = Enhanced3DAnkleTracker(video_path, debug=debug, reflect_y=reflect_y)
    tracker.run()


if __name__ == "__main__":
    main()