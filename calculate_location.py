#!/usr/bin/env python3
"""
Enhanced Badminton Player Ankle Tracking Script with Advanced Robust Player ID System

Tracks individual ankle positions using enhanced homography approach with advanced
player identification that maintains consistency across frames and handles occlusion robustly.

Key Features:
- Multi-frame temporal consistency for player IDs
- Advanced trajectory-based assignment using Hungarian algorithm
- Robust occlusion handling with motion prediction
- Enhanced homography with calibration improvements
- Individual ankle position tracking (left and right separately)
- Confidence-based validation and error correction
- Adaptive tracking parameters based on scene dynamics

Compatible with visualize.py stage 3 visualization.

Usage: python calculate_location.py <video_file_path> [--debug]
"""

import sys
import os
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
import math


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    """Represents the state of a tracked player."""
    player_id: str
    last_position: Tuple[float, float]
    last_frame: int
    velocity: Tuple[float, float]
    confidence_history: deque
    trajectory: deque
    occlusion_count: int = 0
    predicted_position: Optional[Tuple[float, float]] = None
    tracking_quality: float = 1.0

    def __post_init__(self):
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            self.trajectory = deque(maxlen=15)


class AdvancedPlayerTracker:
    """Advanced player ID tracking with robust occlusion handling."""

    def __init__(self, court_width: float = 6.1, court_length: float = 13.4, debug: bool = False):
        self.court_width = court_width
        self.court_length = court_length
        self.debug = debug

        # Enhanced tracking parameters
        self.max_distance_threshold = 1.5  # meters - reduced for better accuracy
        self.trajectory_history_frames = 15  # increased for better prediction
        self.occlusion_max_frames = 12  # increased for better occlusion handling
        self.confidence_history_frames = 10

        # Adaptive parameters
        self.velocity_weight = 0.4
        self.position_weight = 0.6
        self.confidence_weight = 0.3
        self.temporal_weight = 0.2

        # Motion prediction parameters
        self.max_velocity = 8.0  # m/s - maximum realistic player velocity
        self.acceleration_limit = 15.0  # m/s² - maximum realistic acceleration
        self.prediction_frames = 3  # frames to predict ahead

        # Player state tracking
        self.player_states: Dict[str, PlayerState] = {}
        self.next_player_id = 0
        self.frame_rate = 30.0  # default, will be updated

        # Occlusion handling
        self.occlusion_zones = []  # zones where occlusion commonly occurs
        self.global_motion_estimate = (0.0, 0.0)  # global motion compensation

    def set_frame_rate(self, fps: float):
        """Set the video frame rate for motion calculations."""
        self.frame_rate = max(1.0, fps)

    def _calculate_velocity(self, positions: List[Tuple[float, float, int]]) -> Tuple[float, float]:
        """Calculate velocity from position history."""
        if len(positions) < 2:
            return (0.0, 0.0)

        # Use weighted average of recent velocities
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

            # Check for realistic velocity
            speed = math.sqrt(vx*vx + vy*vy)
            if speed <= self.max_velocity:
                velocities.append((vx, vy))
                weights.append(math.exp(-i * 0.1))  # Exponential decay for older velocities

        if not velocities:
            return (0.0, 0.0)

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return (0.0, 0.0)

        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights)) / total_weight
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights)) / total_weight

        return (avg_vx, avg_vy)

    def _predict_position(self, player_state: PlayerState, target_frame: int) -> Tuple[float, float]:
        """Predict player position using motion model."""
        if not player_state.trajectory:
            return player_state.last_position

        dt = (target_frame - player_state.last_frame) / self.frame_rate
        if dt <= 0:
            return player_state.last_position

        # Use Kalman-like prediction with velocity and acceleration
        pos_x, pos_y = player_state.last_position
        vel_x, vel_y = player_state.velocity

        # Simple ballistic prediction with damping
        damping = 0.95 ** dt  # velocity decay over time
        predicted_x = pos_x + vel_x * dt * damping
        predicted_y = pos_y + vel_y * dt * damping

        # Apply boundary constraints with soft limits
        predicted_x = np.clip(predicted_x, -1.5, self.court_width + 1.5)
        predicted_y = np.clip(predicted_y, -1.5, self.court_length + 1.5)

        return (predicted_x, predicted_y)

    def _calculate_position_confidence(self, ankle_detections: List[Dict]) -> float:
        """Calculate overall position confidence based on ankle detections."""
        if not ankle_detections:
            return 0.0

        # Average joint confidence
        joint_confidences = [ankle['joint_confidence'] for ankle in ankle_detections]
        avg_confidence = sum(joint_confidences) / len(joint_confidences)

        # Adjust based on number of ankles detected
        if len(ankle_detections) == 2:
            # Check ankle distance consistency
            ankle1, ankle2 = ankle_detections[0], ankle_detections[1]
            distance = np.sqrt((ankle1['world_x'] - ankle2['world_x'])**2 +
                               (ankle1['world_y'] - ankle2['world_y'])**2)
            # Typical distance between ankles is 0.1-0.4m
            if 0.05 <= distance <= 0.6:
                return min(1.0, avg_confidence * 1.15)  # Bonus for realistic ankle distance
            else:
                return avg_confidence * 0.7  # Penalty for unrealistic distance
        else:
            return avg_confidence * 0.85  # Penalty for single ankle

    def _calculate_tracking_quality(self, player_state: PlayerState) -> float:
        """Calculate overall tracking quality for a player."""
        if not player_state.confidence_history:
            return 0.5

        # Recent confidence trend
        recent_confidences = list(player_state.confidence_history)
        avg_confidence = sum(recent_confidences) / len(recent_confidences)

        # Consistency bonus/penalty
        if len(recent_confidences) > 3:
            confidence_std = np.std(recent_confidences)
            consistency_factor = max(0.5, 1.0 - confidence_std)
        else:
            consistency_factor = 1.0

        # Occlusion penalty
        occlusion_factor = max(0.3, 1.0 - player_state.occlusion_count * 0.1)

        # Trajectory smoothness
        smoothness_factor = 1.0
        if len(player_state.trajectory) > 4:
            positions = [(t[0], t[1]) for t in player_state.trajectory]
            velocities = []
            for i in range(len(positions) - 1):
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                velocities.append(math.sqrt(dx*dx + dy*dy))

            if velocities:
                vel_std = np.std(velocities)
                smoothness_factor = max(0.5, 1.0 - vel_std * 0.5)

        quality = avg_confidence * consistency_factor * occlusion_factor * smoothness_factor
        return np.clip(quality, 0.0, 1.0)

    def _calculate_assignment_cost(self, detection: Dict, player_state: PlayerState, frame_idx: int) -> float:
        """Calculate enhanced cost of assigning a detection to a specific player."""
        detection_pos = (detection['center_position']['x'], detection['center_position']['y'])

        # Spatial cost
        if player_state.predicted_position:
            predicted_pos = player_state.predicted_position
        else:
            predicted_pos = self._predict_position(player_state, frame_idx)

        spatial_distance = np.sqrt((detection_pos[0] - predicted_pos[0])**2 +
                                   (detection_pos[1] - predicted_pos[1])**2)

        # Velocity consistency cost
        velocity_cost = 0.0
        if len(player_state.trajectory) >= 2:
            dt = (frame_idx - player_state.last_frame) / self.frame_rate
            if dt > 0:
                implied_velocity = (
                    (detection_pos[0] - player_state.last_position[0]) / dt,
                    (detection_pos[1] - player_state.last_position[1]) / dt
                )

                vel_diff_x = implied_velocity[0] - player_state.velocity[0]
                vel_diff_y = implied_velocity[1] - player_state.velocity[1]
                velocity_cost = math.sqrt(vel_diff_x*vel_diff_x + vel_diff_y*vel_diff_y) * 0.1

        # Temporal cost (higher for longer gaps)
        frames_gap = frame_idx - player_state.last_frame
        temporal_cost = min(2.0, frames_gap * 0.05)

        # Confidence cost
        detection_confidence = self._calculate_position_confidence(detection['ankles'])
        confidence_cost = (1.0 - detection_confidence) * 0.4

        # Tracking quality cost
        quality_cost = (1.0 - player_state.tracking_quality) * 0.3

        # Occlusion penalty
        occlusion_cost = min(1.0, player_state.occlusion_count * 0.1)

        total_cost = (spatial_distance * self.position_weight +
                      velocity_cost * self.velocity_weight +
                      temporal_cost * self.temporal_weight +
                      confidence_cost * self.confidence_weight +
                      quality_cost * 0.2 +
                      occlusion_cost * 0.15)

        if self.debug:
            print(f"    Cost for {player_state.player_id}: spatial={spatial_distance:.2f}, "
                  f"velocity={velocity_cost:.2f}, temporal={temporal_cost:.2f}, "
                  f"confidence={confidence_cost:.2f}, quality={quality_cost:.2f}, "
                  f"occlusion={occlusion_cost:.2f}, total={total_cost:.2f}")

        return total_cost

    def _assign_detections_to_players(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Assign detections to players using enhanced Hungarian algorithm."""
        if not detections:
            return {}

        # Update predictions for all active players
        active_players = []
        for player_id, player_state in self.player_states.items():
            frames_gap = frame_idx - player_state.last_frame
            if frames_gap <= self.occlusion_max_frames:
                player_state.predicted_position = self._predict_position(player_state, frame_idx)
                active_players.append(player_id)

        # Create extended assignment matrix
        max_assignments = max(len(detections), len(active_players))
        all_player_ids = active_players + [f"new_{i}" for i in range(max_assignments - len(active_players))]

        if not all_player_ids:
            # No existing players, create new ones
            assignments = {}
            for i in range(len(detections)):
                new_id = f"player_{self.next_player_id}"
                self.next_player_id += 1
                assignments[i] = new_id
            return assignments

        # Build enhanced cost matrix
        cost_matrix = np.full((len(detections), len(all_player_ids)), np.inf)

        for det_idx, detection in enumerate(detections):
            for player_idx, player_id in enumerate(all_player_ids):
                if player_id.startswith("new_"):
                    # Cost for new player (prefer existing players)
                    detection_confidence = self._calculate_position_confidence(detection['ankles'])
                    new_player_cost = 2.5 - detection_confidence * 0.5
                    cost_matrix[det_idx, player_idx] = new_player_cost
                else:
                    player_state = self.player_states[player_id]
                    cost = self._calculate_assignment_cost(detection, player_state, frame_idx)

                    # Apply distance threshold with adaptive scaling
                    threshold = self.max_distance_threshold * (1.0 + player_state.occlusion_count * 0.2)
                    if cost <= threshold:
                        cost_matrix[det_idx, player_idx] = cost

        # Solve assignment problem
        try:
            det_indices, player_indices = linear_sum_assignment(cost_matrix)
            assignments = {}

            for det_idx, player_idx in zip(det_indices, player_indices):
                if cost_matrix[det_idx, player_idx] < np.inf:
                    player_id = all_player_ids[player_idx]
                    if player_id.startswith("new_"):
                        # Create new player
                        new_id = f"player_{self.next_player_id}"
                        self.next_player_id += 1
                        assignments[det_idx] = new_id
                    else:
                        assignments[det_idx] = player_id
                else:
                    # Create new player for unassignable detection
                    new_id = f"player_{self.next_player_id}"
                    self.next_player_id += 1
                    assignments[det_idx] = new_id

            return assignments

        except Exception as e:
            if self.debug:
                print(f"Hungarian assignment failed: {e}, using enhanced fallback")
            return self._enhanced_fallback_assignment(detections, frame_idx)

    def _enhanced_fallback_assignment(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Enhanced fallback assignment with better occlusion handling."""
        assignments = {}
        used_players = set()

        # Sort detections by confidence (process high-confidence detections first)
        detection_with_indices = [(i, det) for i, det in enumerate(detections)]
        detection_with_indices.sort(key=lambda x: self._calculate_position_confidence(x[1]['ankles']), reverse=True)

        for det_idx, detection in detection_with_indices:
            best_player = None
            best_cost = float('inf')

            # Try existing players
            for player_id, player_state in self.player_states.items():
                if player_id in used_players:
                    continue

                frames_gap = frame_idx - player_state.last_frame
                if frames_gap <= self.occlusion_max_frames:
                    cost = self._calculate_assignment_cost(detection, player_state, frame_idx)
                    adaptive_threshold = self.max_distance_threshold * (1.0 + frames_gap * 0.1)

                    if cost < best_cost and cost <= adaptive_threshold:
                        best_cost = cost
                        best_player = player_id

            if best_player:
                assignments[det_idx] = best_player
                used_players.add(best_player)
            else:
                # Create new player
                new_id = f"player_{self.next_player_id}"
                self.next_player_id += 1
                assignments[det_idx] = new_id

        return assignments

    def _update_player_state(self, player_id: str, detection: Dict, frame_idx: int):
        """Update player state with enhanced tracking information."""
        center_pos = detection['center_position']
        position = (center_pos['x'], center_pos['y'])
        confidence = self._calculate_position_confidence(detection['ankles'])

        if player_id not in self.player_states:
            # Create new player state
            self.player_states[player_id] = PlayerState(
                player_id=player_id,
                last_position=position,
                last_frame=frame_idx,
                velocity=(0.0, 0.0),
                confidence_history=deque([confidence], maxlen=self.confidence_history_frames),
                trajectory=deque([(position[0], position[1], frame_idx)], maxlen=self.trajectory_history_frames),
                occlusion_count=0,
                tracking_quality=confidence
            )
        else:
            player_state = self.player_states[player_id]

            # Update trajectory
            player_state.trajectory.append((position[0], position[1], frame_idx))

            # Calculate new velocity
            trajectory_list = list(player_state.trajectory)
            player_state.velocity = self._calculate_velocity(trajectory_list)

            # Update confidence history
            player_state.confidence_history.append(confidence)

            # Update tracking quality
            player_state.tracking_quality = self._calculate_tracking_quality(player_state)

            # Reset occlusion count (player is visible)
            player_state.occlusion_count = 0

            # Update position and frame
            player_state.last_position = position
            player_state.last_frame = frame_idx

    def _handle_occlusions(self, frame_idx: int):
        """Handle players that are currently occluded."""
        for player_id, player_state in self.player_states.items():
            frames_gap = frame_idx - player_state.last_frame

            if 0 < frames_gap <= self.occlusion_max_frames:
                # Player is occluded, increment occlusion counter
                player_state.occlusion_count = frames_gap

                # Update predicted position
                player_state.predicted_position = self._predict_position(player_state, frame_idx)

                # Reduce tracking quality during occlusion
                player_state.tracking_quality *= 0.9

    def process_frame_detections(self, frame_ankle_data: Dict[int, List[Dict]], frame_idx: int) -> Dict[str, Dict[str, Any]]:
        """Process frame detections with enhanced occlusion handling."""
        if not frame_ankle_data:
            self._handle_occlusions(frame_idx)
            return {}

        # Convert to detection format
        detections = []
        for person_id, ankle_detections in frame_ankle_data.items():
            if not ankle_detections:
                continue

            # Calculate center position
            avg_x = sum(ankle['world_x'] for ankle in ankle_detections) / len(ankle_detections)
            avg_y = sum(ankle['world_y'] for ankle in ankle_detections) / len(ankle_detections)

            detections.append({
                'person_id': person_id,
                'ankles': ankle_detections,
                'center_position': {'x': float(avg_x), 'y': float(avg_y)}
            })

        if self.debug:
            print(f"Frame {frame_idx}: Processing {len(detections)} detections, "
                  f"{len(self.player_states)} tracked players")

        # Handle occlusions first
        self._handle_occlusions(frame_idx)

        # Assign player IDs
        assignments = self._assign_detections_to_players(detections, frame_idx)

        # Build result and update states
        frame_players = {}
        for det_idx, player_id in assignments.items():
            detection = detections[det_idx]
            frame_players[player_id] = {
                'ankles': detection['ankles'],
                'center_position': detection['center_position']
            }

            # Update player state
            self._update_player_state(player_id, detection, frame_idx)

        return frame_players

    def get_final_player_mapping(self) -> Dict[str, str]:
        """Get mapping from internal player IDs to standard player_0, player_1 format."""
        if not self.player_states:
            return {}

        # Find most active and highest quality players
        player_scores = {}
        for player_id, player_state in self.player_states.items():
            # Score based on activity, quality, and recency
            activity_score = len(player_state.trajectory)
            quality_score = player_state.tracking_quality * 100
            recency_score = max(0, 100 - player_state.occlusion_count * 10)

            total_score = activity_score + quality_score + recency_score
            player_scores[player_id] = total_score

        # Sort by score and take top 2
        sorted_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)

        mapping = {}
        if len(sorted_players) >= 1:
            mapping[sorted_players[0][0]] = "player_0"
        if len(sorted_players) >= 2:
            mapping[sorted_players[1][0]] = "player_1"

        return mapping

    def print_tracking_stats(self):
        """Print enhanced tracking statistics."""
        if not self.player_states:
            return

        print(f"\n=== Advanced Player Tracking Statistics ===")
        print(f"Total players tracked: {len(self.player_states)}")

        for player_id, player_state in self.player_states.items():
            if not player_state.trajectory:
                continue

            frames_tracked = len(player_state.trajectory)
            first_frame = player_state.trajectory[0][2]
            last_frame = player_state.trajectory[-1][2]
            frame_span = last_frame - first_frame + 1
            coverage = frames_tracked / frame_span if frame_span > 0 else 0

            avg_confidence = (sum(player_state.confidence_history) / len(player_state.confidence_history)
                              if player_state.confidence_history else 0)

            print(f"{player_id}: {frames_tracked} frames ({coverage:.1%} coverage), "
                  f"quality={player_state.tracking_quality:.2f}, "
                  f"avg_conf={avg_confidence:.2f}, "
                  f"max_occlusion={player_state.occlusion_count}")

        print("==========================================\n")


class EnhancedAnkleTracker:
    """Enhanced tracker with advanced robust player ID system."""

    # Court dimensions (meters) - BWF standard
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Pose joint indices (COCO format)
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    BASE_ANKLE_OFFSET = 0.04  # Base 4cm offset from ankle to ground

    def __init__(self, video_path: str, debug: bool = False):
        """Initialize enhanced ankle tracker."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.calibration_file = self.results_dir / "calibration.csv"
        self.output_file = self.results_dir / "positions.json"
        self.debug = debug

        # Data containers
        self.pose_data = None
        self.court_points = None
        self.video_info = None
        self.homography_matrix = None

        # Calibrated camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_height = None
        self.reprojection_error = None
        self.calibration_available = False
        self.enhanced_ankle_offset = None

        # Advanced robust player tracking
        self.player_tracker = AdvancedPlayerTracker(
            court_width=self.COURT_WIDTH,
            court_length=self.COURT_LENGTH,
            debug=debug
        )

        # Frame data storage (internal tracking format)
        self.frame_data_internal = {}

    def load_calibration_data(self) -> None:
        """Load calibration data for homography enhancement."""
        if not self.calibration_file.exists():
            print("⚠️  No calibration data found - using basic homography")
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
                        elif key == 'reprojection_error_px':
                            calibration_params['reprojection_error_px'] = float(value)
                    except (ValueError, IndexError):
                        continue

            # Reconstruct camera matrix
            if all(param in calibration_params for param in ['fx', 'fy', 'cx', 'cy']):
                self.camera_matrix = np.array([
                    [calibration_params['fx'], 0, calibration_params['cx']],
                    [0, calibration_params['fy'], calibration_params['cy']],
                    [0, 0, 1]
                ], dtype=np.float32)

            # Reconstruct distortion coefficients
            dist_coeffs = []
            for param in ['k1', 'k2', 'p1', 'p2', 'k3']:
                if param in calibration_params:
                    dist_coeffs.append(calibration_params[param])
            if dist_coeffs:
                self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

            # Get camera height and reprojection error
            self.camera_height = calibration_params.get('camera_height_m')
            self.reprojection_error = calibration_params.get('reprojection_error_px')

            # Check if calibration is good enough
            self.calibration_available = (
                    self.camera_matrix is not None and
                    self.camera_height is not None and
                    (self.reprojection_error is None or self.reprojection_error < 30)
            )

            if self.calibration_available:
                print(f"✓ Calibration available for enhancement (error: {self.reprojection_error:.1f}px)")
                # Calculate enhanced ankle offset
                focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2
                pixel_to_meter_ratio = focal_length / self.camera_height
                self.enhanced_ankle_offset = self.BASE_ANKLE_OFFSET * pixel_to_meter_ratio
            else:
                print("⚠️  Calibration quality insufficient for enhancement")

        except Exception as e:
            print(f"⚠️  Error loading calibration data: {e}")
            self.calibration_available = False

    def load_pose_data(self) -> None:
        """Load pose detection data."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.video_info = data.get('video_info', {})
        self.court_points = data.get('court_points', {}) or data.get('all_court_points', {})

        # Set frame rate for motion calculations
        fps = self.video_info.get('fps', 30.0)
        self.player_tracker.set_frame_rate(fps)

        if not self.court_points:
            raise ValueError("No court points found in pose data")

        print(f"✓ Loaded pose data with {len(data.get('pose_data', []))} detections")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        required_corners = ['P1', 'P2', 'P3', 'P4']

        # Check for missing corners
        missing_corners = [corner for corner in required_corners if corner not in self.court_points]
        if missing_corners:
            raise ValueError(f"Missing required court corners: {missing_corners}")

        # Extract corner coordinates
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

        # World coordinates for standard badminton court
        world_points = np.array([
            [0, 0],                                    # P1: Top-left
            [0, self.COURT_LENGTH],                    # P2: Bottom-left
            [self.COURT_WIDTH, self.COURT_LENGTH],     # P3: Bottom-right
            [self.COURT_WIDTH, 0]                      # P4: Top-right
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

        print("✓ Homography matrix calculated")

    def undistort_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Undistort a pixel point if calibration available."""
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
        """Calculate ankle ground position using enhanced homography."""
        try:
            # Undistort if calibration available
            if self.calibration_available:
                undistorted_pixel = self.undistort_point(ankle_pixel)
            else:
                undistorted_pixel = ankle_pixel

            # Apply ankle-to-ground offset
            if self.enhanced_ankle_offset is not None:
                offset_y = self.enhanced_ankle_offset
            else:
                offset_y = 12.0  # pixels - basic fallback

            corrected_pixel = (undistorted_pixel[0], undistorted_pixel[1] + offset_y)

            # Transform to world coordinates
            point = np.array([[corrected_pixel]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)

            world_x = float(world_point[0][0][0])
            world_y = float(world_point[0][0][1])

            # Boundary validation with tolerance
            world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, world_x))
            world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, world_y))

            return world_x, world_y

        except Exception as e:
            if self.debug:
                print(f"Position calculation failed: {e}")
            return 0.0, 0.0

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence is sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return float(joint['x']), float(joint['y'])
        return None

    def process_person_ankles(self, joints: List[Dict]) -> List[Dict[str, Any]]:
        """Process individual ankle positions for a person."""
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
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
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
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
            })

        return ankle_detections

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> None:
        """Process all people in a single frame."""
        frame_ankle_data = {}

        # Process each person in the frame
        for person_id, human_data in enumerate(frame_data):
            joints = human_data.get('joints', [])
            person_ankles = self.process_person_ankles(joints)
            if person_ankles:
                frame_ankle_data[person_id] = person_ankles

        # Use advanced robust player tracking to assign IDs
        if frame_ankle_data:
            player_assignments = self.player_tracker.process_frame_detections(frame_ankle_data, frame_index)
            if player_assignments:
                self.frame_data_internal[frame_index] = player_assignments

    def process_all_frames(self) -> None:
        """Process all frames with advanced tracking."""
        pose_data = self.pose_data.get('pose_data', [])

        if not pose_data:
            print("⚠️  No pose data found in pose.json")
            return

        # Group by frame
        frames_data = {}
        for entry in pose_data:
            frame_idx = entry['frame_index']
            if frame_idx not in frames_data:
                frames_data[frame_idx] = []
            frames_data[frame_idx].append(entry)

        print(f"Processing {len(frames_data)} frames with advanced tracking...")

        # Process each frame
        for frame_idx in sorted(frames_data.keys()):
            frame_data = frames_data[frame_idx]
            self.process_frame(frame_data, frame_idx)

        print(f"Processed {len(self.frame_data_internal)} frames with ankle detections")

        # Print advanced tracking statistics
        self.player_tracker.print_tracking_stats()

    def convert_to_standard_format(self) -> Dict[str, Dict]:
        """Convert internal tracking format to standard player_0, player_1 format."""
        # Get player mapping
        player_mapping = self.player_tracker.get_final_player_mapping()

        if self.debug:
            print(f"Player mapping: {player_mapping}")

        # Convert to standard format
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

    def validate_results(self) -> None:
        """Validate advanced tracking results."""
        if not self.frame_data_internal:
            print("⚠️  No frame data to validate")
            return

        # Enhanced validation with tracking quality metrics
        sample_positions = []
        player_statistics = defaultdict(list)

        for frame_idx, frame_data in self.frame_data_internal.items():
            for player_id, player_data in frame_data.items():
                for ankle in player_data['ankles']:
                    sample_positions.append(ankle)
                    player_statistics[player_id].append({
                        'frame': frame_idx,
                        'position': (ankle['world_x'], ankle['world_y']),
                        'confidence': ankle['joint_confidence']
                    })

        if len(sample_positions) < 10:
            print("⚠️  Too few positions for validation")
            return

        x_positions = [pos['world_x'] for pos in sample_positions]
        y_positions = [pos['world_y'] for pos in sample_positions]

        out_of_bounds = sum(1 for pos in sample_positions
                            if pos['world_x'] < -0.5 or pos['world_x'] > self.COURT_WIDTH + 0.5 or
                            pos['world_y'] < -0.5 or pos['world_y'] > self.COURT_LENGTH + 0.5)

        # Calculate tracking continuity
        continuity_scores = {}
        for player_id, stats in player_statistics.items():
            if len(stats) < 2:
                continue

            frames = [s['frame'] for s in stats]
            frame_gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            avg_gap = sum(frame_gaps) / len(frame_gaps) if frame_gaps else 1.0
            continuity_scores[player_id] = 1.0 / avg_gap if avg_gap > 0 else 1.0

        print(f"=== Advanced Ankle Tracking Quality ===")
        print(f"Positions analyzed: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds}/{len(sample_positions)} ({out_of_bounds/len(sample_positions):.1%})")

        for player_id, score in continuity_scores.items():
            print(f"Player {player_id} continuity score: {score:.2f}")

        # Overall quality assessment
        spatial_quality = 1.0 - (out_of_bounds / len(sample_positions))
        avg_continuity = sum(continuity_scores.values()) / len(continuity_scores) if continuity_scores else 0.5
        overall_quality = (spatial_quality + avg_continuity) / 2

        print(f"Overall tracking quality: {overall_quality:.2f}")

        if overall_quality > 0.8:
            print("✅ Advanced tracking quality excellent")
        elif overall_quality > 0.6:
            print("✅ Advanced tracking quality good")
        else:
            print("⚠️  Tracking quality needs improvement - check calibration and parameters")
        print("=====================================\n")

    def save_results(self) -> None:
        """Save results with advanced tracking metadata."""
        # Convert to standard format
        frame_data_dict = self.convert_to_standard_format()

        # Calculate enhanced summary statistics
        total_frames_with_data = len(frame_data_dict)
        total_ankle_detections = 0
        player_0_detections = 0
        player_1_detections = 0
        left_ankle_detections = 0
        right_ankle_detections = 0

        for frame_data in frame_data_dict.values():
            if 'player_0' in frame_data:
                player_0_detections += len(frame_data['player_0']['ankles'])
            if 'player_1' in frame_data:
                player_1_detections += len(frame_data['player_1']['ankles'])

            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    total_ankle_detections += 1
                    if ankle['ankle_side'] == 'left':
                        left_ankle_detections += 1
                    else:
                        right_ankle_detections += 1

        # Get tracking quality metrics
        tracking_quality_metrics = {}
        player_mapping = self.player_tracker.get_final_player_mapping()
        for internal_id, standard_id in player_mapping.items():
            if internal_id in self.player_tracker.player_states:
                player_state = self.player_tracker.player_states[internal_id]
                tracking_quality_metrics[standard_id] = {
                    'tracking_quality': float(player_state.tracking_quality),
                    'max_occlusion_frames': int(player_state.occlusion_count),
                    'trajectory_length': len(player_state.trajectory),
                    'avg_confidence': float(sum(player_state.confidence_history) / len(player_state.confidence_history))
                    if player_state.confidence_history else 0.0
                }

        # Create enhanced output data structure
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
                'left_ankle_detections': left_ankle_detections,
                'right_ankle_detections': right_ankle_detections,
                'primary_method': 'enhanced_homography' if self.calibration_available else 'basic_homography',
                'ankle_ground_offset_meters': float(self.BASE_ANKLE_OFFSET),
                'calibration_enhanced': self.calibration_available,
                'advanced_tracking_enabled': True
            },
            'advanced_tracking_info': {
                'camera_height_meters': float(self.camera_height) if self.calibration_available and self.camera_height else None,
                'enhanced_ankle_offset_pixels': float(self.enhanced_ankle_offset) if self.calibration_available and self.enhanced_ankle_offset else None,
                'reprojection_error_px': float(self.reprojection_error) if self.calibration_available and self.reprojection_error else None,
                'tracking_algorithm': 'Advanced Hungarian Assignment with Motion Prediction',
                'occlusion_handling': 'Multi-frame prediction with adaptive thresholds',
                'max_distance_threshold_m': self.player_tracker.max_distance_threshold,
                'occlusion_max_frames': self.player_tracker.occlusion_max_frames,
                'trajectory_history_frames': self.player_tracker.trajectory_history_frames,
                'velocity_prediction_enabled': True,
                'adaptive_thresholds_enabled': True,
                'quality_metrics': tracking_quality_metrics
            },
            'tracking_parameters': {
                'velocity_weight': self.player_tracker.velocity_weight,
                'position_weight': self.player_tracker.position_weight,
                'confidence_weight': self.player_tracker.confidence_weight,
                'temporal_weight': self.player_tracker.temporal_weight,
                'max_velocity_ms': self.player_tracker.max_velocity,
                'acceleration_limit_ms2': self.player_tracker.acceleration_limit,
                'prediction_frames': self.player_tracker.prediction_frames
            },
            'frame_data': frame_data_dict
        }

        # Convert numpy types
        output_data_clean = convert_numpy_types(output_data)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data_clean, f, indent=2)

        print(f"✅ Results saved to: {self.output_file}")
        print(f"📊 Frames with data: {total_frames_with_data}")
        print(f"📊 Player 0: {player_0_detections} ankle detections")
        print(f"📊 Player 1: {player_1_detections} ankle detections")
        print(f"📊 Left ankles: {left_ankle_detections}, Right ankles: {right_ankle_detections}")
        print(f"📊 Advanced tracking: Enabled with occlusion handling")
        if tracking_quality_metrics:
            for player_id, metrics in tracking_quality_metrics.items():
                print(f"📊 {player_id}: quality={metrics['tracking_quality']:.2f}, "
                      f"max_occlusion={metrics['max_occlusion_frames']} frames")

    def run(self) -> None:
        """Run the advanced ankle tracking pipeline."""
        print(f"🚀 Starting advanced ankle tracking with robust occlusion handling: {self.video_name}")
        print("="*80)

        try:
            print("📊 Step 1: Loading calibration data...")
            self.load_calibration_data()

            print("📍 Step 2: Loading pose data...")
            self.load_pose_data()

            print("🔧 Step 3: Calculating homography...")
            self.calculate_homography()

            print("🏃 Step 4: Processing all frames with advanced tracking...")
            self.process_all_frames()

            print("✅ Step 5: Validating advanced tracking results...")
            self.validate_results()

            print("💾 Step 6: Saving results...")
            self.save_results()

            print("="*80)
            print("✅ Advanced ankle tracking with robust occlusion handling completed!")
            print("✅ Output is compatible with stage 3 visualization")

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_location.py <video_file_path> [--debug]")
        print("\nExample:")
        print("  python calculate_location.py samples/badminton_match.mp4")
        print("\nRequirements:")
        print("  - Court detection must be run first (detect_court.py)")
        print("  - Pose estimation must be run (detect_pose.py)")
        print("  - OpenCV, NumPy, SciPy")
        print("\nPipeline:")
        print("  1. python detect_court.py <video_path>")
        print("  2. python detect_pose.py <video_path>")
        print("  3. python calculate_location.py <video_path>")
        print("  4. python visualize.py <video_path> --stage 3")
        print("\nAdvanced Features:")
        print("  - Multi-frame temporal consistency for robust player identification")
        print("  - Advanced Hungarian algorithm with motion prediction")
        print("  - Robust occlusion handling with adaptive thresholds")
        print("  - Velocity-based trajectory prediction and validation")
        print("  - Enhanced homography with camera calibration support")
        print("  - Quality-based player assignment with confidence tracking")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    # Check for required dependencies
    try:
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("❌ Error: SciPy is required for advanced player tracking")
        print("Install with: pip install scipy")
        sys.exit(1)

    tracker = EnhancedAnkleTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()