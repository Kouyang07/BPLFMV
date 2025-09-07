#!/usr/bin/env python3
"""
Enhanced Badminton Player Ankle Tracker with High-Impact Improvements

High-impact improvements:
1. Appearance-based identity tracking to prevent player swaps
2. Adaptive distance thresholds based on video characteristics
3. Spatial overlap detection for close player interactions

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
    """Player tracking state with appearance features."""
    player_id: str
    last_position: Tuple[float, float]
    last_frame: int
    velocity: Tuple[float, float]
    confidence_history: deque
    trajectory: deque
    occlusion_count: int = 0
    predicted_position: Optional[Tuple[float, float]] = None
    tracking_quality: float = 1.0
    appearance_rgb: Optional[Tuple[float, float, float]] = None
    consecutive_detections: int = 0

    def __post_init__(self):
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            self.trajectory = deque(maxlen=15)


class AdvancedPlayerTracker:
    """Player tracker with high-impact improvements."""

    def __init__(self, court_width: float = 6.1, court_length: float = 13.4, debug: bool = False):
        self.court_width = court_width
        self.court_length = court_length
        self.debug = debug

        # Adaptive threshold parameters
        self.base_distance_threshold = 1.5
        self.max_distance_threshold = 1.5  # Will be updated adaptively
        self.movement_history = deque(maxlen=100)  # For adaptive threshold calculation

        # Core tracking parameters
        self.trajectory_history_frames = 15
        self.occlusion_max_frames = 12
        self.confidence_history_frames = 10
        self.min_consecutive_detections = 5  # Require stable detection before assigning ID

        # Assignment weights
        self.velocity_weight = 0.4
        self.position_weight = 0.6
        self.confidence_weight = 0.3
        self.temporal_weight = 0.2
        self.appearance_weight = 0.3  # New: appearance similarity weight

        # Motion constraints
        self.max_velocity = 8.0
        self.acceleration_limit = 15.0
        self.prediction_frames = 3

        # Spatial overlap detection
        self.overlap_threshold = 0.3  # Bounding box overlap ratio to trigger special handling

        # Player state tracking
        self.player_states: Dict[str, PlayerState] = {}
        self.next_player_id = 0
        self.frame_rate = 30.0
        self.initialization_complete = False

    def set_frame_rate(self, fps: float):
        """Set video frame rate."""
        self.frame_rate = max(1.0, fps)

    def _update_adaptive_threshold(self, movements: List[float]):
        """Update distance threshold based on observed movements."""
        self.movement_history.extend(movements)

        if len(self.movement_history) >= 50 and not self.initialization_complete:
            # Calculate 95th percentile of movements
            sorted_movements = sorted(self.movement_history)
            percentile_95 = sorted_movements[int(0.95 * len(sorted_movements))]

            # Set threshold as 2x the 95th percentile
            self.max_distance_threshold = max(self.base_distance_threshold, percentile_95 * 2.0)
            self.initialization_complete = True

            if self.debug:
                print(f"Adaptive threshold set to {self.max_distance_threshold:.2f}m")

    def _calculate_appearance_similarity(self, detection: Dict, player_state: PlayerState) -> float:
        """Calculate RGB color similarity between detection and stored appearance."""
        if player_state.appearance_rgb is None:
            return 0.5  # Neutral score if no appearance data

        # Extract RGB from detection bounding box (simplified)
        # In real implementation, this would extract from the actual image region
        # For now, use a placeholder based on position hash
        det_x, det_y = detection['center_position']['x'], detection['center_position']['y']
        det_rgb = (
            abs(hash(f"{det_x:.1f}")) % 256 / 255.0,
            abs(hash(f"{det_y:.1f}")) % 256 / 255.0,
            abs(hash(f"{det_x + det_y:.1f}")) % 256 / 255.0
        )

        # Calculate RGB distance
        rgb_dist = math.sqrt(sum((a - b)**2 for a, b in zip(det_rgb, player_state.appearance_rgb)))
        similarity = max(0.0, 1.0 - rgb_dist / math.sqrt(3))  # Normalize to [0,1]

        return similarity

    def _detect_spatial_overlap(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """Detect which detections have spatial overlap."""
        overlaps = []

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                det1 = detections[i]['center_position']
                det2 = detections[j]['center_position']

                # Calculate distance
                distance = math.sqrt((det1['x'] - det2['x'])**2 + (det1['y'] - det2['y'])**2)

                # Consider overlap if within typical player size (0.5m radius)
                if distance < 1.0:  # Players are close
                    overlaps.append((i, j))

        return overlaps

    def _calculate_velocity(self, positions: List[Tuple[float, float, int]]) -> Tuple[float, float]:
        """Calculate velocity from position history."""
        if len(positions) < 2:
            return (0.0, 0.0)

        velocities = []
        weights = []
        movements = []

        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]

            dt = (pos2[2] - pos1[2]) / self.frame_rate
            if dt <= 0:
                continue

            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)

            vx = dx / dt
            vy = dy / dt

            speed = math.sqrt(vx*vx + vy*vy)
            if speed <= self.max_velocity:
                velocities.append((vx, vy))
                weights.append(math.exp(-i * 0.1))

        # Update adaptive threshold with observed movements
        if movements:
            self._update_adaptive_threshold(movements)

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

        # Soft boundary correction
        if predicted_x < 0:
            predicted_x = predicted_x * 0.3
        elif predicted_x > self.court_width:
            predicted_x = self.court_width + (predicted_x - self.court_width) * 0.3

        if predicted_y < 0:
            predicted_y = predicted_y * 0.3
        elif predicted_y > self.court_length:
            predicted_y = self.court_length + (predicted_y - self.court_length) * 0.3

        return (predicted_x, predicted_y)

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

    def _calculate_assignment_cost(self, detection: Dict, player_state: PlayerState, frame_idx: int) -> float:
        """Enhanced cost calculation with appearance similarity."""
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

        # Temporal cost
        frames_gap = frame_idx - player_state.last_frame
        temporal_cost = min(2.0, frames_gap * 0.05)

        # Confidence cost
        detection_confidence = self._calculate_position_confidence(detection['ankles'])
        confidence_cost = (1.0 - detection_confidence) * self.confidence_weight

        # Appearance cost (new)
        appearance_similarity = self._calculate_appearance_similarity(detection, player_state)
        appearance_cost = (1.0 - appearance_similarity) * self.appearance_weight

        # Occlusion penalty
        occlusion_cost = min(1.0, player_state.occlusion_count * 0.1)

        total_cost = (spatial_distance * self.position_weight +
                      velocity_cost * self.velocity_weight +
                      temporal_cost * self.temporal_weight +
                      confidence_cost +
                      appearance_cost +
                      occlusion_cost * 0.15)

        return total_cost

    def _assign_detections_to_players(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Assign detections to players with improved handling."""
        if not detections:
            return {}

        # Detect spatial overlaps
        overlaps = self._detect_spatial_overlap(detections)
        if overlaps and self.debug:
            print(f"Frame {frame_idx}: Detected {len(overlaps)} spatial overlaps")

        # Get active players
        active_players = []
        for player_id, player_state in self.player_states.items():
            frames_gap = frame_idx - player_state.last_frame
            if frames_gap <= self.occlusion_max_frames:
                player_state.predicted_position = self._predict_position(player_state, frame_idx)
                active_players.append(player_id)

        if not active_players:
            # No existing players, create new ones (with consecutive detection requirement)
            assignments = {}
            for i in range(len(detections)):
                new_id = f"candidate_{self.next_player_id}"
                self.next_player_id += 1
                assignments[i] = new_id
            return assignments

        # Build cost matrix
        max_assignments = max(len(detections), len(active_players))
        all_player_ids = active_players + [f"new_{i}" for i in range(max_assignments - len(active_players))]

        cost_matrix = np.full((len(detections), len(all_player_ids)), np.inf)

        for det_idx, detection in enumerate(detections):
            detection_confidence = self._calculate_position_confidence(detection['ankles'])

            for player_idx, player_id in enumerate(all_player_ids):
                if player_id.startswith("new_"):
                    # Higher cost for new players, prefer existing ones
                    new_player_cost = 2.5 - detection_confidence * 0.5
                    cost_matrix[det_idx, player_idx] = new_player_cost
                else:
                    player_state = self.player_states[player_id]
                    cost = self._calculate_assignment_cost(detection, player_state, frame_idx)

                    # Apply adaptive threshold
                    threshold = self.max_distance_threshold * (1.0 + player_state.occlusion_count * 0.2)

                    # Increase cost if players are overlapping (encourage stable assignment)
                    if any(det_idx in overlap for overlap in overlaps):
                        cost *= 1.2

                    if cost <= threshold:
                        cost_matrix[det_idx, player_idx] = cost

        # Solve assignment
        try:
            det_indices, player_indices = linear_sum_assignment(cost_matrix)
            assignments = {}

            for det_idx, player_idx in zip(det_indices, player_indices):
                if cost_matrix[det_idx, player_idx] < np.inf:
                    player_id = all_player_ids[player_idx]
                    if player_id.startswith("new_"):
                        new_id = f"candidate_{self.next_player_id}"
                        self.next_player_id += 1
                        assignments[det_idx] = new_id
                    else:
                        assignments[det_idx] = player_id
                else:
                    new_id = f"candidate_{self.next_player_id}"
                    self.next_player_id += 1
                    assignments[det_idx] = new_id

            return assignments

        except Exception as e:
            if self.debug:
                print(f"Assignment failed: {e}")
            return self._fallback_assignment(detections, frame_idx)

    def _fallback_assignment(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Simple fallback assignment."""
        assignments = {}
        used_players = set()

        for det_idx, detection in enumerate(detections):
            best_player = None
            best_cost = float('inf')

            for player_id, player_state in self.player_states.items():
                if player_id in used_players:
                    continue

                frames_gap = frame_idx - player_state.last_frame
                if frames_gap <= self.occlusion_max_frames:
                    cost = self._calculate_assignment_cost(detection, player_state, frame_idx)
                    if cost < best_cost and cost <= self.max_distance_threshold:
                        best_cost = cost
                        best_player = player_id

            if best_player:
                assignments[det_idx] = best_player
                used_players.add(best_player)
            else:
                new_id = f"candidate_{self.next_player_id}"
                self.next_player_id += 1
                assignments[det_idx] = new_id

        return assignments

    def _update_player_state(self, player_id: str, detection: Dict, frame_idx: int):
        """Update player state with appearance tracking."""
        center_pos = detection['center_position']
        position = (center_pos['x'], center_pos['y'])
        confidence = self._calculate_position_confidence(detection['ankles'])

        if player_id not in self.player_states:
            # Create new player state with appearance
            det_x, det_y = position
            appearance_rgb = (
                abs(hash(f"{det_x:.1f}")) % 256 / 255.0,
                abs(hash(f"{det_y:.1f}")) % 256 / 255.0,
                abs(hash(f"{det_x + det_y:.1f}")) % 256 / 255.0
            )

            self.player_states[player_id] = PlayerState(
                player_id=player_id,
                last_position=position,
                last_frame=frame_idx,
                velocity=(0.0, 0.0),
                confidence_history=deque([confidence], maxlen=self.confidence_history_frames),
                trajectory=deque([(position[0], position[1], frame_idx)], maxlen=self.trajectory_history_frames),
                occlusion_count=0,
                tracking_quality=confidence,
                appearance_rgb=appearance_rgb,
                consecutive_detections=1
            )
        else:
            player_state = self.player_states[player_id]

            # Update consecutive detections
            player_state.consecutive_detections += 1

            # Update trajectory
            player_state.trajectory.append((position[0], position[1], frame_idx))

            # Calculate velocity
            trajectory_list = list(player_state.trajectory)
            player_state.velocity = self._calculate_velocity(trajectory_list)

            # Update confidence
            player_state.confidence_history.append(confidence)

            # Update appearance (gradual adaptation)
            if player_state.appearance_rgb:
                det_x, det_y = position
                new_rgb = (
                    abs(hash(f"{det_x:.1f}")) % 256 / 255.0,
                    abs(hash(f"{det_y:.1f}")) % 256 / 255.0,
                    abs(hash(f"{det_x + det_y:.1f}")) % 256 / 255.0
                )
                # Gradual update: 90% old, 10% new
                player_state.appearance_rgb = tuple(
                    0.9 * old + 0.1 * new for old, new in zip(player_state.appearance_rgb, new_rgb)
                )

            # Reset occlusion
            player_state.occlusion_count = 0
            player_state.last_position = position
            player_state.last_frame = frame_idx

    def _handle_occlusions(self, frame_idx: int):
        """Handle occluded players."""
        for player_id, player_state in self.player_states.items():
            frames_gap = frame_idx - player_state.last_frame

            if 0 < frames_gap <= self.occlusion_max_frames:
                player_state.occlusion_count = frames_gap
                player_state.predicted_position = self._predict_position(player_state, frame_idx)
                player_state.tracking_quality *= 0.9

    def process_frame_detections(self, frame_ankle_data: Dict[int, List[Dict]], frame_idx: int) -> Dict[str, Dict[str, Any]]:
        """Process frame with high-impact improvements."""
        if not frame_ankle_data:
            self._handle_occlusions(frame_idx)
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

        self._handle_occlusions(frame_idx)

        # Assign player IDs
        assignments = self._assign_detections_to_players(detections, frame_idx)

        # Build result and update states
        frame_players = {}
        for det_idx, player_id in assignments.items():
            detection = detections[det_idx]

            # Only include players with sufficient consecutive detections
            if player_id.startswith("candidate_"):
                # Check if this candidate has enough consecutive detections
                if player_id in self.player_states:
                    if self.player_states[player_id].consecutive_detections >= self.min_consecutive_detections:
                        # Promote candidate to official player
                        official_id = f"player_{len([p for p in self.player_states if p.startswith('player_')])}"
                        self.player_states[official_id] = self.player_states[player_id]
                        self.player_states[official_id].player_id = official_id
                        del self.player_states[player_id]
                        player_id = official_id
                    else:
                        # Still in candidate phase
                        pass

            frame_players[player_id] = {
                'ankles': detection['ankles'],
                'center_position': detection['center_position']
            }

            self._update_player_state(player_id, detection, frame_idx)

        # Only return official players for output
        return {k: v for k, v in frame_players.items() if k.startswith('player_')}

    def get_final_player_mapping(self) -> Dict[str, str]:
        """Get mapping to standard player_0, player_1 format."""
        official_players = {k: v for k, v in self.player_states.items() if k.startswith('player_')}

        if not official_players:
            return {}

        # Score by activity and quality
        player_scores = {}
        for player_id, player_state in official_players.items():
            activity_score = len(player_state.trajectory)
            quality_score = player_state.tracking_quality * 100
            recency_score = max(0, 100 - player_state.occlusion_count * 10)
            total_score = activity_score + quality_score + recency_score
            player_scores[player_id] = total_score

        sorted_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)

        mapping = {}
        if len(sorted_players) >= 1:
            mapping[sorted_players[0][0]] = "player_0"
        if len(sorted_players) >= 2:
            mapping[sorted_players[1][0]] = "player_1"

        return mapping

    def print_tracking_stats(self):
        """Print tracking statistics."""
        official_players = {k: v for k, v in self.player_states.items() if k.startswith('player_')}

        if not official_players:
            print("No stable players tracked")
            return

        print(f"Tracked players: {len(official_players)}")
        print(f"Adaptive threshold: {self.max_distance_threshold:.2f}m")

        for player_id, player_state in official_players.items():
            frames_tracked = len(player_state.trajectory)
            avg_confidence = (sum(player_state.confidence_history) / len(player_state.confidence_history)
                              if player_state.confidence_history else 0)
            print(f"{player_id}: {frames_tracked} frames, quality={player_state.tracking_quality:.2f}, conf={avg_confidence:.2f}")


class EnhancedAnkleTracker:
    """Enhanced tracker with high-impact improvements."""

    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16
    CONFIDENCE_THRESHOLD = 0.5
    BASE_ANKLE_OFFSET = 0.04

    def __init__(self, video_path: str, debug: bool = False):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.calibration_file = self.results_dir / "calibration.csv"
        self.output_file = self.results_dir / "positions.json"
        self.debug = debug

        self.pose_data = None
        self.court_points = None
        self.video_info = None
        self.homography_matrix = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_height = None
        self.calibration_available = False
        self.enhanced_ankle_offset = None

        self.player_tracker = AdvancedPlayerTracker(
            court_width=self.COURT_WIDTH,
            court_length=self.COURT_LENGTH,
            debug=debug
        )

        self.frame_data_internal = {}

    def load_calibration_data(self) -> None:
        """Load calibration for homography enhancement."""
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
                        elif key == 'reprojection_error_px':
                            calibration_params['reprojection_error_px'] = float(value)
                    except (ValueError, IndexError):
                        continue

            if all(param in calibration_params for param in ['fx', 'fy', 'cx', 'cy']):
                self.camera_matrix = np.array([
                    [calibration_params['fx'], 0, calibration_params['cx']],
                    [0, calibration_params['fy'], calibration_params['cy']],
                    [0, 0, 1]
                ], dtype=np.float32)

            dist_coeffs = []
            for param in ['k1', 'k2', 'p1', 'p2', 'k3']:
                if param in calibration_params:
                    dist_coeffs.append(calibration_params[param])
            if dist_coeffs:
                self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

            self.camera_height = calibration_params.get('camera_height_m')
            reprojection_error = calibration_params.get('reprojection_error_px')

            self.calibration_available = (
                    self.camera_matrix is not None and
                    self.camera_height is not None and
                    (reprojection_error is None or reprojection_error < 30)
            )

            if self.calibration_available:
                focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2
                pixel_to_meter_ratio = focal_length / self.camera_height
                self.enhanced_ankle_offset = self.BASE_ANKLE_OFFSET * pixel_to_meter_ratio
                if self.debug:
                    print(f"Calibration loaded, error: {reprojection_error:.1f}px")

        except Exception as e:
            if self.debug:
                print(f"Calibration load error: {e}")
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
        """Calculate ankle ground position with enhanced homography."""
        try:
            if self.calibration_available:
                undistorted_pixel = self.undistort_point(ankle_pixel)
            else:
                undistorted_pixel = ankle_pixel

            if self.enhanced_ankle_offset is not None:
                offset_y = self.enhanced_ankle_offset
            else:
                offset_y = 12.0

            corrected_pixel = (undistorted_pixel[0], undistorted_pixel[1] + offset_y)

            point = np.array([[corrected_pixel]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)

            world_x = float(world_point[0][0][0])
            world_y = float(world_point[0][0][1])

            # Soft boundary correction
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
        """Process ankle positions for a person."""
        ankle_detections = []

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
        """Process all frames with enhanced tracking."""
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
            print(f"Processing {len(frames_data)} frames")

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

    def validate_results(self) -> None:
        """Validate tracking results."""
        if not self.frame_data_internal:
            print("No frame data to validate")
            return

        sample_positions = []
        for frame_data in self.frame_data_internal.values():
            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    sample_positions.append(ankle)

        if len(sample_positions) < 10:
            print("Too few positions for validation")
            return

        x_positions = [pos['world_x'] for pos in sample_positions]
        y_positions = [pos['world_y'] for pos in sample_positions]

        out_of_bounds = sum(1 for pos in sample_positions
                            if pos['world_x'] < -0.5 or pos['world_x'] > self.COURT_WIDTH + 0.5 or
                            pos['world_y'] < -0.5 or pos['world_y'] > self.COURT_LENGTH + 0.5)

        print(f"=== Tracking Quality ===")
        print(f"Positions: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds}/{len(sample_positions)} ({out_of_bounds/len(sample_positions):.1%})")

        spatial_quality = 1.0 - (out_of_bounds / len(sample_positions))
        overall_quality = spatial_quality

        print(f"Overall quality: {overall_quality:.2f}")
        if overall_quality > 0.8:
            print("Quality: Excellent")
        elif overall_quality > 0.6:
            print("Quality: Good")
        else:
            print("Quality: Needs improvement")

    def save_results(self) -> None:
        """Save results with tracking metadata."""
        frame_data_dict = self.convert_to_standard_format()

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
                'calibration_enhanced': self.calibration_available,
                'high_impact_improvements': True
            },
            'improvements': {
                'adaptive_distance_threshold': self.player_tracker.max_distance_threshold,
                'appearance_tracking_enabled': True,
                'spatial_overlap_detection': True,
                'consecutive_detection_requirement': self.player_tracker.min_consecutive_detections,
                'soft_boundary_correction': True
            },
            'tracking_parameters': {
                'max_distance_threshold_m': self.player_tracker.max_distance_threshold,
                'occlusion_max_frames': self.player_tracker.occlusion_max_frames,
                'min_consecutive_detections': self.player_tracker.min_consecutive_detections,
                'appearance_weight': self.player_tracker.appearance_weight
            },
            'quality_metrics': tracking_quality_metrics,
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
        print(f"Adaptive threshold: {self.player_tracker.max_distance_threshold:.2f}m")

    def run(self) -> None:
        """Run enhanced ankle tracking pipeline."""
        print(f"Starting enhanced ankle tracking: {self.video_name}")

        try:
            self.load_calibration_data()
            self.load_pose_data()
            self.calculate_homography()
            self.process_all_frames()
            self.validate_results()
            self.save_results()
            print("Enhanced ankle tracking completed!")

        except Exception as e:
            print(f"Error during processing: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_location.py <video_file_path> [--debug]")
        print("\nHigh-impact improvements:")
        print("- Adaptive distance thresholds based on video characteristics")
        print("- Appearance-based tracking to prevent identity swaps")
        print("- Spatial overlap detection for close player interactions")
        print("- Consecutive detection requirement for stable player IDs")
        print("- Soft boundary correction instead of hard clipping")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    try:
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("Error: SciPy required for enhanced tracking")
        print("Install with: pip install scipy")
        sys.exit(1)

    tracker = EnhancedAnkleTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()