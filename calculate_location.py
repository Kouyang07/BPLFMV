#!/usr/bin/env python3
"""
Enhanced Badminton Player Ankle Tracking Script with Robust Player ID System

Tracks individual ankle positions using enhanced homography approach with robust
player identification that maintains consistency across frames.

Key Features:
- Temporal consistency for player IDs across frames
- Trajectory-based assignment using Hungarian algorithm
- Enhanced homography with calibration improvements
- Individual ankle position tracking (left and right separately)
- Comprehensive validation and error correction

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


class RobustPlayerTracker:
    """Robust player ID tracking with temporal consistency."""

    def __init__(self, court_width: float = 6.1, court_length: float = 13.4, debug: bool = False):
        self.court_width = court_width
        self.court_length = court_length
        self.debug = debug

        # Tracking parameters
        self.max_distance_threshold = 2.0  # meters - max reasonable movement between frames
        self.trajectory_history_frames = 8  # frames to keep in history
        self.occlusion_max_frames = 5  # max frames to maintain ID during occlusion

        # Player state tracking
        self.player_trajectories = {}  # player_id -> deque of (x, y, frame_idx)
        self.player_last_seen = {}     # player_id -> frame_index
        self.next_player_id = 0

    def _calculate_position_confidence(self, ankle_detections: List[Dict]) -> float:
        """Calculate overall position confidence based on ankle detections."""
        if not ankle_detections:
            return 0.0

        # Average joint confidence
        joint_confidences = [ankle['joint_confidence'] for ankle in ankle_detections]
        avg_confidence = sum(joint_confidences) / len(joint_confidences)

        # Adjust based on number of ankles detected
        if len(ankle_detections) == 2:
            # Bonus for having both ankles, check if distance is reasonable
            ankle1, ankle2 = ankle_detections[0], ankle_detections[1]
            distance = np.sqrt((ankle1['world_x'] - ankle2['world_x'])**2 +
                               (ankle1['world_y'] - ankle2['world_y'])**2)
            # Typical distance between ankles is 0.1-0.4m
            if 0.05 <= distance <= 0.6:
                return min(1.0, avg_confidence * 1.1)  # Small bonus
            else:
                return avg_confidence * 0.8  # Small penalty
        else:
            return avg_confidence * 0.9  # Small penalty for single ankle

    def _predict_next_position(self, player_id: str) -> Optional[Tuple[float, float]]:
        """Predict next position based on recent trajectory."""
        if player_id not in self.player_trajectories:
            return None

        trajectory = self.player_trajectories[player_id]
        if len(trajectory) < 2:
            return trajectory[-1][:2] if trajectory else None

        # Simple linear prediction using last two positions
        recent = list(trajectory)[-2:]
        if len(recent) < 2:
            return recent[-1][:2]

        # Calculate velocity (assuming 1 frame difference)
        dx = recent[1][0] - recent[0][0]
        dy = recent[1][1] - recent[0][1]

        # Predict next position
        predicted_x = recent[1][0] + dx
        predicted_y = recent[1][1] + dy

        # Clamp to reasonable bounds
        predicted_x = max(-1.0, min(self.court_width + 1.0, predicted_x))
        predicted_y = max(-1.0, min(self.court_length + 1.0, predicted_y))

        return predicted_x, predicted_y

    def _calculate_assignment_cost(self, detection: Dict, player_id: str, frame_idx: int) -> float:
        """Calculate cost of assigning a detection to a specific player."""
        detection_pos = (detection['center_position']['x'], detection['center_position']['y'])

        if player_id not in self.player_trajectories or not self.player_trajectories[player_id]:
            # New player - moderate cost
            return 1.0

        # Get last known position
        last_pos = self.player_trajectories[player_id][-1][:2]
        spatial_distance = np.sqrt((detection_pos[0] - last_pos[0])**2 +
                                   (detection_pos[1] - last_pos[1])**2)

        # Add prediction cost if we can predict
        prediction_cost = 0.0
        predicted_pos = self._predict_next_position(player_id)
        if predicted_pos:
            prediction_distance = np.sqrt((detection_pos[0] - predicted_pos[0])**2 +
                                          (detection_pos[1] - predicted_pos[1])**2)
            prediction_cost = prediction_distance * 0.3

        # Add temporal cost for long gaps
        frames_since_last_seen = frame_idx - self.player_last_seen.get(player_id, frame_idx)
        temporal_cost = min(1.0, frames_since_last_seen * 0.1)

        # Add confidence cost (lower confidence = higher cost)
        confidence = self._calculate_position_confidence(detection['ankles'])
        confidence_cost = (1.0 - confidence) * 0.3

        total_cost = spatial_distance + prediction_cost + temporal_cost + confidence_cost

        if self.debug:
            print(f"    Cost for player_{player_id}: spatial={spatial_distance:.2f}, "
                  f"prediction={prediction_cost:.2f}, temporal={temporal_cost:.2f}, "
                  f"confidence={confidence_cost:.2f}, total={total_cost:.2f}")

        return total_cost

    def _assign_detections_to_players(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Assign detections to players using Hungarian algorithm."""
        if not detections:
            return {}

        # Get active players (seen recently)
        active_players = []
        for pid, last_frame in self.player_last_seen.items():
            if frame_idx - last_frame <= self.occlusion_max_frames:
                active_players.append(pid)

        # We need at least as many potential assignments as detections
        max_players = max(len(detections), len(active_players))
        all_players = active_players + [f"new_{i}" for i in range(max_players - len(active_players))]

        if not all_players:
            # No existing players, create new ones
            assignments = {}
            for i in range(len(detections)):
                assignments[i] = f"player_{self.next_player_id}"
                self.next_player_id += 1
            return assignments

        # Build cost matrix
        cost_matrix = np.full((len(detections), len(all_players)), np.inf)

        for det_idx, detection in enumerate(detections):
            for player_idx, player_id in enumerate(all_players):
                if player_id.startswith("new_"):
                    # Cost for new player (prefer existing players)
                    cost_matrix[det_idx, player_idx] = 2.0
                else:
                    cost = self._calculate_assignment_cost(detection, player_id, frame_idx)
                    if cost <= self.max_distance_threshold:
                        cost_matrix[det_idx, player_idx] = cost

        # Solve assignment problem
        try:
            det_indices, player_indices = linear_sum_assignment(cost_matrix)
            assignments = {}

            for det_idx, player_idx in zip(det_indices, player_indices):
                if cost_matrix[det_idx, player_idx] < np.inf:
                    player_id = all_players[player_idx]
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
                print(f"Hungarian assignment failed: {e}, using fallback")
            # Fallback: simple greedy assignment
            return self._fallback_assignment(detections, frame_idx)

    def _fallback_assignment(self, detections: List[Dict], frame_idx: int) -> Dict[int, str]:
        """Fallback assignment when Hungarian algorithm fails."""
        assignments = {}
        used_players = set()

        for det_idx, detection in enumerate(detections):
            best_player = None
            best_cost = float('inf')

            # Try existing players
            for player_id in self.player_last_seen:
                if player_id in used_players:
                    continue
                if frame_idx - self.player_last_seen[player_id] <= self.occlusion_max_frames:
                    cost = self._calculate_assignment_cost(detection, player_id, frame_idx)
                    if cost < best_cost and cost <= self.max_distance_threshold:
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
        """Update player trajectory."""
        center_pos = detection['center_position']
        position = (center_pos['x'], center_pos['y'], frame_idx)

        if player_id not in self.player_trajectories:
            self.player_trajectories[player_id] = deque(maxlen=self.trajectory_history_frames)

        self.player_trajectories[player_id].append(position)
        self.player_last_seen[player_id] = frame_idx

    def process_frame_detections(self, frame_ankle_data: Dict[int, List[Dict]], frame_idx: int) -> Dict[str, Dict[str, Any]]:
        """Process frame detections and assign robust player IDs."""
        if not frame_ankle_data:
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
            print(f"Frame {frame_idx}: Processing {len(detections)} detections")

        # Assign player IDs
        assignments = self._assign_detections_to_players(detections, frame_idx)

        # Build result
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
        if not self.player_trajectories:
            return {}

        # Find most active players
        player_activity = {}
        for player_id, trajectory in self.player_trajectories.items():
            player_activity[player_id] = len(trajectory)

        # Sort by activity and take top 2
        sorted_players = sorted(player_activity.items(), key=lambda x: x[1], reverse=True)

        mapping = {}
        if len(sorted_players) >= 1:
            mapping[sorted_players[0][0]] = "player_0"
        if len(sorted_players) >= 2:
            mapping[sorted_players[1][0]] = "player_1"

        return mapping

    def print_tracking_stats(self):
        """Print tracking statistics."""
        if not self.player_trajectories:
            return

        print(f"\n=== Player Tracking Statistics ===")
        print(f"Total players tracked: {len(self.player_trajectories)}")

        for player_id, trajectory in self.player_trajectories.items():
            if not trajectory:
                continue

            frames_tracked = len(trajectory)
            first_frame = trajectory[0][2]
            last_frame = trajectory[-1][2]
            frame_span = last_frame - first_frame + 1
            coverage = frames_tracked / frame_span if frame_span > 0 else 0

            print(f"{player_id}: {frames_tracked} frames ({coverage:.1%} coverage)")

        print("==================================\n")


class EnhancedAnkleTracker:
    """Enhanced tracker with robust player ID system."""

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

        # Robust player tracking
        self.player_tracker = RobustPlayerTracker(
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

        # Use robust player tracking to assign IDs
        if frame_ankle_data:
            player_assignments = self.player_tracker.process_frame_detections(frame_ankle_data, frame_index)
            if player_assignments:
                self.frame_data_internal[frame_index] = player_assignments

    def process_all_frames(self) -> None:
        """Process all frames."""
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

        print(f"Processing {len(frames_data)} frames with pose data...")

        # Process each frame
        for frame_idx in sorted(frames_data.keys()):
            frame_data = frames_data[frame_idx]
            self.process_frame(frame_data, frame_idx)

        print(f"Processed {len(self.frame_data_internal)} frames with ankle detections")

        # Print tracking statistics
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
        """Validate tracking results."""
        if not self.frame_data_internal:
            print("⚠️  No frame data to validate")
            return

        # Sample positions for validation
        sample_positions = []
        for frame_data in list(self.frame_data_internal.values())[::max(1, len(self.frame_data_internal)//50)]:
            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    sample_positions.append(ankle)

        if len(sample_positions) < 10:
            print("⚠️  Too few positions for validation")
            return

        x_positions = [pos['world_x'] for pos in sample_positions]
        y_positions = [pos['world_y'] for pos in sample_positions]

        out_of_bounds = sum(1 for pos in sample_positions
                            if pos['world_x'] < -0.5 or pos['world_x'] > self.COURT_WIDTH + 0.5 or
                            pos['world_y'] < -0.5 or pos['world_y'] > self.COURT_LENGTH + 0.5)

        print(f"=== Ankle Tracking Quality ===")
        print(f"Positions analyzed: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds}/{len(sample_positions)} ({out_of_bounds/len(sample_positions):.1%})")

        if out_of_bounds/len(sample_positions) < 0.1:
            print("✅ Ankle tracking quality good")
        else:
            print("⚠️  High out-of-bounds ratio - check calibration")
        print("===============================\n")

    def save_results(self) -> None:
        """Save results in format compatible with stage 3 visualization."""
        # Convert to standard format
        frame_data_dict = self.convert_to_standard_format()

        # Calculate summary statistics
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

        # Create output data structure
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
                'robust_player_tracking': True
            },
            'enhancement_info': {
                'camera_height_meters': float(self.camera_height) if self.calibration_available and self.camera_height else None,
                'enhanced_ankle_offset_pixels': float(self.enhanced_ankle_offset) if self.calibration_available and self.enhanced_ankle_offset else None,
                'reprojection_error_px': float(self.reprojection_error) if self.calibration_available and self.reprojection_error else None,
                'robust_tracking_enabled': True,
                'max_distance_threshold_m': self.player_tracker.max_distance_threshold,
                'occlusion_max_frames': self.player_tracker.occlusion_max_frames
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
        print(f"📊 Robust tracking: Enabled")

    def run(self) -> None:
        """Run the enhanced ankle tracking pipeline."""
        print(f"🚀 Starting enhanced ankle tracking with robust player IDs: {self.video_name}")
        print("="*80)

        try:
            print("📊 Step 1: Loading calibration data...")
            self.load_calibration_data()

            print("📍 Step 2: Loading pose data...")
            self.load_pose_data()

            print("🔧 Step 3: Calculating homography...")
            self.calculate_homography()

            print("🏃 Step 4: Processing all frames with robust tracking...")
            self.process_all_frames()

            print("✅ Step 5: Validating results...")
            self.validate_results()

            print("💾 Step 6: Saving results...")
            self.save_results()

            print("="*80)
            print("✅ Enhanced ankle tracking with robust player IDs completed!")
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
        print("\nNew Features:")
        print("  - Robust player ID tracking with temporal consistency")
        print("  - Hungarian algorithm for optimal player-detection assignment")
        print("  - Trajectory-based prediction and validation")
        print("  - Enhanced homography with camera calibration support")
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
        print("❌ Error: SciPy is required for robust player tracking")
        print("Install with: pip install scipy")
        sys.exit(1)

    tracker = EnhancedAnkleTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()