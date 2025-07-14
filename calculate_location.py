#!/usr/bin/env python3
"""
Badminton Player Position Tracking Script - Simplified Version

Simplified player ID assignment based on Y position:
- Player with lowest average Y value (ankle + hip) = Player 0
- Player with highest average Y value (ankle + hip) = Player 1

Key features:
1. Ankle-prioritized positioning with hip fallback
2. Simple Y-position based player ID assignment
3. Enhanced trajectory analysis
4. Boundary validation

Usage: python calculate_location.py <video_file_path> [--debug]
"""

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque


class BadmintonCourtTracker:
    """Tracks player positions on badminton court using ankle-prioritized method."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4
    DEFAULT_HIP_HEIGHT = 0.9  # Fallback hip height

    # Pose joint indices
    HIP_LEFT = 11
    HIP_RIGHT = 12
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16
    NOSE = 0

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    ANKLE_TO_GROUND_OFFSET = 0.04  # 4cm from ankle joint to ground contact

    # Dynamic frame windowing parameters
    MIN_WINDOW_SIZE = 10
    MAX_WINDOW_SIZE = 60
    MOTION_THRESHOLD = 0.3  # meters/frame for detecting rapid movement

    def __init__(self, video_path: str, debug: bool = False):
        """Initialize tracker."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.output_file = self.results_dir / "positions.json"
        self.debug = debug

        # Data containers
        self.pose_data = None
        self.court_points = None
        self.video_info = None
        self.homography_matrix = None
        self.camera_matrix = None
        self.camera_height = None
        self.camera_position = None
        self.rotation_vector = None
        self.translation_vector = None
        self.player_positions = []

        # Enhanced tracking with dynamic windowing
        self._hip_height_history = deque(maxlen=self.MAX_WINDOW_SIZE)
        self._position_history = deque(maxlen=20)  # For motion analysis
        self._current_window_size = self.MIN_WINDOW_SIZE

    def load_pose_data(self) -> None:
        """Load pose detection data from JSON file."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.court_points = data.get('court_points', {})
        self.video_info = data.get('video_info', {})

        # Check required court points (assume they're already standardized)
        required_points = ['P1', 'P2', 'P3', 'P4']
        if not all(point in self.court_points for point in required_points):
            raise ValueError(f"Missing required court points: {required_points}")

        print(f"Loaded pose data with {len(data.get('pose_data', []))} pose detections")
        print("✓ Assuming court points are already standardized in clockwise order")

    def estimate_camera_intrinsics(self) -> np.ndarray:
        """Estimate camera intrinsic matrix from video dimensions."""
        width = self.video_info.get('width', 1920)
        height = self.video_info.get('height', 1080)

        # Conservative focal length estimation
        focal_length = float(width)
        cx = width / 2.0
        cy = height / 2.0

        if self.debug:
            print(f"Camera intrinsics: f={focal_length:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def solve_camera_pose(self) -> None:
        """Determine camera position and height using solvePnP."""
        # Image points (pixels) - standardized order expected
        image_points = np.array([
            self.court_points['P1'],  # Top-left
            self.court_points['P2'],  # Bottom-left
            self.court_points['P3'],  # Bottom-right
            self.court_points['P4']   # Top-right
        ], dtype=np.float32)

        # World coordinates (meters, Z=0 for ground) - corresponding to standardized order
        world_points_3d = np.array([
            [0, 0, 0],                    # P1: Top-left (0, 0)
            [0, self.COURT_LENGTH, 0],    # P2: Bottom-left (0, 13.4)
            [self.COURT_WIDTH, self.COURT_LENGTH, 0],  # P3: Bottom-right (6.1, 13.4)
            [self.COURT_WIDTH, 0, 0]      # P4: Top-right (6.1, 0)
        ], dtype=np.float32)

        self.camera_matrix = self.estimate_camera_intrinsics()
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            world_points_3d, image_points, self.camera_matrix, dist_coeffs
        )

        if not success:
            raise ValueError("Failed to solve camera pose")

        self.rotation_vector = rvec
        self.translation_vector = tvec
        self.camera_height = abs(float(tvec[2][0]))

        # Calculate camera position in world coordinates
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_position = -rotation_matrix.T @ tvec
        self.camera_position = camera_position.flatten()

        print(f"Camera height: {self.camera_height:.2f} meters")
        print(f"Camera position: X={self.camera_position[0]:.2f}, Y={self.camera_position[1]:.2f}")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        # Image points (standardized order expected)
        image_points = np.array([
            self.court_points['P1'],  # Top-left
            self.court_points['P2'],  # Bottom-left
            self.court_points['P3'],  # Bottom-right
            self.court_points['P4']   # Top-right
        ], dtype=np.float32)

        # World points (corresponding to the standardized image points)
        world_points = np.array([
            [0, 0],                    # P1: Top-left (0, 0)
            [0, self.COURT_LENGTH],    # P2: Bottom-left (0, 13.4)
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Bottom-right (6.1, 13.4)
            [self.COURT_WIDTH, 0]      # P4: Top-right (6.1, 0)
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

        print("Homography matrix calculated successfully")

    def validate_coordinate_system(self) -> None:
        """Validate coordinate system accuracy."""
        print("\n=== Coordinate System Validation ===")

        corners = ['P1', 'P2', 'P3', 'P4']
        expected_world = [(0, 0), (0, self.COURT_LENGTH), (self.COURT_WIDTH, self.COURT_LENGTH), (self.COURT_WIDTH, 0)]
        corner_names = ['Top-Left', 'Bottom-Left', 'Bottom-Right', 'Top-Right']

        max_error = 0.0
        for corner, (exp_x, exp_y), name in zip(corners, expected_world, corner_names):
            image_point = np.array([[self.court_points[corner]]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(image_point, self.homography_matrix)
            actual_x, actual_y = world_point[0][0]

            error = np.sqrt((actual_x - exp_x)**2 + (actual_y - exp_y)**2)
            max_error = max(max_error, error)

            print(f"  {corner} ({name}): Expected ({exp_x:.1f}, {exp_y:.1f}), Got ({actual_x:.1f}, {actual_y:.1f}), Error: {error:.3f}m")

        print(f"Maximum transformation error: {max_error:.3f} meters")

        if max_error < 0.15:
            print("✓ Coordinate system validation passed")
        else:
            print("⚠ Coordinate system has errors - check court point detection")

        print("=======================================\n")

    def transform_point_to_world(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform pixel point to world coordinates using homography."""
        point = np.array([[pixel_point]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def estimate_distance_to_point(self, pixel_point: Tuple[float, float]) -> float:
        """Estimate distance from camera to a point."""
        try:
            world_x, world_y = self.transform_point_to_world(pixel_point)
            distance = np.sqrt(
                (world_x - self.camera_position[0])**2 +
                (world_y - self.camera_position[1])**2
            )
            return max(distance, 1.0)  # Minimum 1m distance
        except:
            return 5.0  # Fallback distance

    def calculate_ankle_pixel_offset(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate pixel offset for ankle to ground correction."""
        try:
            distance_to_player = self.estimate_distance_to_point(ankle_pixel)
            vertical_angle = np.arctan(self.camera_height / distance_to_player)

            # Calculate pixel offset for ankle-to-ground correction
            pixel_offset_y = (self.ANKLE_TO_GROUND_OFFSET * self.camera_matrix[1, 1]) / distance_to_player
            perspective_factor = 1.0 / max(np.cos(vertical_angle), 0.1)
            corrected_offset_y = pixel_offset_y * perspective_factor

            return (0.0, corrected_offset_y)
        except Exception:
            return (0.0, 0.0)

    def project_ankle_to_ground_with_homography(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Project ankle to ground using pre-homography pixel offset correction."""
        offset_x, offset_y = self.calculate_ankle_pixel_offset(ankle_pixel)
        corrected_pixel = (ankle_pixel[0] + offset_x, ankle_pixel[1] + offset_y)

        corrected_world_x, corrected_world_y = self.transform_point_to_world(corrected_pixel)

        # Boundary validation - clamp to reasonable court area
        corrected_world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, corrected_world_x))
        corrected_world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, corrected_world_y))

        return float(corrected_world_x), float(corrected_world_y)

    def analyze_motion_dynamics(self, current_pos: Tuple[float, float]) -> float:
        """Analyze motion dynamics to determine optimal window size."""
        if len(self._position_history) < 3:
            return self.MIN_WINDOW_SIZE

        # Calculate recent velocities
        velocities = []
        for i in range(1, min(5, len(self._position_history))):
            prev_pos = self._position_history[-i-1]
            curr_pos = self._position_history[-i]
            velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            velocities.append(velocity)

        avg_velocity = np.mean(velocities) if velocities else 0.0

        # Adaptive window sizing based on motion
        if avg_velocity > self.MOTION_THRESHOLD:
            # Rapid movement - use smaller window for responsiveness
            return self.MIN_WINDOW_SIZE
        else:
            # Slower movement - use larger window for stability
            return min(self.MAX_WINDOW_SIZE, self.MIN_WINDOW_SIZE + int(20 * (1 - avg_velocity)))

    def estimate_hip_height_from_pose(self, joints: List[Dict]) -> Optional[float]:
        """Estimate hip height using improved body proportions."""
        head = self.extract_joint_position(joints, self.NOSE)
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)

        if not head or not (ankle_left or ankle_right) or not (hip_left or hip_right):
            return None

        # Get reference points
        ankle_y = min(ankle_left[1] if ankle_left else float('inf'),
                      ankle_right[1] if ankle_right else float('inf'))

        if hip_left and hip_right:
            hip_y = (hip_left[1] + hip_right[1]) / 2
        else:
            hip_y = hip_left[1] if hip_left else hip_right[1]

        # Calculate proportional height
        total_height_pixels = ankle_y - head[1]
        hip_height_pixels = ankle_y - hip_y

        if total_height_pixels <= 0:
            return None

        # Improved body proportion estimation
        hip_ratio = hip_height_pixels / total_height_pixels

        # Estimate total height from hip ratio (typical range 0.45-0.58)
        if 0.4 <= hip_ratio <= 0.65:
            estimated_total_height = 1.65 + (hip_ratio - 0.52) * 2.0  # Adaptive height
            estimated_hip_height = estimated_total_height * hip_ratio
        else:
            # Fallback to default
            estimated_hip_height = self.DEFAULT_HIP_HEIGHT

        return max(0.6, min(1.2, estimated_hip_height))

    def estimate_hip_height_from_geometry(self, hip_pixel: Tuple[float, float],
                                          ankle_pixels: List[Tuple[float, float]]) -> Optional[float]:
        """Estimate hip height using camera geometry."""
        if not ankle_pixels:
            return None

        try:
            avg_ankle_x = sum(pos[0] for pos in ankle_pixels) / len(ankle_pixels)
            avg_ankle_y = sum(pos[1] for pos in ankle_pixels) / len(ankle_pixels)

            pixel_height_diff = abs(hip_pixel[1] - avg_ankle_y)

            # Use corrected ankle position for distance calculation
            ankle_ground = self.project_ankle_to_ground_with_homography((avg_ankle_x, avg_ankle_y))
            camera_to_player_distance = np.sqrt(
                (ankle_ground[0] - self.camera_position[0])**2 +
                (ankle_ground[1] - self.camera_position[1])**2
            )

            focal_length = self.camera_matrix[0, 0]
            estimated_hip_height = (pixel_height_diff * camera_to_player_distance) / focal_length

            return max(0.6, min(1.2, estimated_hip_height))

        except Exception:
            return None

    def get_adaptive_hip_height(self, joints: List[Dict], frame_index: int) -> float:
        """Calculate adaptive hip height with dynamic windowing."""
        height_estimates = []

        # Method 1: Improved body proportions
        pose_height = self.estimate_hip_height_from_pose(joints)
        if pose_height is not None:
            height_estimates.append(pose_height)

        # Method 2: Geometry-based estimation
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        if hip_left and hip_right:
            hip_center = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
            ankle_positions = [pos for pos in [ankle_left, ankle_right] if pos is not None]

            if ankle_positions:
                geom_height = self.estimate_hip_height_from_geometry(hip_center, ankle_positions)
                if geom_height is not None:
                    height_estimates.append(geom_height)

        # Process estimates with adaptive windowing
        if height_estimates:
            current_estimate = np.median(height_estimates)  # Use median for robustness

            # Determine optimal window size based on motion
            if hip_left and hip_right:
                current_pos = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
                self._position_history.append(current_pos)
                self._current_window_size = self.analyze_motion_dynamics(current_pos)

            # Add to history with bounds checking
            if 0.6 <= current_estimate <= 1.2:
                self._hip_height_history.append(current_estimate)

            # Calculate smoothed estimate using current window size
            if self._hip_height_history:
                window_data = list(self._hip_height_history)[-self._current_window_size:]
                smoothed_height = np.mean(window_data)

                if self.debug:
                    print(f"Frame {frame_index}: Hip height = {current_estimate:.3f}m, "
                          f"Smoothed = {smoothed_height:.3f}m, Window = {len(window_data)}")

                return smoothed_height

        return self.DEFAULT_HIP_HEIGHT

    def calculate_hip_pixel_offset(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Calculate pixel offset for hip to ground correction."""
        try:
            distance_to_player = self.estimate_distance_to_point(hip_pixel)
            vertical_angle = np.arctan(self.camera_height / distance_to_player)

            pixel_offset_y = (hip_height * self.camera_matrix[1, 1]) / distance_to_player
            perspective_factor = 1.0 / max(np.cos(vertical_angle), 0.1)
            corrected_offset_y = pixel_offset_y * perspective_factor

            return (0.0, corrected_offset_y)
        except Exception:
            return (0.0, 0.0)

    def project_hip_to_ground(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Project hip to ground using pre-homography pixel offset correction."""
        offset_x, offset_y = self.calculate_hip_pixel_offset(hip_pixel, hip_height)
        corrected_pixel = (hip_pixel[0] + offset_x, hip_pixel[1] + offset_y)

        corrected_world_x, corrected_world_y = self.transform_point_to_world(corrected_pixel)

        # Boundary validation
        corrected_world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, corrected_world_x))
        corrected_world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, corrected_world_y))

        return float(corrected_world_x), float(corrected_world_y)

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence is sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return joint['x'], joint['y']
        return None

    def calculate_player_position(self, joints: List[Dict], frame_index: int) -> Optional[Dict[str, float]]:
        """Calculate world position using ankle-prioritized method."""
        # Extract joints
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        # Prioritize ankle positions
        ankle_positions = []
        left_ankle_world = (0.0, 0.0)
        right_ankle_world = (0.0, 0.0)

        if ankle_left:
            left_ankle_world = self.project_ankle_to_ground_with_homography(ankle_left)
            ankle_positions.append(left_ankle_world)
        if ankle_right:
            right_ankle_world = self.project_ankle_to_ground_with_homography(ankle_right)
            ankle_positions.append(right_ankle_world)

        # Hip position as supplementary data
        hip_ground_x, hip_ground_y = 0.0, 0.0
        if hip_left or hip_right:
            if hip_left and hip_right:
                hip_center = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
            else:
                hip_center = hip_left if hip_left else hip_right

            adaptive_hip_height = self.get_adaptive_hip_height(joints, frame_index)
            hip_ground_x, hip_ground_y = self.project_hip_to_ground(hip_center, adaptive_hip_height)

        # Ankle-prioritized weighted average
        if ankle_positions:
            positions = ankle_positions + [(hip_ground_x, hip_ground_y)] if (hip_left or hip_right) else ankle_positions
            # Higher weight for ankles, lower for hip
            weights = [0.4] * len(ankle_positions) + ([0.2] if (hip_left or hip_right) else [])
        elif hip_left or hip_right:
            # Fallback to hip only
            positions = [(hip_ground_x, hip_ground_y)]
            weights = [1.0]
        else:
            return None

        # Normalize weights and calculate final position
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0]

        final_x = sum(pos[0] * weight for pos, weight in zip(positions, normalized_weights))
        final_y = sum(pos[1] * weight for pos, weight in zip(positions, normalized_weights))

        return {
            'hip_world_X': final_x,
            'hip_world_Y': final_y,
            'left_ankle_world_X': left_ankle_world[0],
            'left_ankle_world_Y': left_ankle_world[1],
            'right_ankle_world_X': right_ankle_world[0],
            'right_ankle_world_Y': right_ankle_world[1],
            'estimated_hip_height': self.get_adaptive_hip_height(joints, frame_index) if (hip_left or hip_right) else self.DEFAULT_HIP_HEIGHT
        }

    def assign_simple_player_ids(self, frame_positions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Simple Y-position based player ID assignment - consistent lowest Y = Player 0."""
        if len(frame_positions) == 0:
            return []
        elif len(frame_positions) == 1:
            position_entry = frame_positions[0].copy()

            # Get Y values from ankle and hip positions
            y_values = []
            y_values.append(position_entry['hip_world_Y'])

            if position_entry['left_ankle_world_X'] != 0.0 or position_entry['left_ankle_world_Y'] != 0.0:
                y_values.append(position_entry['left_ankle_world_Y'])
            if position_entry['right_ankle_world_X'] != 0.0 or position_entry['right_ankle_world_Y'] != 0.0:
                y_values.append(position_entry['right_ankle_world_Y'])

            avg_y = sum(y_values) / len(y_values) if y_values else 6.7

            # Assign based on court position relative to center line (6.7m)
            # Lower Y values = closer to near baseline = Player 0
            # Higher Y values = closer to far baseline = Player 1
            if avg_y < 6.7:
                position_entry['player_id'] = 0  # Closer to near baseline (lower Y)
            else:
                position_entry['player_id'] = 1  # Closer to far baseline (higher Y)

            return [position_entry]
        else:
            # Multiple players - consistent assignment based on Y position
            positions_with_y = []
            for pos in frame_positions:
                # Get Y values from ankle and hip positions
                y_values = []
                y_values.append(pos['hip_world_Y'])

                if pos['left_ankle_world_X'] != 0.0 or pos['left_ankle_world_Y'] != 0.0:
                    y_values.append(pos['left_ankle_world_Y'])
                if pos['right_ankle_world_X'] != 0.0 or pos['right_ankle_world_Y'] != 0.0:
                    y_values.append(pos['right_ankle_world_Y'])

                avg_y = sum(y_values) / len(y_values) if y_values else float('inf')
                positions_with_y.append((pos, avg_y))

            # Sort by Y position: lowest Y first
            positions_with_y.sort(key=lambda x: x[1])

            # Assign IDs consistently: lowest Y = Player 0, highest Y = Player 1
            result = []
            for i, (pos, y_pos) in enumerate(positions_with_y[:2]):  # Max 2 players
                position_entry = pos.copy()
                position_entry['player_id'] = i  # i=0 for lowest Y, i=1 for highest Y
                result.append(position_entry)

            return result

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> List[Dict[str, Any]]:
        """Process all players in a single frame."""
        frame_positions = []

        for human_data in frame_data:
            joints = human_data.get('joints', [])
            position = self.calculate_player_position(joints, frame_index)

            if position:
                frame_positions.append(position)

        if self.debug and len(frame_positions) > 2:
            print(f"Frame {frame_index}: {len(frame_positions)} detections before processing")

        # Assign player IDs based on Y position
        frame_positions = self.assign_simple_player_ids(frame_positions)

        if self.debug and len(frame_positions) > 2:
            print(f"Frame {frame_index}: {len(frame_positions)} detections after processing")

        return frame_positions

    def process_all_frames(self) -> None:
        """Process all frames with simplified ID assignment."""
        pose_data = self.pose_data.get('pose_data', [])

        # Group by frame
        frames_data = {}
        for entry in pose_data:
            frame_idx = entry['frame_index']
            if frame_idx not in frames_data:
                frames_data[frame_idx] = []
            frames_data[frame_idx].append(entry)

        print(f"Processing {len(frames_data)} frames...")

        # Process each frame
        for frame_idx in sorted(frames_data.keys()):
            frame_data = frames_data[frame_idx]
            positions = self.process_frame(frame_data, frame_idx)

            for position in positions:
                position['frame_index'] = frame_idx
                self.player_positions.append(position)

        print(f"Extracted {len(self.player_positions)} player positions")

        # Statistics
        if self._hip_height_history:
            heights = list(self._hip_height_history)
            avg_height = np.mean(heights)
            std_height = np.std(heights)
            print(f"Hip height stats: avg={avg_height:.3f}m ±{std_height:.3f}m")

        # Report final distribution
        player_0_frames = len([p for p in self.player_positions if p['player_id'] == 0])
        player_1_frames = len([p for p in self.player_positions if p['player_id'] == 1])

        print(f"Final player distribution:")
        print(f"  Player 0 (lower Y values): {player_0_frames} frames")
        print(f"  Player 1 (higher Y values): {player_1_frames} frames")

    def detect_potential_issues(self) -> None:
        """Detect potential issues in tracking results."""
        if len(self.player_positions) < 10:
            return

        # Sample positions for analysis
        sample_positions = self.player_positions[::max(1, len(self.player_positions)//50)]
        x_positions = [pos['hip_world_X'] for pos in sample_positions]
        y_positions = [pos['hip_world_Y'] for pos in sample_positions]

        # Check for boundary violations
        out_of_bounds = 0
        for pos in sample_positions:
            x, y = pos['hip_world_X'], pos['hip_world_Y']
            if x < -0.5 or x > self.COURT_WIDTH + 0.5 or y < -0.5 or y > self.COURT_LENGTH + 0.5:
                out_of_bounds += 1

        out_of_bounds_ratio = out_of_bounds / len(sample_positions)

        print(f"\n=== Tracking Quality Assessment ===")
        print(f"Positions analyzed: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds_ratio:.1%}")
        print(f"ID assignment method: Y-position based (lower Y = Player 0)")

        if out_of_bounds_ratio > 0.1:
            print("⚠ High out-of-bounds ratio detected - check calibration")
        else:
            print("✓ Tracking quality appears acceptable")
        print("=====================================\n")

    def save_results(self) -> None:
        """Save tracking results to JSON file."""
        frames_with_players = len(set(pos['frame_index'] for pos in self.player_positions))

        output_data = {
            'court_points': self.court_points,
            'all_court_points': self.pose_data.get('all_court_points', self.court_points),
            'video_info': self.video_info,
            'player_positions': self.player_positions,
            'tracking_method': "Simplified ankle-prioritized positioning with Y-position based player ID assignment",
            'processing_info': {
                'total_positions': len(self.player_positions),
                'frames_with_players': frames_with_players,
                'ankle_to_ground_offset_meters': self.ANKLE_TO_GROUND_OFFSET,
                'camera_height_meters': self.camera_height,
                'camera_position': self.camera_position.tolist() if self.camera_position is not None else None,
                'court_points_pre_standardized': True,
                'id_assignment_method': 'Consistent Y-position based (single pose: above/below 6.7m = Player 0/1, multiple poses: lowest Y = Player 0, highest Y = Player 1)',
                'simplified_features': {
                    'ankle_prioritized_weighting': True,
                    'dynamic_frame_windowing': True,
                    'adaptive_hip_height': True,
                    'motion_based_windowing': True,
                    'boundary_validation': True,
                    'simple_y_based_id_assignment': True
                },
                'weighting_scheme': {
                    'ankle_weight': 0.4,
                    'hip_weight': 0.2,
                    'note': 'Per ankle joint when detected'
                },
                'window_size_range': [self.MIN_WINDOW_SIZE, self.MAX_WINDOW_SIZE],
                'y_position_calculation': 'Average of hip_world_Y + valid ankle_world_Y values. Single pose: Player 0 if <6.7m, Player 1 if >=6.7m. Multiple poses: sorted by Y - lowest Y = Player 0, highest Y = Player 1.'
            }
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {self.output_file}")
        print(f"Total positions: {len(self.player_positions)}")
        print(f"Frames with players: {frames_with_players}")

    def run(self) -> None:
        """Run the simplified tracking pipeline."""
        print(f"Starting simplified position tracking for: {self.video_name}")

        try:
            self.load_pose_data()
            self.solve_camera_pose()
            self.calculate_homography()
            self.validate_coordinate_system()
            self.process_all_frames()
            self.detect_potential_issues()
            self.save_results()
            print("✓ Simplified position tracking completed successfully!")

        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_location.py <video_file_path> [--debug]")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    tracker = BadmintonCourtTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()