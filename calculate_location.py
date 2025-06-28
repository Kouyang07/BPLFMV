#!/usr/bin/env python3
"""
Badminton Player Position Tracking Script

Converts pose detection data to precise 2D world coordinates on badminton court.
Uses adaptive hip height calculation and ray-plane intersection for accuracy.
Limits tracking to exactly two players by merging multiple detections.
FIXED: Applies ankle offset in pixel space before homography transformation for better accuracy.

Usage: python track_positions.py <video_file_path> [--debug]
"""

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class BadmintonCourtTracker:
    """Tracks player positions on badminton court using adaptive hip height."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4
    DEFAULT_HIP_HEIGHT = 0.9  # Fallback hip height

    # Pose joint indices
    HIP_LEFT = 11
    HIP_RIGHT = 12
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    MAX_TRACKING_DISTANCE = 2.0  # Max distance for player ID tracking
    MERGE_DISTANCE_THRESHOLD = 1.5  # Distance threshold for merging players (meters)
    MIN_FRAMES_FOR_PLAYER = 5  # Minimum frames to consider a valid player

    # Ankle offset correction (distance from ankle joint to ground contact point)
    ANKLE_TO_GROUND_OFFSET = 0.07

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
        self._hip_height_history = []
        self._last_positions = []
        self._next_id = 0

        # Two-player system
        self._player_trajectories = defaultdict(list)  # Track all positions per player ID
        self._final_player_mapping = {}  # Maps original IDs to final player IDs (0 or 1)

    def load_pose_data(self) -> None:
        """Load pose detection data from JSON file."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.court_points = data.get('court_points', {})
        self.video_info = data.get('video_info', {})

        # Check required court points
        required_points = ['P1', 'P2', 'P3', 'P4']
        if not all(point in self.court_points for point in required_points):
            raise ValueError(f"Missing required court points: {required_points}")

        print(f"Loaded pose data with {len(data.get('pose_data', []))} pose detections")
        print("Court points mapping:")
        print("  P1: Upper baseline + LEFT sideline")
        print("  P2: Lower baseline + LEFT sideline")
        print("  P3: Lower baseline + RIGHT sideline")
        print("  P4: Upper baseline + RIGHT sideline")

    def estimate_camera_intrinsics(self) -> np.ndarray:
        """Estimate camera intrinsic matrix from video dimensions."""
        width = self.video_info.get('width', 1920)
        height = self.video_info.get('height', 1080)

        # Focal length equals video width (paper's method)
        focal_length = float(width)
        cx = width / 2.0
        cy = height / 2.0

        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def solve_camera_pose(self) -> None:
        """Determine camera position and height using solvePnP."""
        # Image points (pixels) - corrected order based on actual court point definitions
        image_points = np.array([
            self.court_points['P1'],  # Upper baseline + LEFT sideline
            self.court_points['P2'],  # Lower baseline + LEFT sideline
            self.court_points['P3'],  # Lower baseline + RIGHT sideline
            self.court_points['P4']   # Upper baseline + RIGHT sideline
        ], dtype=np.float32)

        # World coordinates (meters, Z=0 for ground) - corrected to match P1-P4 definitions
        world_points_3d = np.array([
            [0, 0, 0],                    # P1: Left side, top (0, 0)
            [0, self.COURT_LENGTH, 0],    # P2: Left side, bottom (0, 13.4)
            [self.COURT_WIDTH, self.COURT_LENGTH, 0],  # P3: Right side, bottom (6.1, 13.4)
            [self.COURT_WIDTH, 0, 0]      # P4: Right side, top (6.1, 0)
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
        self.camera_height = abs(float(tvec[2][0]))  # Fix deprecation warning

        # Calculate camera position in world coordinates
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_position = -rotation_matrix.T @ tvec
        self.camera_position = camera_position.flatten()

        print(f"Camera height: {self.camera_height:.2f} meters")
        print(f"Camera position: X={self.camera_position[0]:.2f}, Y={self.camera_position[1]:.2f}")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        # Image points - same corrected order as solve_camera_pose
        image_points = np.array([
            self.court_points['P1'],  # Upper baseline + LEFT sideline
            self.court_points['P2'],  # Lower baseline + LEFT sideline
            self.court_points['P3'],  # Lower baseline + RIGHT sideline
            self.court_points['P4']   # Upper baseline + RIGHT sideline
        ], dtype=np.float32)

        # World points - corrected to match the actual court layout
        world_points = np.array([
            [0, 0],                      # P1: Left side, top (0, 0)
            [0, self.COURT_LENGTH],      # P2: Left side, bottom (0, 13.4)
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Right side, bottom (6.1, 13.4)
            [self.COURT_WIDTH, 0]        # P4: Right side, top (6.1, 0)
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

        print("Homography matrix calculated successfully")

    def validate_coordinate_system(self) -> None:
        """Validate that the coordinate system is set up correctly."""
        print("\n=== Coordinate System Validation ===")

        # Test transformation of court center
        court_center_world = np.array([[self.COURT_WIDTH/2, self.COURT_LENGTH/2]], dtype=np.float32)

        # Transform to image coordinates
        court_center_image = cv2.perspectiveTransform(
            court_center_world.reshape(1, 1, 2),
            np.linalg.inv(self.homography_matrix)
        )

        print(f"Court center (world): ({self.COURT_WIDTH/2:.1f}, {self.COURT_LENGTH/2:.1f})")
        print(f"Court center (image): ({court_center_image[0][0][0]:.1f}, {court_center_image[0][0][1]:.1f})")

        # Test corners - verify homography accuracy
        corners = ['P1', 'P2', 'P3', 'P4']
        expected_world = [(0, 0), (0, self.COURT_LENGTH), (self.COURT_WIDTH, self.COURT_LENGTH), (self.COURT_WIDTH, 0)]

        print("\nCorner validation:")
        max_error = 0.0
        for corner, (exp_x, exp_y) in zip(corners, expected_world):
            image_point = np.array([[self.court_points[corner]]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(image_point, self.homography_matrix)
            actual_x, actual_y = world_point[0][0]

            error = np.sqrt((actual_x - exp_x)**2 + (actual_y - exp_y)**2)
            max_error = max(max_error, error)

            print(f"  {corner}: Expected ({exp_x:.1f}, {exp_y:.1f}), Got ({actual_x:.1f}, {actual_y:.1f}), Error: {error:.3f}m")

            # Check if transformation is reasonable (within 0.5m tolerance)
            if error > 0.5:
                print(f"  WARNING: Large error in {corner} transformation!")

        print(f"Maximum transformation error: {max_error:.3f} meters")

        if max_error < 0.1:
            print("✓ Coordinate system appears correct")
        elif max_error < 0.5:
            print("⚠ Coordinate system has minor errors but should work")
        else:
            print("✗ Coordinate system has significant errors - check court point detection")

        print("=======================================\n")

    def transform_point_to_world(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform pixel point to world coordinates using homography."""
        point = np.array([[pixel_point]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def estimate_distance_to_point(self, pixel_point: Tuple[float, float]) -> float:
        """Estimate distance from camera to a point using rough homography projection."""
        try:
            # Get rough world position
            world_x, world_y = self.transform_point_to_world(pixel_point)

            # Calculate 2D distance to camera position
            distance = np.sqrt(
                (world_x - self.camera_position[0])**2 +
                (world_y - self.camera_position[1])**2
            )

            return max(distance, 1.0)  # Ensure minimum distance of 1m
        except:
            return 5.0  # Default fallback distance

    def calculate_ankle_pixel_offset(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate pixel offset for ankle to ground correction (applied before homography)."""
        try:
            # Estimate distance to player for this ankle position
            distance_to_player = self.estimate_distance_to_point(ankle_pixel)

            # Calculate the vertical angle from camera to player
            vertical_angle = np.arctan(self.camera_height / distance_to_player)

            # Calculate pixel offset for 14cm vertical drop
            # This projects the 3D offset to 2D pixel space
            pixel_offset_y = (self.ANKLE_TO_GROUND_OFFSET * self.camera_matrix[1, 1]) / distance_to_player

            # Apply perspective correction based on viewing angle
            # Steeper angles (higher vertical_angle) need more correction
            perspective_factor = 1.0 / np.cos(vertical_angle) if np.cos(vertical_angle) > 0.1 else 1.0
            corrected_offset_y = pixel_offset_y * perspective_factor

            if self.debug:
                print(f"Ankle pixel: {ankle_pixel}")
                print(f"Distance to player: {distance_to_player:.2f}m")
                print(f"Vertical angle: {np.degrees(vertical_angle):.1f}°")
                print(f"Pixel offset Y: {corrected_offset_y:.2f}px")

            return (0.0, corrected_offset_y)  # Only Y offset, no X offset

        except Exception as e:
            if self.debug:
                print(f"Error calculating ankle offset: {e}")
            return (0.0, 0.0)

    def project_ankle_to_ground_with_homography(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Project ankle to ground using pre-homography pixel offset correction."""
        # Calculate pixel offset for 14cm drop
        offset_x, offset_y = self.calculate_ankle_pixel_offset(ankle_pixel)

        # Apply offset in pixel space BEFORE homography transformation
        corrected_pixel = (ankle_pixel[0] + offset_x, ankle_pixel[1] + offset_y)

        # Now transform the corrected pixel position to world coordinates
        corrected_world_x, corrected_world_y = self.transform_point_to_world(corrected_pixel)

        if self.debug:
            # Compare with uncorrected version for debugging
            uncorrected_world_x, uncorrected_world_y = self.transform_point_to_world(ankle_pixel)
            print(f"Original pixel: {ankle_pixel}")
            print(f"Corrected pixel: {corrected_pixel}")
            print(f"Uncorrected world: ({uncorrected_world_x:.2f}, {uncorrected_world_y:.2f})")
            print(f"Corrected world: ({corrected_world_x:.2f}, {corrected_world_y:.2f})")
            print(f"World correction: ({corrected_world_x - uncorrected_world_x:.3f}, {corrected_world_y - uncorrected_world_y:.3f})")
            print("---")

        return float(corrected_world_x), float(corrected_world_y)

    def estimate_hip_height_from_pose(self, joints: List[Dict]) -> Optional[float]:
        """Estimate hip height using body proportions."""
        # Get key joints
        head_top = self.extract_joint_position(joints, 0)  # Nose
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)

        if not head_top or not (ankle_left or ankle_right) or not (hip_left or hip_right):
            return None

        # Get lowest ankle and hip center
        ankle_y = min(ankle_left[1] if ankle_left else float('inf'),
                      ankle_right[1] if ankle_right else float('inf'))

        if hip_left and hip_right:
            hip_y = (hip_left[1] + hip_right[1]) / 2
        else:
            hip_y = hip_left[1] if hip_left else hip_right[1]

        # Calculate pixel heights
        total_height_pixels = ankle_y - head_top[1]
        hip_height_pixels = ankle_y - hip_y

        if total_height_pixels <= 0:
            return None

        # Hip is ~53% of total height, average human is 1.7m
        estimated_total_height = 1.7
        hip_ratio = hip_height_pixels / total_height_pixels
        estimated_hip_height = estimated_total_height * hip_ratio

        # Clamp to reasonable range
        return max(0.5, min(1.5, estimated_hip_height))

    def estimate_hip_height_from_geometry(self, hip_pixel: Tuple[float, float],
                                          ankle_pixels: List[Tuple[float, float]]) -> Optional[float]:
        """Estimate hip height using camera geometry."""
        if not ankle_pixels or self.camera_matrix is None:
            return None

        try:
            # Average ankle position
            avg_ankle_x = sum(pos[0] for pos in ankle_pixels) / len(ankle_pixels)
            avg_ankle_y = sum(pos[1] for pos in ankle_pixels) / len(ankle_pixels)

            # Pixel height difference
            pixel_height_diff = abs(hip_pixel[1] - avg_ankle_y)

            # Get distance to player using projected ground position
            ankle_ground = self.project_ankle_to_ground_with_homography((avg_ankle_x, avg_ankle_y))
            camera_to_player_distance = np.sqrt(
                (ankle_ground[0] - self.camera_position[0])**2 +
                (ankle_ground[1] - self.camera_position[1])**2
            )

            # Convert pixel height to real height using similar triangles
            focal_length = self.camera_matrix[0, 0]
            estimated_hip_height = (pixel_height_diff * camera_to_player_distance) / focal_length

            return max(0.5, min(1.5, estimated_hip_height))

        except Exception:
            return None

    def get_adaptive_hip_height(self, joints: List[Dict], frame_index: int) -> float:
        """Calculate adaptive hip height with smoothing."""
        height_estimates = []

        # Method 1: Body proportions
        pose_height = self.estimate_hip_height_from_pose(joints)
        if pose_height is not None:
            height_estimates.append(pose_height)

        # Method 2: Geometry
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

        # Use estimates with smoothing
        if height_estimates:
            current_estimate = sum(height_estimates) / len(height_estimates)

            # Add to history (keep last 30 frames)
            self._hip_height_history.append(current_estimate)
            if len(self._hip_height_history) > 30:
                self._hip_height_history.pop(0)

            # Return smoothed estimate
            smoothed_height = sum(self._hip_height_history) / len(self._hip_height_history)

            if self.debug:
                print(f"Frame {frame_index}: Hip height = {current_estimate:.3f}m, "
                      f"Smoothed = {smoothed_height:.3f}m")

            return smoothed_height

        return self.DEFAULT_HIP_HEIGHT

    def calculate_hip_pixel_offset(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Calculate pixel offset for hip to ground correction (applied before homography)."""
        try:
            # Estimate distance to player for this hip position
            distance_to_player = self.estimate_distance_to_point(hip_pixel)

            # Calculate the vertical angle from camera to player
            vertical_angle = np.arctan(self.camera_height / distance_to_player)

            # Calculate pixel offset for hip_height vertical drop
            # This projects the 3D offset to 2D pixel space
            pixel_offset_y = (hip_height * self.camera_matrix[1, 1]) / distance_to_player

            # Apply perspective correction based on viewing angle
            # Steeper angles (higher vertical_angle) need more correction
            perspective_factor = 1.0 / np.cos(vertical_angle) if np.cos(vertical_angle) > 0.1 else 1.0
            corrected_offset_y = pixel_offset_y * perspective_factor

            if self.debug:
                print(f"Hip pixel: {hip_pixel}")
                print(f"Hip height: {hip_height:.3f}m")
                print(f"Distance to player: {distance_to_player:.2f}m")
                print(f"Vertical angle: {np.degrees(vertical_angle):.1f}°")
                print(f"Pixel offset Y: {corrected_offset_y:.2f}px")

            return (0.0, corrected_offset_y)  # Only Y offset, no X offset

        except Exception as e:
            if self.debug:
                print(f"Error calculating hip offset: {e}")
            return (0.0, 0.0)

    def project_hip_to_ground(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Project hip to ground using pre-homography pixel offset correction (same as ankle method)."""
        # Calculate pixel offset for hip_height drop
        offset_x, offset_y = self.calculate_hip_pixel_offset(hip_pixel, hip_height)

        # Apply offset in pixel space BEFORE homography transformation
        corrected_pixel = (hip_pixel[0] + offset_x, hip_pixel[1] + offset_y)

        # Now transform the corrected pixel position to world coordinates
        corrected_world_x, corrected_world_y = self.transform_point_to_world(corrected_pixel)

        if self.debug:
            # Compare with uncorrected version for debugging
            uncorrected_world_x, uncorrected_world_y = self.transform_point_to_world(hip_pixel)
            print(f"Hip original pixel: {hip_pixel}")
            print(f"Hip corrected pixel: {corrected_pixel}")
            print(f"Hip uncorrected world: ({uncorrected_world_x:.2f}, {uncorrected_world_y:.2f})")
            print(f"Hip corrected world: ({corrected_world_x:.2f}, {corrected_world_y:.2f})")
            print(f"Hip world correction: ({corrected_world_x - uncorrected_world_x:.3f}, {corrected_world_y - uncorrected_world_y:.3f})")
            print("---")

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
        """Calculate world position using adaptive hip height and pre-homography offset for ankles."""
        # Extract joints
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        # Need at least one hip
        if not hip_left and not hip_right:
            return None

        # Calculate hip center
        if hip_left and hip_right:
            hip_center = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
        else:
            hip_center = hip_left if hip_left else hip_right

        # Get adaptive hip height and project to ground
        adaptive_hip_height = self.get_adaptive_hip_height(joints, frame_index)
        hip_ground_x, hip_ground_y = self.project_hip_to_ground(hip_center, adaptive_hip_height)

        # Project ankles to ground using improved pre-homography offset correction
        ankle_positions = []
        left_ankle_world = (0.0, 0.0)
        right_ankle_world = (0.0, 0.0)

        if ankle_left:
            left_ankle_world = self.project_ankle_to_ground_with_homography(ankle_left)
            ankle_positions.append(left_ankle_world)
        if ankle_right:
            right_ankle_world = self.project_ankle_to_ground_with_homography(ankle_right)
            ankle_positions.append(right_ankle_world)

        # Weighted average - now with higher confidence in ankle positions due to improved correction
        positions = [(hip_ground_x, hip_ground_y)]
        weights = [0.5]  # Reduced hip weight since ankle correction is now more accurate

        for ankle_pos in ankle_positions:
            positions.append(ankle_pos)
            weights.append(0.25)  # Increased ankle weight due to better correction

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0]

        # Calculate final position
        final_x = sum(pos[0] * weight for pos, weight in zip(positions, normalized_weights))
        final_y = sum(pos[1] * weight for pos, weight in zip(positions, normalized_weights))

        if self.debug:
            print(f"Positions: {positions}")
            print(f"Weights: {normalized_weights}")
            print(f"Final: ({final_x:.2f}, {final_y:.2f})")
            print("===")

        return {
            'hip_world_X': final_x,
            'hip_world_Y': final_y,
            'left_ankle_world_X': left_ankle_world[0],
            'left_ankle_world_Y': left_ankle_world[1],
            'right_ankle_world_X': right_ankle_world[0],
            'right_ankle_world_Y': right_ankle_world[1],
            'estimated_hip_height': adaptive_hip_height
        }

    def assign_player_ids(self, frame_positions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Assign consistent player IDs using distance-based tracking."""
        assigned_positions = []
        used_ids = set()

        for current_pos in frame_positions:
            best_id = None
            min_distance = float('inf')

            # Find closest previous position
            for last_pos in self._last_positions:
                if last_pos['tracked_id'] in used_ids:
                    continue

                dx = current_pos['hip_world_X'] - last_pos['hip_world_X']
                dy = current_pos['hip_world_Y'] - last_pos['hip_world_Y']
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < self.MAX_TRACKING_DISTANCE and distance < min_distance:
                    min_distance = distance
                    best_id = last_pos['tracked_id']

            # Assign ID
            if best_id is not None:
                tracked_id = best_id
                used_ids.add(best_id)
            else:
                tracked_id = self._next_id
                self._next_id += 1

            position_entry = current_pos.copy()
            position_entry['tracked_id'] = tracked_id
            assigned_positions.append(position_entry)

        self._last_positions = assigned_positions.copy()
        return assigned_positions

    def merge_close_detections_in_frame(self, frame_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge multiple detections that are too close to be different players."""
        if len(frame_positions) <= 2:
            return frame_positions

        # Calculate distances between all pairs
        merged_positions = []
        used_indices = set()

        for i, pos1 in enumerate(frame_positions):
            if i in used_indices:
                continue

            # Find all positions close to this one
            close_positions = [pos1]
            close_indices = {i}

            for j, pos2 in enumerate(frame_positions):
                if j <= i or j in used_indices:
                    continue

                dx = pos1['hip_world_X'] - pos2['hip_world_X']
                dy = pos1['hip_world_Y'] - pos2['hip_world_Y']
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < self.MERGE_DISTANCE_THRESHOLD:
                    close_positions.append(pos2)
                    close_indices.add(j)

            # Merge close positions by averaging
            if len(close_positions) > 1:
                merged_pos = self.average_positions(close_positions)
                merged_positions.append(merged_pos)
                used_indices.update(close_indices)
            else:
                merged_positions.append(pos1)
                used_indices.add(i)

        return merged_positions

    def average_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average multiple position detections into one."""
        if len(positions) == 1:
            return positions[0]

        # Average all numerical values
        averaged = {}
        numerical_keys = ['hip_world_X', 'hip_world_Y', 'left_ankle_world_X',
                          'left_ankle_world_Y', 'right_ankle_world_X', 'right_ankle_world_Y',
                          'estimated_hip_height']

        for key in numerical_keys:
            values = [pos[key] for pos in positions if key in pos]
            if values:
                averaged[key] = sum(values) / len(values)
            else:
                averaged[key] = 0.0

        # Use the first tracked_id if available, otherwise assign a temporary one
        if 'tracked_id' in positions[0]:
            averaged['tracked_id'] = positions[0]['tracked_id']
        else:
            averaged['tracked_id'] = -1  # Temporary ID, will be assigned later

        if self.debug:
            print(f"Merged {len(positions)} close detections into one position")

        return averaged

    def analyze_player_trajectories(self) -> None:
        """Analyze all trajectories to identify the two main players."""
        # Group positions by tracked_id
        for position in self.player_positions:
            player_id = position['tracked_id']
            self._player_trajectories[player_id].append(position)

        # Filter out trajectories with too few frames
        valid_trajectories = {
            player_id: trajectory
            for player_id, trajectory in self._player_trajectories.items()
            if len(trajectory) >= self.MIN_FRAMES_FOR_PLAYER
        }

        if self.debug:
            print(f"Found {len(self._player_trajectories)} total trajectories")
            print(f"Valid trajectories (>={self.MIN_FRAMES_FOR_PLAYER} frames): {len(valid_trajectories)}")
            for player_id, trajectory in valid_trajectories.items():
                print(f"  Player {player_id}: {len(trajectory)} frames")

        # If we have exactly 2 valid trajectories, we're done
        if len(valid_trajectories) == 2:
            player_ids = list(valid_trajectories.keys())
            self._final_player_mapping = {player_ids[0]: 0, player_ids[1]: 1}
            print("Found exactly 2 players - no merging needed")
            return

        # If we have more than 2, we need to merge similar trajectories
        if len(valid_trajectories) > 2:
            print(f"Found {len(valid_trajectories)} players, merging to 2...")
            self.merge_similar_trajectories(valid_trajectories)
        else:
            # Less than 2 valid trajectories - use what we have
            player_ids = list(valid_trajectories.keys())
            for i, player_id in enumerate(player_ids):
                self._final_player_mapping[player_id] = i
            print(f"Only {len(valid_trajectories)} valid players found")

    def merge_similar_trajectories(self, valid_trajectories: Dict[int, List[Dict]]) -> None:
        """Merge trajectories that likely belong to the same player."""
        # Calculate average positions for each trajectory
        trajectory_centers = {}
        for player_id, trajectory in valid_trajectories.items():
            avg_x = sum(pos['hip_world_X'] for pos in trajectory) / len(trajectory)
            avg_y = sum(pos['hip_world_Y'] for pos in trajectory) / len(trajectory)
            trajectory_centers[player_id] = (avg_x, avg_y, len(trajectory))

        # Sort by trajectory length (longer trajectories are more reliable)
        sorted_trajectories = sorted(
            trajectory_centers.items(),
            key=lambda x: x[1][2],
            reverse=True
        )

        # Keep the two longest trajectories as main players
        main_players = sorted_trajectories[:2]
        self._final_player_mapping = {main_players[0][0]: 0, main_players[1][0]: 1}

        # Merge remaining trajectories with the closest main player
        for player_id, (avg_x, avg_y, length) in sorted_trajectories[2:]:
            min_distance = float('inf')
            closest_main_player = None

            for main_id, final_id in self._final_player_mapping.items():
                main_x, main_y, _ = trajectory_centers[main_id]
                distance = np.sqrt((avg_x - main_x)**2 + (avg_y - main_y)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_main_player = final_id

            # Map this trajectory to the closest main player
            self._final_player_mapping[player_id] = closest_main_player

            if self.debug:
                print(f"Merged player {player_id} ({length} frames) with main player {closest_main_player} "
                      f"(distance: {min_distance:.2f}m)")

    def apply_two_player_mapping(self) -> None:
        """Apply the final two-player mapping to all positions."""
        # Update all positions with the final player mapping
        for position in self.player_positions:
            original_id = position['tracked_id']
            if original_id in self._final_player_mapping:
                position['player_id'] = self._final_player_mapping[original_id]
            else:
                # This shouldn't happen with valid trajectories, but handle gracefully
                position['player_id'] = 0  # Default to player 0

        # Sort positions for cleaner output
        self.player_positions.sort(key=lambda x: (x['frame_index'], x['player_id']))

        # Print final statistics
        player_0_frames = len([p for p in self.player_positions if p['player_id'] == 0])
        player_1_frames = len([p for p in self.player_positions if p['player_id'] == 1])

        print(f"Final player distribution:")
        print(f"  Player 0: {player_0_frames} frames")
        print(f"  Player 1: {player_1_frames} frames")

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> List[Dict[str, Any]]:
        """Process all players in a single frame."""
        frame_positions = []

        for human_data in frame_data:
            joints = human_data.get('joints', [])
            position = self.calculate_player_position(joints, frame_index)

            if position:
                frame_positions.append(position)

        # First assign temporary tracking IDs
        frame_positions = self.assign_player_ids(frame_positions)

        # Then merge close detections (now that all positions have tracked_id)
        frame_positions = self.merge_close_detections_in_frame(frame_positions)

        return frame_positions

    def process_all_frames(self) -> None:
        """Process all frames with adaptive hip height."""
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

        # Apply two-player system
        print("Analyzing player trajectories...")
        self.analyze_player_trajectories()
        self.apply_two_player_mapping()

        # Print hip height statistics
        if self._hip_height_history:
            avg_height = sum(self._hip_height_history) / len(self._hip_height_history)
            min_height = min(self._hip_height_history)
            max_height = max(self._hip_height_history)
            print(f"Hip height stats: avg={avg_height:.3f}m, min={min_height:.3f}m, max={max_height:.3f}m")

    def detect_mirroring_issues(self) -> bool:
        """Detect potential mirroring issues in the coordinate system."""
        if len(self.player_positions) < 10:
            return False

        # Sample positions from different parts of the video
        sample_size = min(50, len(self.player_positions))
        step = len(self.player_positions) // sample_size
        sample_positions = [self.player_positions[i] for i in range(0, len(self.player_positions), step)]

        x_positions = [pos['hip_world_X'] for pos in sample_positions]
        y_positions = [pos['hip_world_Y'] for pos in sample_positions]

        # Check for clustering at extremes (potential mirroring)
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)

        # Check if most players are clustered on one extreme side
        left_heavy = sum(1 for x in x_positions if x < 1.0) > len(x_positions) * 0.8
        right_heavy = sum(1 for x in x_positions if x > 5.1) > len(x_positions) * 0.8
        top_heavy = sum(1 for y in y_positions if y < 1.0) > len(y_positions) * 0.8
        bottom_heavy = sum(1 for y in y_positions if y > 12.4) > len(y_positions) * 0.8

        suspicious = left_heavy or right_heavy or top_heavy or bottom_heavy

        if suspicious:
            print("⚠ Potential coordinate system issues detected:")
            print(f"  X range: {x_min:.2f} to {x_max:.2f} (court width: 0 to {self.COURT_WIDTH})")
            print(f"  Y range: {y_min:.2f} to {y_max:.2f} (court length: 0 to {self.COURT_LENGTH})")
            if left_heavy:
                print("  - Most players clustered on left side")
            if right_heavy:
                print("  - Most players clustered on right side")
            if top_heavy:
                print("  - Most players clustered at top")
            if bottom_heavy:
                print("  - Most players clustered at bottom")
            print("  Consider checking court point detection accuracy")

        return suspicious

    def save_results(self) -> None:
        """Save tracking results to JSON file."""
        frames_with_players = len(set(pos['frame_index'] for pos in self.player_positions))

        output_data = {
            'court_points': self.court_points,
            'all_court_points': self.pose_data.get('all_court_points', self.court_points),
            'video_info': self.video_info,
            'player_positions': self.player_positions,
            'tracking_method': "Adaptive hip height with ray-plane intersection for hips, pre-homography pixel offset correction for ankles, weighted averaging, two-player constraint, and corrected court coordinate mapping",
            'processing_info': {
                'total_positions': len(self.player_positions),
                'frames_with_players': frames_with_players,
                'default_hip_height_meters': self.DEFAULT_HIP_HEIGHT,
                'ankle_to_ground_offset_meters': self.ANKLE_TO_GROUND_OFFSET,
                'camera_height_meters': self.camera_height,
                'camera_position': self.camera_position.tolist() if self.camera_position is not None else None,
                'adaptive_hip_height': True,
                'ankle_method': 'pre_homography_pixel_offset',
                'hip_method': 'ray_plane_intersection_with_offset',
                'two_player_constraint': True,
                'coordinate_system_corrected': True,
                'merge_distance_threshold_meters': self.MERGE_DISTANCE_THRESHOLD,
                'min_frames_for_valid_player': self.MIN_FRAMES_FOR_PLAYER,
                'original_trajectories_found': len(self._player_trajectories),
                'final_player_mapping': self._final_player_mapping
            }
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {self.output_file}")
        print(f"Total positions: {len(self.player_positions)}")
        print(f"Frames with players: {frames_with_players}")

    def run(self) -> None:
        """Run the complete tracking pipeline."""
        print(f"Starting position tracking for: {self.video_name}")

        try:
            self.load_pose_data()
            self.solve_camera_pose()
            self.calculate_homography()
            self.validate_coordinate_system()  # Validate coordinate system
            self.process_all_frames()
            self.detect_mirroring_issues()  # Check for potential issues
            self.save_results()
            print("Position tracking completed successfully!")

        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python track_positions.py <video_file_path> [--debug]")
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