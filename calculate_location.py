#!/usr/bin/env python3
"""
Enhanced Badminton Player Position Tracking Script with Camera Calibration

Enhanced features using calibrated camera parameters:
- Precise 3D-to-2D projections using calibrated intrinsics and distortion
- Accurate distance measurements and depth estimation
- Improved perspective corrections with real camera pose
- Enhanced boundary validation with proper distortion correction
- Automatic undistortion of joint positions

Key features:
1. Ankle-prioritized positioning with hip fallback
2. Calibrated camera parameter integration
3. Enhanced trajectory analysis with 3D geometry
4. Distortion-corrected boundary validation
5. Precise depth-aware position calculation

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


class EnhancedBadmintonCourtTracker:
    """Enhanced tracker using calibrated camera parameters for improved accuracy."""

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
        """Initialize enhanced tracker."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.calibration_file = self.results_dir / f"{self.video_name}_calibration_complete.csv"
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
        self.rotation_vector = None
        self.translation_vector = None
        self.camera_height = None
        self.camera_position = None
        self.reprojection_error = None
        self.calibration_method = None

        self.player_positions = []

        # Enhanced tracking with dynamic windowing
        self._hip_height_history = deque(maxlen=self.MAX_WINDOW_SIZE)
        self._position_history = deque(maxlen=20)  # For motion analysis
        self._current_window_size = self.MIN_WINDOW_SIZE

    def load_calibration_data(self) -> None:
        """Load calibration data from the new detect_court.py format."""
        if not self.calibration_file.exists():
            # Fallback to old format
            old_pose_file = self.results_dir / "pose.json"
            if old_pose_file.exists():
                print("⚠️  Using fallback to pose.json (no calibration data)")
                return
            else:
                raise FileNotFoundError(f"No calibration data found: {self.calibration_file}")

        calibration_data = {
            'camera_matrix': None,
            'dist_coeffs': None,
            'rvec': None,
            'tvec': None,
            'court_points': {},
            'reprojection_error': None,
            'calibration_method': None
        }

        intrinsic_params = {}
        distortion_params = {}
        rotation_params = {}
        translation_params = {}

        with open(self.calibration_file, 'r') as file:
            csv_reader = csv.reader(file)
            in_court_points_section = False

            for row in csv_reader:
                if not row or row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    continue

                key = row[0].strip()
                value = row[1].strip() if len(row) > 1 else ''

                # Parse different sections
                if key == 'Point':  # Start of court points section
                    in_court_points_section = True
                    continue
                elif in_court_points_section and len(row) >= 3:
                    # Court points: Point, Image_X, Image_Y, World_X, World_Y, World_Z, Error_Pixels
                    point_name = row[0].strip()
                    try:
                        x_coord = float(row[1])
                        y_coord = float(row[2])
                        calibration_data['court_points'][point_name] = [x_coord, y_coord]
                    except (ValueError, IndexError):
                        continue
                elif key.startswith('intrinsic_'):
                    intrinsic_params[key] = float(value)
                elif key.startswith('distortion_'):
                    distortion_params[key] = float(value)
                elif key.startswith('rotation_'):
                    rotation_params[key] = float(value)
                elif key.startswith('translation_'):
                    translation_params[key] = float(value)
                elif key == 'reprojection_error_pixels':
                    calibration_data['reprojection_error'] = float(value)
                elif key == 'calibration_method':
                    calibration_data['calibration_method'] = value

        # Reconstruct camera matrix
        if all(param in intrinsic_params for param in ['intrinsic_fx', 'intrinsic_fy', 'intrinsic_cx', 'intrinsic_cy']):
            self.camera_matrix = np.array([
                [intrinsic_params['intrinsic_fx'], 0, intrinsic_params['intrinsic_cx']],
                [0, intrinsic_params['intrinsic_fy'], intrinsic_params['intrinsic_cy']],
                [0, 0, 1]
            ], dtype=np.float32)

        # Reconstruct distortion coefficients
        dist_keys = sorted([k for k in distortion_params.keys() if k.startswith('distortion_')])
        if dist_keys:
            self.dist_coeffs = np.array([distortion_params[k] for k in dist_keys], dtype=np.float32)

        # Reconstruct rotation vector
        if all(param in rotation_params for param in ['rotation_x', 'rotation_y', 'rotation_z']):
            self.rotation_vector = np.array([
                rotation_params['rotation_x'],
                rotation_params['rotation_y'],
                rotation_params['rotation_z']
            ], dtype=np.float32)

        # Reconstruct translation vector
        if all(param in translation_params for param in ['translation_x', 'translation_y', 'translation_z']):
            self.translation_vector = np.array([
                translation_params['translation_x'],
                translation_params['translation_y'],
                translation_params['translation_z']
            ], dtype=np.float32)

        # Set additional parameters
        self.court_points = calibration_data['court_points']
        self.reprojection_error = calibration_data['reprojection_error']
        self.calibration_method = calibration_data['calibration_method']

        if self.translation_vector is not None:
            self.camera_height = abs(float(self.translation_vector[2]))

        # Calculate camera position
        if self.rotation_vector is not None and self.translation_vector is not None:
            rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
            camera_position = -rotation_matrix.T @ self.translation_vector
            self.camera_position = camera_position.flatten()

        print(f"✓ Loaded calibrated camera parameters:")
        print(f"  - Camera matrix: {'✓' if self.camera_matrix is not None else '✗'}")
        print(f"  - Distortion coefficients: {'✓' if self.dist_coeffs is not None else '✗'}")
        print(f"  - Camera pose: {'✓' if self.rotation_vector is not None else '✗'}")
        if self.reprojection_error is not None:
            print(f"  - Reprojection error: {self.reprojection_error:.2f} pixels")
            if self.reprojection_error > 20.0:
                print(f"    ⚠️  High reprojection error - will prefer homography method")
        print(f"  - Calibration method: {self.calibration_method}")
        print(f"  - Camera height: {self.camera_height:.2f}m")

    def load_pose_data(self) -> None:
        """Load pose detection data from JSON file."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.video_info = data.get('video_info', {})

        # If no calibration data was loaded, get court points from pose data
        if not self.court_points:
            self.court_points = data.get('court_points', {})

        print(f"Loaded pose data with {len(data.get('pose_data', []))} pose detections")

    def undistort_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Undistort a pixel point using calibrated distortion coefficients."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return point  # No distortion correction available

        # Convert to numpy array format expected by OpenCV
        point_array = np.array([[point]], dtype=np.float32)

        try:
            # Undistort the point
            undistorted = cv2.undistortPoints(
                point_array,
                self.camera_matrix,
                self.dist_coeffs,
                P=self.camera_matrix
            )
            return float(undistorted[0][0][0]), float(undistorted[0][0][1])
        except:
            return point  # Fallback to original if undistortion fails

    def project_3d_to_pixel(self, world_point_3d: np.ndarray) -> Tuple[float, float]:
        """Project a 3D world point to pixel coordinates using calibrated parameters."""
        if (self.camera_matrix is None or self.rotation_vector is None or
                self.translation_vector is None):
            raise ValueError("Calibrated camera parameters required for 3D projection")

        # Project 3D point to image coordinates
        projected_points, _ = cv2.projectPoints(
            world_point_3d.reshape(1, 1, 3),
            self.rotation_vector,
            self.translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )

        return float(projected_points[0][0][0]), float(projected_points[0][0][1])

    def calculate_3d_ray_from_pixel(self, pixel_point: Tuple[float, float]) -> np.ndarray:
        """Calculate 3D ray direction from pixel point using calibrated parameters."""
        if self.camera_matrix is None:
            raise ValueError("Camera matrix required for ray calculation")

        # Undistort the pixel point first
        undistorted_point = self.undistort_point(pixel_point)

        # Convert to normalized camera coordinates
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        x_norm = (undistorted_point[0] - cx) / fx
        y_norm = (undistorted_point[1] - cy) / fy

        # Ray direction in camera coordinates
        ray_camera = np.array([x_norm, y_norm, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # Transform to world coordinates
        if self.rotation_vector is not None:
            rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
            ray_world = rotation_matrix.T @ ray_camera
            return ray_world / np.linalg.norm(ray_world)
        else:
            return ray_camera

    def intersect_ray_with_ground_plane(self, pixel_point: Tuple[float, float],
                                        height_above_ground: float = 0.0) -> Tuple[float, float]:
        """Intersect ray from pixel with ground plane at specified height."""
        if (self.camera_position is None or self.rotation_vector is None or
                self.camera_matrix is None):
            # Fallback to homography if calibration not available
            return self.transform_point_to_world_homography(pixel_point)

        try:
            # Get ray direction
            ray_direction = self.calculate_3d_ray_from_pixel(pixel_point)

            # Camera position
            camera_pos = self.camera_position

            # Intersect with plane at z = height_above_ground
            plane_z = height_above_ground

            # Calculate intersection parameter t
            # camera_pos + t * ray_direction = point on plane
            # (camera_pos + t * ray_direction)[2] = plane_z
            t = (plane_z - camera_pos[2]) / ray_direction[2]

            if t <= 0:
                # Ray doesn't intersect ground plane in front of camera
                return self.transform_point_to_world_homography(pixel_point)

            # Calculate intersection point
            intersection = camera_pos + t * ray_direction

            # Boundary validation
            world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, intersection[0]))
            world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, intersection[1]))

            return float(world_x), float(world_y)

        except Exception as e:
            if self.debug:
                print(f"Ray intersection failed: {e}, falling back to homography")
            return self.transform_point_to_world_homography(pixel_point)

    def calculate_precise_distance_to_point(self, pixel_point: Tuple[float, float]) -> float:
        """Calculate precise distance to a point using calibrated 3D geometry."""
        if self.camera_position is None:
            # Fallback to estimation
            return self.estimate_distance_to_point_homography(pixel_point)

        try:
            # Get ground position using ray intersection
            world_x, world_y = self.intersect_ray_with_ground_plane(pixel_point, 0.0)

            # Calculate 3D distance to ground point
            ground_point = np.array([world_x, world_y, 0.0])
            distance = np.linalg.norm(self.camera_position - ground_point)

            return max(distance, 1.0)  # Minimum 1m distance

        except Exception:
            return self.estimate_distance_to_point_homography(pixel_point)

    def validate_coordinate_system_with_calibration(self) -> None:
        """Enhanced coordinate system validation using calibrated parameters."""
        print("\n=== Enhanced Coordinate System Validation ===")

        corners = ['P1', 'P2', 'P3', 'P4']
        expected_world = [(0, 0), (0, self.COURT_LENGTH), (self.COURT_WIDTH, self.COURT_LENGTH), (self.COURT_WIDTH, 0)]
        corner_names = ['Top-Left', 'Bottom-Left', 'Bottom-Right', 'Top-Right']

        max_error_homography = 0.0
        max_error_3d = 0.0

        for corner, (exp_x, exp_y), name in zip(corners, expected_world, corner_names):
            if corner not in self.court_points:
                continue

            # Test 1: Homography-based transformation
            image_point = np.array([[self.court_points[corner]]], dtype=np.float32)
            world_point_homo = cv2.perspectiveTransform(image_point, self.homography_matrix)
            actual_x_homo, actual_y_homo = world_point_homo[0][0]
            error_homo = np.sqrt((actual_x_homo - exp_x)**2 + (actual_y_homo - exp_y)**2)
            max_error_homography = max(max_error_homography, error_homo)

            # Test 2: 3D ray intersection (if calibration available)
            if self.camera_matrix is not None:
                actual_x_3d, actual_y_3d = self.intersect_ray_with_ground_plane(self.court_points[corner], 0.0)
                error_3d = np.sqrt((actual_x_3d - exp_x)**2 + (actual_y_3d - exp_y)**2)
                max_error_3d = max(max_error_3d, error_3d)

                print(f"  {corner} ({name}):")
                print(f"    Expected: ({exp_x:.1f}, {exp_y:.1f})")
                print(f"    Homography: ({actual_x_homo:.1f}, {actual_y_homo:.1f}), Error: {error_homo:.3f}m")
                print(f"    3D Ray: ({actual_x_3d:.1f}, {actual_y_3d:.1f}), Error: {error_3d:.3f}m")
            else:
                print(f"  {corner} ({name}): Expected ({exp_x:.1f}, {exp_y:.1f}), Homography ({actual_x_homo:.1f}, {actual_y_homo:.1f}), Error: {error_homo:.3f}m")

        print(f"Maximum homography error: {max_error_homography:.3f} meters")
        if self.camera_matrix is not None:
            print(f"Maximum 3D ray error: {max_error_3d:.3f} meters")
            print(f"3D method improvement: {((max_error_homography - max_error_3d) / max_error_homography * 100):.1f}%")

        if max_error_homography < 0.15:
            print("✓ Coordinate system validation passed")
        else:
            print("⚠ Coordinate system has errors - check court point detection")

        print("============================================\n")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        # Extract corner points (ensure P1-P4 are available)
        required_corners = ['P1', 'P2', 'P3', 'P4']
        if not all(corner in self.court_points for corner in required_corners):
            raise ValueError(f"Missing required court corners: {required_corners}")

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

    def transform_point_to_world_homography(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform pixel point to world coordinates using homography (fallback method)."""
        point = np.array([[pixel_point]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def estimate_distance_to_point_homography(self, pixel_point: Tuple[float, float]) -> float:
        """Estimate distance from camera to a point using homography (fallback method)."""
        try:
            world_x, world_y = self.transform_point_to_world_homography(pixel_point)
            if self.camera_position is not None:
                distance = np.sqrt(
                    (world_x - self.camera_position[0])**2 +
                    (world_y - self.camera_position[1])**2
                )
            else:
                # Rough estimation if no camera position
                distance = 5.0
            return max(distance, 1.0)  # Minimum 1m distance
        except:
            return 5.0  # Fallback distance

    def calculate_enhanced_ankle_projection(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Enhanced ankle projection with intelligent fallback strategy."""
        # Check calibration quality first
        if (self.camera_matrix is not None and
                self.reprojection_error is not None and
                self.reprojection_error < 15.0):  # Only use 3D if good calibration
            try:
                result = self.intersect_ray_with_ground_plane(ankle_pixel, -self.ANKLE_TO_GROUND_OFFSET)
                # Validate result is reasonable
                if (-2.0 <= result[0] <= 8.0 and -2.0 <= result[1] <= 16.0):
                    return result
            except:
                pass

        # Fallback to homography-based method
        return self.project_ankle_to_ground_with_homography(ankle_pixel)

    def calculate_enhanced_hip_projection(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Enhanced hip projection with intelligent fallback strategy."""
        # Check calibration quality first
        if (self.camera_matrix is not None and
                self.reprojection_error is not None and
                self.reprojection_error < 15.0):  # Only use 3D if good calibration
            try:
                result = self.intersect_ray_with_ground_plane(hip_pixel, -hip_height)
                # Validate result is reasonable
                if (-2.0 <= result[0] <= 8.0 and -2.0 <= result[1] <= 16.0):
                    return result
            except:
                pass

        # Fallback to homography-based method
        return self.project_hip_to_ground(hip_pixel, hip_height)

    def calculate_ankle_pixel_offset(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate pixel offset for ankle to ground correction (fallback method)."""
        try:
            distance_to_player = self.estimate_distance_to_point_homography(ankle_pixel)
            if self.camera_height is not None:
                vertical_angle = np.arctan(self.camera_height / distance_to_player)
            else:
                vertical_angle = np.radians(30)  # Default angle

            # Use camera matrix if available, otherwise estimate
            if self.camera_matrix is not None:
                focal_length = self.camera_matrix[1, 1]
            else:
                focal_length = 1000.0  # Default focal length

            pixel_offset_y = (self.ANKLE_TO_GROUND_OFFSET * focal_length) / distance_to_player
            perspective_factor = 1.0 / max(np.cos(vertical_angle), 0.1)
            corrected_offset_y = pixel_offset_y * perspective_factor

            return (0.0, corrected_offset_y)
        except Exception:
            return (0.0, 0.0)

    def project_ankle_to_ground_with_homography(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Project ankle to ground using homography (fallback method)."""
        offset_x, offset_y = self.calculate_ankle_pixel_offset(ankle_pixel)
        corrected_pixel = (ankle_pixel[0] + offset_x, ankle_pixel[1] + offset_y)

        corrected_world_x, corrected_world_y = self.transform_point_to_world_homography(corrected_pixel)

        # Boundary validation
        corrected_world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, corrected_world_x))
        corrected_world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, corrected_world_y))

        return float(corrected_world_x), float(corrected_world_y)

    def calculate_hip_pixel_offset(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Calculate pixel offset for hip to ground correction (fallback method)."""
        try:
            distance_to_player = self.estimate_distance_to_point_homography(hip_pixel)
            if self.camera_height is not None:
                vertical_angle = np.arctan(self.camera_height / distance_to_player)
            else:
                vertical_angle = np.radians(30)  # Default angle

            # Use camera matrix if available, otherwise estimate
            if self.camera_matrix is not None:
                focal_length = self.camera_matrix[1, 1]
            else:
                focal_length = 1000.0  # Default focal length

            pixel_offset_y = (hip_height * focal_length) / distance_to_player
            perspective_factor = 1.0 / max(np.cos(vertical_angle), 0.1)
            corrected_offset_y = pixel_offset_y * perspective_factor

            return (0.0, corrected_offset_y)
        except Exception:
            return (0.0, 0.0)

    def project_hip_to_ground(self, hip_pixel: Tuple[float, float], hip_height: float) -> Tuple[float, float]:
        """Project hip to ground using homography (fallback method)."""
        offset_x, offset_y = self.calculate_hip_pixel_offset(hip_pixel, hip_height)
        corrected_pixel = (hip_pixel[0] + offset_x, hip_pixel[1] + offset_y)

        corrected_world_x, corrected_world_y = self.transform_point_to_world_homography(corrected_pixel)

        # Boundary validation
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

        # Get reference points - apply undistortion if calibration available
        if self.camera_matrix is not None:
            head = self.undistort_point(head)
            if ankle_left:
                ankle_left = self.undistort_point(ankle_left)
            if ankle_right:
                ankle_right = self.undistort_point(ankle_right)
            if hip_left:
                hip_left = self.undistort_point(hip_left)
            if hip_right:
                hip_right = self.undistort_point(hip_right)

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
        """Enhanced hip height estimation using calibrated camera geometry."""
        if not ankle_pixels:
            return None

        try:
            avg_ankle_x = sum(pos[0] for pos in ankle_pixels) / len(ankle_pixels)
            avg_ankle_y = sum(pos[1] for pos in ankle_pixels) / len(ankle_pixels)

            # Apply undistortion if calibration available
            if self.camera_matrix is not None:
                hip_pixel = self.undistort_point(hip_pixel)
                avg_ankle_x, avg_ankle_y = self.undistort_point((avg_ankle_x, avg_ankle_y))

            pixel_height_diff = abs(hip_pixel[1] - avg_ankle_y)

            # Use enhanced distance calculation with calibration
            camera_to_player_distance = self.calculate_precise_distance_to_point((avg_ankle_x, avg_ankle_y))

            # Use calibrated focal length if available
            if self.camera_matrix is not None:
                focal_length = self.camera_matrix[0, 0]
            else:
                focal_length = 1000.0  # Fallback

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

        # Method 2: Enhanced geometry-based estimation with calibration
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

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence is sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return joint['x'], joint['y']
        return None

    def calculate_player_position(self, joints: List[Dict], frame_index: int) -> Optional[Dict[str, float]]:
        """Calculate world position using enhanced calibrated method."""
        # Extract joints
        hip_left = self.extract_joint_position(joints, self.HIP_LEFT)
        hip_right = self.extract_joint_position(joints, self.HIP_RIGHT)
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        # Enhanced ankle position calculation with calibration
        ankle_positions = []
        left_ankle_world = (0.0, 0.0)
        right_ankle_world = (0.0, 0.0)

        if ankle_left:
            left_ankle_world = self.calculate_enhanced_ankle_projection(ankle_left)
            ankle_positions.append(left_ankle_world)
        if ankle_right:
            right_ankle_world = self.calculate_enhanced_ankle_projection(ankle_right)
            ankle_positions.append(right_ankle_world)

        # Enhanced hip position calculation
        hip_ground_x, hip_ground_y = 0.0, 0.0
        if hip_left or hip_right:
            if hip_left and hip_right:
                hip_center = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
            else:
                hip_center = hip_left if hip_left else hip_right

            adaptive_hip_height = self.get_adaptive_hip_height(joints, frame_index)
            hip_ground_x, hip_ground_y = self.calculate_enhanced_hip_projection(hip_center, adaptive_hip_height)

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
            'estimated_hip_height': self.get_adaptive_hip_height(joints, frame_index) if (hip_left or hip_right) else self.DEFAULT_HIP_HEIGHT,
            'calibration_used': self.camera_matrix is not None
        }

    def calculate_player_center_y(self, position: Dict[str, float]) -> float:
        """Calculate the center Y coordinate from all valid real-world points for a player."""
        y_values = []

        # Collect all valid Y coordinates
        if position['hip_world_Y'] != 0.0:
            y_values.append(position['hip_world_Y'])

        if position['left_ankle_world_Y'] != 0.0:
            y_values.append(position['left_ankle_world_Y'])

        if position['right_ankle_world_Y'] != 0.0:
            y_values.append(position['right_ankle_world_Y'])

        # Return center of all valid points, or hip position as fallback
        if y_values:
            return sum(y_values) / len(y_values)
        else:
            return position['hip_world_Y']  # Fallback to hip if no other points

    def assign_player_ids_by_y_coordinate(self, frame_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign player IDs based on Y-coordinate center of all real-world points."""
        if len(frame_positions) == 0:
            return []
        elif len(frame_positions) == 1:
            position_entry = frame_positions[0].copy()
            center_y = self.calculate_player_center_y(position_entry)

            # Assign based on court position relative to center line (6.7m)
            # Lower Y values = closer to near baseline = Player 0
            # Higher Y values = closer to far baseline = Player 1
            if center_y < 6.7:
                position_entry['player_id'] = 0  # Closer to near baseline (lower Y)
            else:
                position_entry['player_id'] = 1  # Closer to far baseline (higher Y)

            position_entry['center_y_coordinate'] = center_y
            position_entry['y_assignment_method'] = 'single_player_court_position'

            return [position_entry]
        else:
            # Multiple players - consistent assignment based on center Y position
            positions_with_center_y = []
            for pos in frame_positions:
                center_y = self.calculate_player_center_y(pos)
                positions_with_center_y.append((pos, center_y))

            # Sort by center Y position: lowest Y first
            positions_with_center_y.sort(key=lambda x: x[1])

            # Assign IDs consistently: lowest center Y = Player 0, highest center Y = Player 1
            result = []
            for i, (pos, center_y) in enumerate(positions_with_center_y[:2]):  # Max 2 players
                position_entry = pos.copy()
                position_entry['player_id'] = i  # i=0 for lowest Y, i=1 for highest Y
                position_entry['center_y_coordinate'] = center_y
                position_entry['y_assignment_method'] = 'multi_player_center_y_sorting'
                result.append(position_entry)

            return result

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> List[Dict[str, Any]]:
        """Process all players in a single frame with Y-coordinate based ID assignment."""
        frame_positions = []

        for human_data in frame_data:
            joints = human_data.get('joints', [])
            position = self.calculate_player_position(joints, frame_index)

            if position:
                # Add frame and detection metadata
                position['frame_index'] = frame_index
                position['human_index'] = human_data.get('human_index', 0)
                position['detection_confidence'] = human_data.get('valid_ankle_knee_joints', 0)
                frame_positions.append(position)

        if self.debug and len(frame_positions) > 2:
            print(f"Frame {frame_index}: {len(frame_positions)} valid detections before ID assignment")

        # Assign player IDs based on Y-coordinate center
        frame_positions = self.assign_player_ids_by_y_coordinate(frame_positions)

        if self.debug:
            for pos in frame_positions:
                center_y = pos.get('center_y_coordinate', 0)
                player_id = pos.get('player_id', -1)
                method = pos.get('y_assignment_method', 'unknown')
                print(f"  Player {player_id}: Center Y = {center_y:.2f}m (method: {method})")

        return frame_positions

    def process_all_frames(self) -> None:
        """Process all frames with Y-coordinate based ID assignment."""
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
            self.player_positions.extend(positions)

        print(f"Extracted {len(self.player_positions)} player positions")

        # Statistics
        if self._hip_height_history:
            heights = list(self._hip_height_history)
            avg_height = np.mean(heights)
            std_height = np.std(heights)
            print(f"Hip height stats: avg={avg_height:.3f}m ±{std_height:.3f}m")

        # Report calibration usage and method preference
        calibrated_positions = len([p for p in self.player_positions if p.get('calibration_used', False)])
        homography_fallback_used = calibrated_positions < len(self.player_positions)

        print(f"Positions using calibrated parameters: {calibrated_positions}/{len(self.player_positions)}")
        if homography_fallback_used and self.reprojection_error and self.reprojection_error > 15.0:
            print(f"Note: High calibration error ({self.reprojection_error:.1f}px) caused fallback to homography method")

        # Report final player ID distribution
        player_0_frames = len([p for p in self.player_positions if p.get('player_id') == 0])
        player_1_frames = len([p for p in self.player_positions if p.get('player_id') == 1])

        print(f"Final player distribution:")
        print(f"  Player 0 (lower center Y): {player_0_frames} frames")
        print(f"  Player 1 (higher center Y): {player_1_frames} frames")

        # Report Y-coordinate assignment statistics
        single_player_assignments = len([p for p in self.player_positions
                                         if p.get('y_assignment_method') == 'single_player_court_position'])
        multi_player_assignments = len([p for p in self.player_positions
                                        if p.get('y_assignment_method') == 'multi_player_center_y_sorting'])

        print(f"Y-coordinate assignment methods:")
        print(f"  Single player (court position): {single_player_assignments} frames")
        print(f"  Multi-player (center Y sorting): {multi_player_assignments} frames")

    def detect_potential_issues(self) -> None:
        """Enhanced issue detection with calibration awareness."""
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

        # Calculate precision improvement with calibration
        calibrated_count = len([p for p in sample_positions if p.get('calibration_used', False)])
        calibration_ratio = calibrated_count / len(sample_positions)

        print(f"\n=== Enhanced Tracking Quality Assessment ===")
        print(f"Positions analyzed: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds_ratio:.1%}")
        print(f"Calibration usage: {calibration_ratio:.1%}")

        # Y-coordinate assignment analysis
        center_y_values = [pos.get('center_y_coordinate', 0) for pos in sample_positions if pos.get('center_y_coordinate')]
        if center_y_values:
            avg_center_y = np.mean(center_y_values)
            std_center_y = np.std(center_y_values)
            print(f"Center Y coordinates: avg={avg_center_y:.2f}m ±{std_center_y:.2f}m")

        # Player ID distribution in sample
        player_0_sample = len([p for p in sample_positions if p.get('player_id') == 0])
        player_1_sample = len([p for p in sample_positions if p.get('player_id') == 1])
        print(f"Sample player distribution: Player 0: {player_0_sample}, Player 1: {player_1_sample}")

        if self.reprojection_error is not None:
            print(f"Camera calibration error: {self.reprojection_error:.2f} pixels")
        if self.calibration_method is not None:
            print(f"Calibration method: {self.calibration_method}")

        if out_of_bounds_ratio > 0.1:
            print("⚠ High out-of-bounds ratio detected - check calibration")
        elif calibration_ratio > 0.8:
            print("✓ Enhanced tracking quality with calibrated parameters")
        else:
            print("✓ Tracking quality appears acceptable")
        print("==========================================\n")

    def save_results(self) -> None:
        """Save enhanced tracking results to JSON file."""
        frames_with_players = len(set(pos['frame_index'] for pos in self.player_positions))

        output_data = {
            'court_points': self.court_points,
            'all_court_points': self.pose_data.get('all_court_points', self.court_points),
            'video_info': self.video_info,
            'player_positions': self.player_positions,
            'tracking_method': "Enhanced ankle-prioritized positioning with calibrated camera parameters and Y-coordinate based player ID assignment",
            'camera_calibration': {
                'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                'distortion_coefficients': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                'rotation_vector': self.rotation_vector.tolist() if self.rotation_vector is not None else None,
                'translation_vector': self.translation_vector.tolist() if self.translation_vector is not None else None,
                'camera_height': self.camera_height,
                'camera_position': self.camera_position.tolist() if self.camera_position is not None else None,
                'reprojection_error': self.reprojection_error,
                'calibration_method': self.calibration_method
            },
            'processing_info': {
                'total_positions': len(self.player_positions),
                'frames_with_players': frames_with_players,
                'ankle_to_ground_offset_meters': self.ANKLE_TO_GROUND_OFFSET,
                'camera_height_meters': self.camera_height,
                'camera_position': self.camera_position.tolist() if self.camera_position is not None else None,
                'court_points_from_calibration': True,
                'id_assignment_method': 'Y-coordinate based using center of all real-world points',
                'enhanced_features': {
                    'calibrated_camera_parameters': self.camera_matrix is not None,
                    'distortion_correction': self.dist_coeffs is not None,
                    '3d_ray_intersection': self.camera_matrix is not None,
                    'precise_distance_calculation': True,
                    'ankle_prioritized_weighting': True,
                    'dynamic_frame_windowing': True,
                    'adaptive_hip_height': True,
                    'motion_based_windowing': True,
                    'boundary_validation': True,
                    'enhanced_perspective_correction': True,
                    'y_coordinate_center_calculation': True,
                    'automatic_player_id_assignment': True
                },
                'calibration_benefits': {
                    'undistorted_joint_positions': self.camera_matrix is not None,
                    'accurate_3d_projections': self.rotation_vector is not None,
                    'precise_distance_measurements': True,
                    'improved_boundary_validation': True,
                    'enhanced_perspective_corrections': True
                },
                'weighting_scheme': {
                    'ankle_weight': 0.4,
                    'hip_weight': 0.2,
                    'note': 'Per ankle joint when detected'
                },
                'window_size_range': [self.MIN_WINDOW_SIZE, self.MAX_WINDOW_SIZE],
                'projection_methods': {
                    'primary': '3D ray intersection with ground plane (if calibrated)',
                    'fallback': 'Homography-based transformation with pixel offset correction'
                },
                'y_coordinate_assignment': {
                    'calculation_method': 'Center of all valid real-world points (hip_world_Y, left_ankle_world_Y, right_ankle_world_Y)',
                    'single_player_logic': 'Player 0 if center_y < 6.7m, Player 1 if center_y >= 6.7m',
                    'multi_player_logic': 'Sort by center_y: lowest = Player 0, highest = Player 1',
                    'court_center_line': '6.7m (half of 13.4m court length)',
                    'assignment_consistency': 'Frame-by-frame based on real-world coordinates'
                }
            }
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Enhanced results saved to: {self.output_file}")
        print(f"Total positions: {len(self.player_positions)}")
        print(f"Frames with players: {frames_with_players}")

    def run(self) -> None:
        """Run the enhanced tracking pipeline with calibrated parameters."""
        print(f"Starting enhanced position tracking for: {self.video_name}")

        try:
            self.load_calibration_data()
            self.load_pose_data()
            self.calculate_homography()
            self.validate_coordinate_system_with_calibration()
            self.process_all_frames()
            self.detect_potential_issues()
            self.save_results()
            print("✓ Enhanced position tracking completed successfully!")

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

    tracker = EnhancedBadmintonCourtTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()