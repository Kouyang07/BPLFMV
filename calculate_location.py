#!/usr/bin/env python3
"""
Simplified Badminton Player Position Tracking Script - Ankle-Only Approach

Focused on ankle-based positioning for maximum accuracy and simplicity:
- Precise ankle-to-ground projection using calibrated camera parameters
- Clean fallback to homography when calibration unavailable
- Simplified player ID assignment based on court position
- Streamlined output with only essential data

Key features:
1. Ankle-only positioning (no complex hip calculations)
2. Calibrated camera parameter integration when available
3. Clean boundary validation
4. Simple and maintainable codebase

Usage: python ankle_tracker.py <video_file_path> [--debug]
"""

import sys
import os
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class AnkleBasedCourtTracker:
    """Simplified tracker focusing on ankle positions for maximum accuracy."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Pose joint indices
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    ANKLE_TO_GROUND_OFFSET = 0.04  # 4cm from ankle joint to ground contact

    def __init__(self, video_path: str, debug: bool = False):
        """Initialize ankle-based tracker."""
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
        self.calibration_available = False

        self.player_positions = []

    def load_calibration_data(self) -> None:
        """Load calibration data if available."""
        if not self.calibration_file.exists():
            print("⚠️  No calibration data found - will use homography method")
            return

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

                if key == 'Point':
                    in_court_points_section = True
                    continue
                elif in_court_points_section and len(row) >= 3:
                    point_name = row[0].strip()
                    try:
                        x_coord = float(row[1])
                        y_coord = float(row[2])
                        if not hasattr(self, 'court_points') or self.court_points is None:
                            self.court_points = {}
                        self.court_points[point_name] = [x_coord, y_coord]
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
                    self.reprojection_error = float(value)

        # Reconstruct camera parameters
        if all(param in intrinsic_params for param in ['intrinsic_fx', 'intrinsic_fy', 'intrinsic_cx', 'intrinsic_cy']):
            self.camera_matrix = np.array([
                [intrinsic_params['intrinsic_fx'], 0, intrinsic_params['intrinsic_cx']],
                [0, intrinsic_params['intrinsic_fy'], intrinsic_params['intrinsic_cy']],
                [0, 0, 1]
            ], dtype=np.float32)

        dist_keys = sorted([k for k in distortion_params.keys() if k.startswith('distortion_')])
        if dist_keys:
            self.dist_coeffs = np.array([distortion_params[k] for k in dist_keys], dtype=np.float32)

        if all(param in rotation_params for param in ['rotation_x', 'rotation_y', 'rotation_z']):
            self.rotation_vector = np.array([
                rotation_params['rotation_x'],
                rotation_params['rotation_y'],
                rotation_params['rotation_z']
            ], dtype=np.float32)

        if all(param in translation_params for param in ['translation_x', 'translation_y', 'translation_z']):
            self.translation_vector = np.array([
                translation_params['translation_x'],
                translation_params['translation_y'],
                translation_params['translation_z']
            ], dtype=np.float32)

        if self.translation_vector is not None:
            self.camera_height = abs(float(self.translation_vector[2]))

        if self.rotation_vector is not None and self.translation_vector is not None:
            rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
            camera_position = -rotation_matrix.T @ self.translation_vector
            self.camera_position = camera_position.flatten()

        # Check if calibration is good quality
        self.calibration_available = (
                self.camera_matrix is not None and
                self.rotation_vector is not None and
                self.translation_vector is not None and
                (self.reprojection_error is None or self.reprojection_error < 20.0)
        )

        if self.calibration_available:
            print(f"✓ Loaded high-quality calibration (error: {self.reprojection_error:.1f}px)")
        else:
            print(f"⚠️  Calibration quality poor (error: {self.reprojection_error:.1f}px) - using homography")

    def load_pose_data(self) -> None:
        """Load pose detection data."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.video_info = data.get('video_info', {})

        if not self.court_points:
            self.court_points = data.get('court_points', {})

        print(f"Loaded pose data with {len(data.get('pose_data', []))} detections")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        required_corners = ['P1', 'P2', 'P3', 'P4']
        if not all(corner in self.court_points for corner in required_corners):
            raise ValueError(f"Missing required court corners: {required_corners}")

        image_points = np.array([
            self.court_points['P1'],  # Top-left
            self.court_points['P2'],  # Bottom-left
            self.court_points['P3'],  # Bottom-right
            self.court_points['P4']   # Top-right
        ], dtype=np.float32)

        world_points = np.array([
            [0, 0],                    # P1: Top-left
            [0, self.COURT_LENGTH],    # P2: Bottom-left
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Bottom-right
            [self.COURT_WIDTH, 0]      # P4: Top-right
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
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
        except:
            return point

    def intersect_ray_with_ground(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate ground position using 3D ray intersection."""
        if not self.calibration_available:
            raise ValueError("Calibration required for ray intersection")

        # Undistort point
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
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
        ray_world = rotation_matrix.T @ ray_camera

        # Intersect with ground plane (z = -ANKLE_TO_GROUND_OFFSET)
        plane_z = -self.ANKLE_TO_GROUND_OFFSET
        t = (plane_z - self.camera_position[2]) / ray_world[2]

        if t <= 0:
            raise ValueError("Ray doesn't intersect ground plane")

        intersection = self.camera_position + t * ray_world
        return float(intersection[0]), float(intersection[1])

    def transform_point_homography(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform pixel point using homography with ankle offset correction."""
        # Apply simple pixel offset for ankle-to-ground correction
        offset_y = 10.0  # Simple fixed offset in pixels
        corrected_pixel = (pixel_point[0], pixel_point[1] + offset_y)

        point = np.array([[corrected_pixel]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def calculate_ankle_position(self, ankle_pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate ankle ground position using best available method."""
        try:
            if self.calibration_available:
                world_x, world_y = self.intersect_ray_with_ground(ankle_pixel)
            else:
                world_x, world_y = self.transform_point_homography(ankle_pixel)

            # Boundary validation
            world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, world_x))
            world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, world_y))

            return world_x, world_y

        except Exception as e:
            if self.debug:
                print(f"Position calculation failed: {e}")
            # Fallback to homography
            return self.transform_point_homography(ankle_pixel)

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence is sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return joint['x'], joint['y']
        return None

    def calculate_player_position(self, joints: List[Dict], frame_index: int) -> Optional[Dict[str, Any]]:
        """Calculate player position from ankle joints."""
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        # Need at least one ankle
        if not ankle_left and not ankle_right:
            return None

        ankle_positions = []
        if ankle_left:
            left_pos = self.calculate_ankle_position(ankle_left)
            ankle_positions.append(left_pos)
        if ankle_right:
            right_pos = self.calculate_ankle_position(ankle_right)
            ankle_positions.append(right_pos)

        # Calculate final position as average of available ankles
        final_x = sum(pos[0] for pos in ankle_positions) / len(ankle_positions)
        final_y = sum(pos[1] for pos in ankle_positions) / len(ankle_positions)

        return {
            'frame_index': frame_index,
            'x': final_x,
            'y': final_y,
            'ankle_count': len(ankle_positions),
            'method': 'calibrated_3d' if self.calibration_available else 'homography'
        }

    def assign_player_ids(self, frame_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign player IDs based on court position."""
        if len(frame_positions) == 0:
            return []
        elif len(frame_positions) == 1:
            position = frame_positions[0].copy()
            # Single player: assign based on court half
            position['player_id'] = 0 if position['y'] < 6.7 else 1
            return [position]
        else:
            # Multiple players: sort by Y position
            sorted_positions = sorted(frame_positions, key=lambda p: p['y'])
            for i, position in enumerate(sorted_positions[:2]):  # Max 2 players
                position['player_id'] = i
            return sorted_positions[:2]

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> List[Dict[str, Any]]:
        """Process all players in a single frame."""
        frame_positions = []

        for human_data in frame_data:
            joints = human_data.get('joints', [])
            position = self.calculate_player_position(joints, frame_index)
            if position:
                frame_positions.append(position)

        return self.assign_player_ids(frame_positions)

    def process_all_frames(self) -> None:
        """Process all frames."""
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

        # Report method usage
        calibrated_count = len([p for p in self.player_positions if p['method'] == 'calibrated_3d'])
        homography_count = len([p for p in self.player_positions if p['method'] == 'homography'])
        print(f"Method usage: Calibrated 3D: {calibrated_count}, Homography: {homography_count}")

    def validate_results(self) -> None:
        """Validate tracking results."""
        if len(self.player_positions) < 10:
            return

        sample_positions = self.player_positions[::max(1, len(self.player_positions)//50)]
        x_positions = [pos['x'] for pos in sample_positions]
        y_positions = [pos['y'] for pos in sample_positions]

        # Check boundary violations
        out_of_bounds = sum(1 for pos in sample_positions
                            if pos['x'] < -0.5 or pos['x'] > self.COURT_WIDTH + 0.5 or
                            pos['y'] < -0.5 or pos['y'] > self.COURT_LENGTH + 0.5)

        print(f"\n=== Tracking Quality ===")
        print(f"Positions analyzed: {len(sample_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds}/{len(sample_positions)} ({out_of_bounds/len(sample_positions):.1%})")

        if out_of_bounds/len(sample_positions) < 0.1:
            print("✓ Tracking quality good")
        else:
            print("⚠️ High out-of-bounds ratio - check calibration")
        print("=======================\n")

    def save_results(self) -> None:
        """Save clean, focused results."""
        frames_with_players = len(set(pos['frame_index'] for pos in self.player_positions))

        # Group positions by player
        player_0_positions = [pos for pos in self.player_positions if pos['player_id'] == 0]
        player_1_positions = [pos for pos in self.player_positions if pos['player_id'] == 1]

        output_data = {
            'video_info': {
                'video_name': self.video_name,
                'total_frames': self.video_info.get('total_frames', 0),
                'fps': self.video_info.get('fps', 0)
            },
            'court_info': {
                'width_meters': self.COURT_WIDTH,
                'length_meters': self.COURT_LENGTH,
                'coordinate_system': 'Origin at top-left corner, X=width, Y=length'
            },
            'tracking_summary': {
                'total_detections': len(self.player_positions),
                'frames_with_players': frames_with_players,
                'player_0_detections': len(player_0_positions),
                'player_1_detections': len(player_1_positions),
                'primary_method': 'calibrated_3d' if self.calibration_available else 'homography',
                'ankle_ground_offset_meters': self.ANKLE_TO_GROUND_OFFSET
            },
            'player_positions': self.player_positions
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {self.output_file}")
        print(f"Player 0: {len(player_0_positions)} positions")
        print(f"Player 1: {len(player_1_positions)} positions")

    def run(self) -> None:
        """Run the simplified tracking pipeline."""
        print(f"Starting ankle-based tracking for: {self.video_name}")

        try:
            self.load_calibration_data()
            self.load_pose_data()
            self.calculate_homography()
            self.process_all_frames()
            self.validate_results()
            self.save_results()
            print("✓ Ankle-based tracking completed successfully!")

        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python ankle_tracker.py <video_file_path> [--debug]")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    tracker = AnkleBasedCourtTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()