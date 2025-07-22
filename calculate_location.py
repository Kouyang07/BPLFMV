#!/usr/bin/env python3
"""
Enhanced Badminton Player Ankle Tracking Script - Frame-Organized Output

Tracks individual ankle positions using enhanced homography approach:
- Organizes output by frame -> players -> ankle detections
- Removes pixel coordinates from output (only world coordinates)
- Uses homography as primary method with calibration-based improvements
- Applies undistortion to improve homography accuracy when calibration available
- Enhanced ankle-to-ground offset calculation using calibration data

Key features:
1. Individual ankle position tracking (left and right separately)
2. Frame-organized output structure
3. Homography-only approach with calibration enhancements
4. Improved ground projection using calibration-informed offsets
5. Clean boundary validation and quality assessment

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
from collections import defaultdict


class EnhancedAnkleTracker:
    """Enhanced tracker focusing on individual ankle positions with improved homography."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Pose joint indices
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    BASE_ANKLE_OFFSET = 0.04  # Base 4cm offset from ankle to ground
    PIXEL_OFFSET_SCALING = 1.0  # Will be adjusted based on calibration

    def __init__(self, video_path: str, debug: bool = False):
        """Initialize enhanced ankle tracker."""
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

        # Calibrated camera parameters for enhancement
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_height = None
        self.reprojection_error = None
        self.calibration_available = False

        # Enhanced parameters
        self.pixel_to_meter_ratio = None
        self.enhanced_ankle_offset = None

        # Frame-organized output structure
        self.frame_data = defaultdict(lambda: defaultdict(list))

    def load_calibration_data(self) -> None:
        """Load calibration data for homography enhancement."""
        if not self.calibration_file.exists():
            print("⚠️  No calibration data found - using basic homography")
            return

        intrinsic_params = {}
        distortion_params = {}
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
                elif key.startswith('translation_'):
                    translation_params[key] = float(value)
                elif key == 'reprojection_error_pixels':
                    self.reprojection_error = float(value)

        # Reconstruct camera parameters for enhancement
        if all(param in intrinsic_params for param in ['intrinsic_fx', 'intrinsic_fy', 'intrinsic_cx', 'intrinsic_cy']):
            self.camera_matrix = np.array([
                [intrinsic_params['intrinsic_fx'], 0, intrinsic_params['intrinsic_cx']],
                [0, intrinsic_params['intrinsic_fy'], intrinsic_params['intrinsic_cy']],
                [0, 0, 1]
            ], dtype=np.float32)

        dist_keys = sorted([k for k in distortion_params.keys() if k.startswith('distortion_')])
        if dist_keys:
            self.dist_coeffs = np.array([distortion_params[k] for k in dist_keys], dtype=np.float32)

        if 'translation_z' in translation_params:
            self.camera_height = abs(float(translation_params['translation_z']))

        # Check if calibration can enhance homography
        self.calibration_available = (
                self.camera_matrix is not None and
                self.camera_height is not None and
                (self.reprojection_error is None or self.reprojection_error < 30.0)
        )

        if self.calibration_available:
            print(f"✓ Calibration available for homography enhancement (error: {self.reprojection_error:.1f}px)")
            self._calculate_enhancement_parameters()
        else:
            print(f"⚠️  Calibration quality insufficient for enhancement")

    def _calculate_enhancement_parameters(self) -> None:
        """Calculate parameters to enhance homography using calibration data."""
        if not self.calibration_available:
            return

        # Estimate pixel-to-meter ratio at court level using camera height
        # This helps scale the ankle offset appropriately
        if self.camera_height and self.camera_matrix is not None:
            # Approximate focal length in pixels
            focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2

            # Estimate pixels per meter at ground level
            self.pixel_to_meter_ratio = focal_length / self.camera_height

            # Enhanced ankle offset in pixels based on camera geometry
            self.enhanced_ankle_offset = self.BASE_ANKLE_OFFSET * self.pixel_to_meter_ratio

            if self.debug:
                print(f"Camera height: {self.camera_height:.2f}m")
                print(f"Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.1f} px/m")
                print(f"Enhanced ankle offset: {self.enhanced_ankle_offset:.1f} pixels")

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

    def calculate_ankle_ground_position(self, ankle_pixel: Tuple[float, float], ankle_side: str) -> Tuple[float, float]:
        """Calculate ankle ground position using enhanced homography."""
        try:
            # Step 1: Undistort the pixel point if calibration available
            if self.calibration_available:
                undistorted_pixel = self.undistort_point(ankle_pixel)
            else:
                undistorted_pixel = ankle_pixel

            # Step 2: Apply ankle-to-ground offset
            if self.enhanced_ankle_offset is not None:
                # Use calibration-enhanced offset
                offset_y = self.enhanced_ankle_offset
            else:
                # Use basic fixed offset
                offset_y = 12.0  # pixels

            corrected_pixel = (undistorted_pixel[0], undistorted_pixel[1] + offset_y)

            # Step 3: Transform to world coordinates using homography
            point = np.array([[corrected_pixel]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)

            world_x = float(world_point[0][0][0])
            world_y = float(world_point[0][0][1])

            # Step 4: Boundary validation with slight tolerance
            world_x = max(-1.0, min(self.COURT_WIDTH + 1.0, world_x))
            world_y = max(-1.0, min(self.COURT_LENGTH + 1.0, world_y))

            return world_x, world_y

        except Exception as e:
            if self.debug:
                print(f"Position calculation failed for {ankle_side} ankle: {e}")
            # Simple fallback
            point = np.array([[ankle_pixel]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)
            return float(world_point[0][0][0]), float(world_point[0][0][1])

    def extract_joint_position(self, joints: List[Dict], joint_index: int) -> Optional[Tuple[float, float]]:
        """Extract joint position if confidence is sufficient."""
        for joint in joints:
            if (joint['joint_index'] == joint_index and
                    joint['confidence'] > self.CONFIDENCE_THRESHOLD and
                    joint['x'] > 0 and joint['y'] > 0):
                return joint['x'], joint['y']
        return None

    def process_person_ankles(self, joints: List[Dict], frame_index: int, person_id: int) -> List[Dict[str, Any]]:
        """Process individual ankle positions for a person."""
        ankle_detections = []

        # Process left ankle
        ankle_left_pixel = self.extract_joint_position(joints, self.ANKLE_LEFT)
        if ankle_left_pixel:
            left_world_x, left_world_y = self.calculate_ankle_ground_position(ankle_left_pixel, "left")
            ankle_detections.append({
                'ankle_side': 'left',
                'world_x': left_world_x,
                'world_y': left_world_y,
                'joint_confidence': next(joint['confidence'] for joint in joints if joint['joint_index'] == self.ANKLE_LEFT),
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
            })

        # Process right ankle
        ankle_right_pixel = self.extract_joint_position(joints, self.ANKLE_RIGHT)
        if ankle_right_pixel:
            right_world_x, right_world_y = self.calculate_ankle_ground_position(ankle_right_pixel, "right")
            ankle_detections.append({
                'ankle_side': 'right',
                'world_x': right_world_x,
                'world_y': right_world_y,
                'joint_confidence': next(joint['confidence'] for joint in joints if joint['joint_index'] == self.ANKLE_RIGHT),
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
            })

        return ankle_detections

    def assign_player_ids(self, frame_ankle_data: Dict[int, List[Dict]]) -> Dict[str, List[Dict]]:
        """Assign player IDs based on court position and person grouping."""
        if not frame_ankle_data:
            return {}

        # Calculate average position for each person to assign player ID
        person_positions = []
        for person_id, ankle_detections in frame_ankle_data.items():
            if not ankle_detections:
                continue

            avg_x = sum(ankle['world_x'] for ankle in ankle_detections) / len(ankle_detections)
            avg_y = sum(ankle['world_y'] for ankle in ankle_detections) / len(ankle_detections)
            person_positions.append({
                'person_id': person_id,
                'avg_x': avg_x,
                'avg_y': avg_y,
                'ankle_detections': ankle_detections
            })

        # Sort by Y position and assign player IDs
        person_positions.sort(key=lambda p: p['avg_y'])

        # Create frame structure with player assignments
        frame_players = {}
        for player_id, person_data in enumerate(person_positions[:2]):  # Max 2 players
            frame_players[f"player_{player_id}"] = {
                'ankles': person_data['ankle_detections'],
                'center_position': {
                    'x': person_data['avg_x'],
                    'y': person_data['avg_y']
                }
            }

        return frame_players

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> None:
        """Process all people in a single frame and organize by frame -> players -> ankles."""
        frame_ankle_data = {}

        # Process each person in the frame
        for person_id, human_data in enumerate(frame_data):
            joints = human_data.get('joints', [])
            person_ankles = self.process_person_ankles(joints, frame_index, person_id)
            if person_ankles:
                frame_ankle_data[person_id] = person_ankles

        # Assign player IDs and organize data
        if frame_ankle_data:
            player_assignments = self.assign_player_ids(frame_ankle_data)
            if player_assignments:
                self.frame_data[frame_index] = player_assignments

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
            self.process_frame(frame_data, frame_idx)

        print(f"Processed {len(self.frame_data)} frames with ankle detections")

        # Report statistics
        total_ankle_detections = 0
        enhanced_count = 0
        basic_count = 0
        left_ankles = 0
        right_ankles = 0

        for frame_data in self.frame_data.values():
            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    total_ankle_detections += 1
                    if ankle['method'] == 'enhanced_homography':
                        enhanced_count += 1
                    else:
                        basic_count += 1
                    if ankle['ankle_side'] == 'left':
                        left_ankles += 1
                    else:
                        right_ankles += 1

        print(f"Total ankle detections: {total_ankle_detections}")
        print(f"Method usage: Enhanced: {enhanced_count}, Basic: {basic_count}")
        print(f"Ankle distribution: Left: {left_ankles}, Right: {right_ankles}")

    def validate_results(self) -> None:
        """Validate tracking results."""
        if not self.frame_data:
            return

        sample_positions = []
        for frame_data in list(self.frame_data.values())[::max(1, len(self.frame_data)//100)]:
            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    sample_positions.append(ankle)

        if len(sample_positions) < 10:
            return

        x_positions = [pos['world_x'] for pos in sample_positions]
        y_positions = [pos['world_y'] for pos in sample_positions]

        # Check boundary violations
        out_of_bounds = sum(1 for pos in sample_positions
                            if pos['world_x'] < -0.5 or pos['world_x'] > self.COURT_WIDTH + 0.5 or
                            pos['world_y'] < -0.5 or pos['world_y'] > self.COURT_LENGTH + 0.5)

        # Analyze by ankle side
        left_positions = [pos for pos in sample_positions if pos['ankle_side'] == 'left']
        right_positions = [pos for pos in sample_positions if pos['ankle_side'] == 'right']

        print(f"\n=== Ankle Tracking Quality ===")
        print(f"Total ankle positions analyzed: {len(sample_positions)}")
        print(f"Left ankle positions: {len(left_positions)}")
        print(f"Right ankle positions: {len(right_positions)}")
        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}m")
        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}m")
        print(f"Out of bounds: {out_of_bounds}/{len(sample_positions)} ({out_of_bounds/len(sample_positions):.1%})")

        if out_of_bounds/len(sample_positions) < 0.1:
            print("✓ Ankle tracking quality good")
        else:
            print("⚠️ High out-of-bounds ratio - check calibration or court corners")
        print("===============================\n")

    def save_results(self) -> None:
        """Save frame-organized ankle position results."""
        # Calculate summary statistics
        total_frames_with_data = len(self.frame_data)
        total_ankle_detections = 0
        player_0_detections = 0
        player_1_detections = 0
        left_ankle_detections = 0
        right_ankle_detections = 0

        for frame_data in self.frame_data.values():
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

        # Convert defaultdict to regular dict for JSON serialization
        frame_data_dict = {str(frame_idx): player_data for frame_idx, player_data in self.frame_data.items()}

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
                'frames_with_ankle_data': total_frames_with_data,
                'total_ankle_detections': total_ankle_detections,
                'player_0_detections': player_0_detections,
                'player_1_detections': player_1_detections,
                'left_ankle_detections': left_ankle_detections,
                'right_ankle_detections': right_ankle_detections,
                'primary_method': 'enhanced_homography' if self.calibration_available else 'basic_homography',
                'ankle_ground_offset_meters': self.BASE_ANKLE_OFFSET,
                'calibration_enhanced': self.calibration_available
            },
            'enhancement_info': {
                'camera_height_meters': self.camera_height if self.calibration_available else None,
                'pixel_to_meter_ratio': self.pixel_to_meter_ratio if self.calibration_available else None,
                'enhanced_ankle_offset_pixels': self.enhanced_ankle_offset if self.calibration_available else None
            },
            'frame_data': frame_data_dict
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {self.output_file}")
        print(f"Frames with data: {total_frames_with_data}")
        print(f"Player 0: {player_0_detections} ankle detections")
        print(f"Player 1: {player_1_detections} ankle detections")
        print(f"Left ankles: {left_ankle_detections}, Right ankles: {right_ankle_detections}")

    def run(self) -> None:
        """Run the enhanced ankle tracking pipeline."""
        print(f"Starting enhanced individual ankle tracking for: {self.video_name}")

        try:
            self.load_calibration_data()
            self.load_pose_data()
            self.calculate_homography()
            self.process_all_frames()
            self.validate_results()
            self.save_results()
            print("✓ Enhanced ankle tracking completed successfully!")

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

    tracker = EnhancedAnkleTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()