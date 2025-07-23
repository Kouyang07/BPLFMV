#!/usr/bin/env python3
"""
Enhanced Badminton Player Ankle Tracking Script - Stage 3 Visualization Compatible

Tracks individual ankle positions using enhanced homography approach:
- Outputs in exact format expected by stage 3 visualization
- Frame-organized data structure: frame_data[frame_idx][player_id] = {ankles: [...], center_position: {...}}
- Individual ankle position tracking (left and right separately)
- Enhanced homography with calibration improvements
- Proper coordinate system and boundary validation

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
from collections import defaultdict


class EnhancedAnkleTracker:
    """Enhanced tracker focusing on individual ankle positions with stage 3 visualization compatibility."""

    # Court dimensions (meters) - BWF standard
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Pose joint indices (COCO format)
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
        self.calibration_file = self.results_dir / "calibration.csv"
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

        # Frame-organized output structure (compatible with stage 3 visualization)
        self.frame_data = {}  # Will be serialized as dict, not defaultdict

    def load_calibration_data(self) -> None:
        """Load calibration data for homography enhancement."""
        if not self.calibration_file.exists():
            print("⚠️  No calibration data found - using basic homography")
            return

        calibration_params = {}

        try:
            with open(self.calibration_file, 'r') as file:
                csv_reader = csv.reader(file)
                current_section = None

                for row in csv_reader:
                    if not row or row[0].startswith('#') or len(row) < 2:
                        continue

                    key = row[0].strip()
                    value = row[1].strip()

                    try:
                        # Camera matrix parameters
                        if key == 'fx':
                            current_section = 'camera_matrix'
                            calibration_params['fx'] = float(value)
                        elif key == 'fy' and current_section == 'camera_matrix':
                            calibration_params['fy'] = float(value)
                        elif key == 'cx' and current_section == 'camera_matrix':
                            calibration_params['cx'] = float(value)
                        elif key == 'cy' and current_section == 'camera_matrix':
                            calibration_params['cy'] = float(value)
                        elif key in ['k1', 'k2', 'p1', 'p2', 'k3']:
                            calibration_params[key] = float(value)
                        elif key == 'camera_height_m':
                            calibration_params['camera_height_m'] = float(value)
                        elif key == 'reprojection_error_px':
                            calibration_params['reprojection_error_px'] = float(value)
                    except (ValueError, IndexError) as e:
                        if self.debug:
                            print(f"Warning: Could not parse calibration row {row}: {e}")
                        continue

            # Reconstruct camera matrix
            if all(param in calibration_params for param in ['fx', 'fy', 'cx', 'cy']):
                self.camera_matrix = np.array([
                    [calibration_params['fx'], 0, calibration_params['cx']],
                    [0, calibration_params['fy'], calibration_params['cy']],
                    [0, 0, 1]
                ], dtype=np.float32)

                if self.debug:
                    print(f"✓ Camera matrix reconstructed:")
                    print(f"  fx: {calibration_params['fx']:.1f}")
                    print(f"  fy: {calibration_params['fy']:.1f}")
                    print(f"  cx: {calibration_params['cx']:.1f}")
                    print(f"  cy: {calibration_params['cy']:.1f}")

            # Reconstruct distortion coefficients
            dist_coeffs = []
            for param in ['k1', 'k2', 'p1', 'p2', 'k3']:
                if param in calibration_params:
                    dist_coeffs.append(calibration_params[param])
            if dist_coeffs:
                self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
                if self.debug:
                    print(f"✓ Distortion coefficients: {self.dist_coeffs}")

            # Get camera height
            if 'camera_height_m' in calibration_params:
                self.camera_height = calibration_params['camera_height_m']
                if self.debug:
                    print(f"✓ Camera height: {self.camera_height:.1f}m")

            # Get reprojection error for quality assessment
            if 'reprojection_error_px' in calibration_params:
                self.reprojection_error = calibration_params['reprojection_error_px']
                if self.debug:
                    print(f"✓ Reprojection error: {self.reprojection_error:.2f}px")

            # Check if calibration can enhance homography
            self.calibration_available = (
                    self.camera_matrix is not None and
                    self.camera_height is not None and
                    (self.reprojection_error is None or self.reprojection_error < 30)
            )

            if self.calibration_available:
                print(f"✓ Calibration available for homography enhancement (error: {self.reprojection_error:.1f}px)")
                self._calculate_enhancement_parameters()
            else:
                print(f"⚠️  Calibration quality insufficient for enhancement")

        except Exception as e:
            print(f"⚠️  Error loading calibration data: {e}")
            print("   Falling back to basic homography")
            self.calibration_available = False

    def _calculate_enhancement_parameters(self) -> None:
        """Calculate parameters to enhance homography using calibration data."""
        if not self.calibration_available:
            return

        # Estimate pixel-to-meter ratio at court level using camera height
        if self.camera_height and self.camera_matrix is not None:
            # Approximate focal length in pixels
            focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2

            # Estimate pixels per meter at ground level
            self.pixel_to_meter_ratio = focal_length / self.camera_height

            # Enhanced ankle offset in pixels based on camera geometry
            self.enhanced_ankle_offset = self.BASE_ANKLE_OFFSET * self.pixel_to_meter_ratio

            if self.debug:
                print(f"Enhancement parameters:")
                print(f"  Camera height: {self.camera_height:.2f}m")
                print(f"  Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.1f} px/m")
                print(f"  Enhanced ankle offset: {self.enhanced_ankle_offset:.1f} pixels")

    def load_pose_data(self) -> None:
        """Load pose detection data."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.pose_data = data
        self.video_info = data.get('video_info', {})

        # Get court points - try multiple possible locations
        self.court_points = data.get('court_points', {})
        if not self.court_points:
            # Try alternative locations in the pose data
            self.court_points = data.get('all_court_points', {})

        if not self.court_points:
            raise ValueError("No court points found in pose data")

        print(f"✓ Loaded pose data with {len(data.get('pose_data', []))} detections")
        print(f"✓ Loaded {len(self.court_points)} court points")

        if self.debug:
            print("Available court points:")
            for name, coords in self.court_points.items():
                print(f"  {name}: {coords}")

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        required_corners = ['P1', 'P2', 'P3', 'P4']

        # Check which corners are available
        available_corners = [corner for corner in required_corners if corner in self.court_points]
        missing_corners = [corner for corner in required_corners if corner not in self.court_points]

        if len(available_corners) < 4:
            print(f"Available corners: {available_corners}")
            print(f"Missing corners: {missing_corners}")
            print(f"All available court points: {list(self.court_points.keys())}")
            raise ValueError(f"Missing required court corners: {missing_corners}")

        # Extract corner coordinates - handle both list and dict formats
        image_points = []
        for corner in required_corners:
            coords = self.court_points[corner]
            if isinstance(coords, list) and len(coords) >= 2:
                image_points.append([coords[0], coords[1]])
            elif isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                image_points.append([coords['x'], coords['y']])
            else:
                raise ValueError(f"Invalid coordinate format for {corner}: {coords}")

        image_points = np.array(image_points, dtype=np.float32)

        # World coordinates for standard badminton court
        world_points = np.array([
            [0, 0],                    # P1: Top-left
            [0, self.COURT_LENGTH],    # P2: Bottom-left
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Bottom-right
            [self.COURT_WIDTH, 0]      # P4: Top-right
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

        print("✓ Homography matrix calculated")
        if self.debug:
            print("Court corners (pixels):")
            for i, corner in enumerate(required_corners):
                coords = self.court_points[corner]
                if isinstance(coords, list):
                    px, py = coords[0], coords[1]
                else:
                    px, py = coords['x'], coords['y']
                print(f"  {corner}: ({px:.1f}, {py:.1f})")

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
        except Exception as e:
            if self.debug:
                print(f"Undistortion failed: {e}")
            return point

    def calculate_ankle_ground_position(self, ankle_pixel: Tuple[float, float], ankle_side: str) -> Tuple[float, float]:
        """Calculate ankle ground position using enhanced homography."""
        try:
            # Step 1: Undistort the pixel point if calibration available
            if self.calibration_available:
                undistorted_pixel = self.undistort_point(ankle_pixel)
                if self.debug and ankle_side == 'left':  # Only debug left ankle to avoid spam
                    px_diff = undistorted_pixel[0] - ankle_pixel[0]
                    py_diff = undistorted_pixel[1] - ankle_pixel[1]
                    print(f"Undistortion shift: ({px_diff:.1f}, {py_diff:.1f}) pixels")
            else:
                undistorted_pixel = ankle_pixel

            # Step 2: Apply ankle-to-ground offset
            if self.enhanced_ankle_offset is not None:
                # Use calibration-enhanced offset
                offset_y = self.enhanced_ankle_offset
                if self.debug and ankle_side == 'left':
                    print(f"Using enhanced ankle offset: {offset_y:.1f} pixels")
            else:
                # Use basic fixed offset
                offset_y = 12.0  # pixels
                if self.debug and ankle_side == 'left':
                    print(f"Using basic ankle offset: {offset_y:.1f} pixels")

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
            try:
                point = np.array([[ankle_pixel]], dtype=np.float32)
                world_point = cv2.perspectiveTransform(point, self.homography_matrix)
                return float(world_point[0][0][0]), float(world_point[0][0][1])
            except:
                return 0.0, 0.0

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

            # Get confidence for this joint
            left_confidence = 0.0
            for joint in joints:
                if joint['joint_index'] == self.ANKLE_LEFT:
                    left_confidence = joint['confidence']
                    break

            ankle_detections.append({
                'ankle_side': 'left',
                'world_x': left_world_x,
                'world_y': left_world_y,
                'joint_confidence': left_confidence,
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
            })

        # Process right ankle
        ankle_right_pixel = self.extract_joint_position(joints, self.ANKLE_RIGHT)
        if ankle_right_pixel:
            right_world_x, right_world_y = self.calculate_ankle_ground_position(ankle_right_pixel, "right")

            # Get confidence for this joint
            right_confidence = 0.0
            for joint in joints:
                if joint['joint_index'] == self.ANKLE_RIGHT:
                    right_confidence = joint['confidence']
                    break

            ankle_detections.append({
                'ankle_side': 'right',
                'world_x': right_world_x,
                'world_y': right_world_y,
                'joint_confidence': right_confidence,
                'method': 'enhanced_homography' if self.calibration_available else 'basic_homography'
            })

        return ankle_detections

    def assign_player_ids(self, frame_ankle_data: Dict[int, List[Dict]]) -> Dict[str, Dict[str, Any]]:
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

        # Create frame structure with player assignments (exact format expected by stage 3)
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
                # Store as regular dict (not defaultdict) for JSON serialization
                self.frame_data[frame_index] = player_assignments

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

        print(f"📊 Total ankle detections: {total_ankle_detections}")
        print(f"📊 Method usage: Enhanced: {enhanced_count}, Basic: {basic_count}")
        print(f"📊 Ankle distribution: Left: {left_ankles}, Right: {right_ankles}")

    def validate_results(self) -> None:
        """Validate tracking results."""
        if not self.frame_data:
            print("⚠️  No frame data to validate")
            return

        sample_positions = []
        for frame_data in list(self.frame_data.values())[::max(1, len(self.frame_data)//100)]:
            for player_data in frame_data.values():
                for ankle in player_data['ankles']:
                    sample_positions.append(ankle)

        if len(sample_positions) < 10:
            print("⚠️  Too few positions for validation")
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
            print("✅ Ankle tracking quality good")
        else:
            print("⚠️  High out-of-bounds ratio - check calibration or court corners")
        print("===============================\n")

    def save_results(self) -> None:
        """Save frame-organized ankle position results in exact format expected by stage 3 visualization."""
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

        # Convert frame indices to strings as expected by visualization
        frame_data_dict = {str(frame_idx): player_data for frame_idx, player_data in self.frame_data.items()}

        # Create output data structure exactly as expected by stage 3 visualization
        output_data = {
            'video_info': {
                'video_name': self.video_name,
                'frame_count': self.video_info.get('frame_count', 0),
                'fps': self.video_info.get('fps', 0),
                'width': self.video_info.get('width', 0),
                'height': self.video_info.get('height', 0)
            },
            'court_info': {
                'width_meters': self.COURT_WIDTH,
                'length_meters': self.COURT_LENGTH,
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
                'ankle_ground_offset_meters': self.BASE_ANKLE_OFFSET,
                'calibration_enhanced': self.calibration_available
            },
            'enhancement_info': {
                'camera_height_meters': self.camera_height if self.calibration_available else None,
                'pixel_to_meter_ratio': self.pixel_to_meter_ratio if self.calibration_available else None,
                'enhanced_ankle_offset_pixels': self.enhanced_ankle_offset if self.calibration_available else None,
                'reprojection_error_px': self.reprojection_error if self.calibration_available else None
            },
            # This is the key structure that stage 3 visualization expects
            'frame_data': frame_data_dict
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Results saved to: {self.output_file}")
        print(f"📊 Frames with data: {total_frames_with_data}")
        print(f"📊 Player 0: {player_0_detections} ankle detections")
        print(f"📊 Player 1: {player_1_detections} ankle detections")
        print(f"📊 Left ankles: {left_ankle_detections}, Right ankles: {right_ankle_detections}")
        print(f"📊 Format: Stage 3 visualization compatible")

    def run(self) -> None:
        """Run the enhanced ankle tracking pipeline."""
        print(f"🚀 Starting enhanced individual ankle tracking for: {self.video_name}")
        print("="*80)

        try:
            print("📊 Step 1: Loading calibration data...")
            self.load_calibration_data()

            print("📍 Step 2: Loading pose data...")
            self.load_pose_data()

            print("🔧 Step 3: Calculating homography...")
            self.calculate_homography()

            print("🏃 Step 4: Processing all frames...")
            self.process_all_frames()

            print("✅ Step 5: Validating results...")
            self.validate_results()

            print("💾 Step 6: Saving results...")
            self.save_results()

            print("="*80)
            print("✅ Enhanced ankle tracking completed successfully!")
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
        print("  - OpenCV, NumPy")
        print("\nPipeline:")
        print("  1. python detect_court.py <video_path>")
        print("  2. python detect_pose.py <video_path>")
        print("  3. python calculate_location.py <video_path>")
        print("  4. python visualize.py <video_path> --stage 3")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    tracker = EnhancedAnkleTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()