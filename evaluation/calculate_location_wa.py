#!/usr/bin/env python3
"""
Weighted Average Badminton Position Tracker

Uses weighted averaging of different joints with simple homography transformation.
Higher weights for lower body joints, lower weights for upper body.

Usage: python weighted_average_tracker.py <video_file_path> [--debug]
"""

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class WeightedAverageTracker:
    """Tracker using weighted average of multiple joints."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Pose joint indices
    HIP_LEFT = 11
    HIP_RIGHT = 12
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16
    KNEE_LEFT = 13
    KNEE_RIGHT = 14

    # Joint weights (higher = more important for ground position)
    JOINT_WEIGHTS = {
        ANKLE_LEFT: 0.35,
        ANKLE_RIGHT: 0.35,
        HIP_LEFT: 0.15,
        HIP_RIGHT: 0.15,
        KNEE_LEFT: 0.0,   # Not used in this simple version
        KNEE_RIGHT: 0.0   # Not used in this simple version
    }

    # Processing parameters
    CONFIDENCE_THRESHOLD = 0.5
    MAX_TRACKING_DISTANCE = 2.0

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
        self.player_positions = []

        # Simple tracking
        self._last_positions = []
        self._next_id = 0
        self._player_trajectories = defaultdict(list)
        self._final_player_mapping = {}

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

    def calculate_homography(self) -> None:
        """Calculate homography matrix from court corners."""
        # Image points
        image_points = np.array([
            self.court_points['P1'],
            self.court_points['P2'],
            self.court_points['P3'],
            self.court_points['P4']
        ], dtype=np.float32)

        # World points
        world_points = np.array([
            [0, 0],
            [0, self.COURT_LENGTH],
            [self.COURT_WIDTH, self.COURT_LENGTH],
            [self.COURT_WIDTH, 0]
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

        print("Homography matrix calculated successfully")

    def transform_point_to_world(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform pixel point to world coordinates using homography."""
        point = np.array([[pixel_point]], dtype=np.float32)
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

    def calculate_weighted_position(self, joints: List[Dict]) -> Optional[Tuple[float, float]]:
        """Calculate weighted average position of relevant joints."""
        weighted_positions = []
        total_weight = 0.0

        for joint_index, weight in self.JOINT_WEIGHTS.items():
            if weight > 0:  # Only process joints with non-zero weights
                joint_pos = self.extract_joint_position(joints, joint_index)
                if joint_pos:
                    world_pos = self.transform_point_to_world(joint_pos)
                    weighted_positions.append((world_pos[0], world_pos[1], weight))
                    total_weight += weight

        if not weighted_positions or total_weight == 0:
            return None

        # Calculate weighted average
        weighted_x = sum(pos[0] * pos[2] for pos in weighted_positions) / total_weight
        weighted_y = sum(pos[1] * pos[2] for pos in weighted_positions) / total_weight

        return (weighted_x, weighted_y)

    def calculate_player_position(self, joints: List[Dict]) -> Optional[Dict[str, float]]:
        """Calculate world position using weighted average of multiple joints."""
        # Calculate weighted position
        weighted_pos = self.calculate_weighted_position(joints)

        if weighted_pos is None:
            return None

        world_x, world_y = weighted_pos

        # Extract specific ankle positions for consistency
        ankle_left = self.extract_joint_position(joints, self.ANKLE_LEFT)
        ankle_right = self.extract_joint_position(joints, self.ANKLE_RIGHT)

        left_ankle_world = (0.0, 0.0)
        right_ankle_world = (0.0, 0.0)

        if ankle_left:
            left_ankle_world = self.transform_point_to_world(ankle_left)
        if ankle_right:
            right_ankle_world = self.transform_point_to_world(ankle_right)

        return {
            'hip_world_X': world_x,
            'hip_world_Y': world_y,
            'left_ankle_world_X': left_ankle_world[0],
            'left_ankle_world_Y': left_ankle_world[1],
            'right_ankle_world_X': right_ankle_world[0],
            'right_ankle_world_Y': right_ankle_world[1],
            'estimated_hip_height': 0.9  # Fixed default value
        }

    def assign_player_ids(self, frame_positions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Assign consistent player IDs using simple distance-based tracking."""
        assigned_positions = []
        used_ids = set()

        for current_pos in frame_positions:
            best_id = None
            min_distance = float('inf')

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

    def process_frame(self, frame_data: List[Dict], frame_index: int) -> List[Dict[str, Any]]:
        """Process all players in a single frame."""
        frame_positions = []

        for human_data in frame_data:
            joints = human_data.get('joints', [])
            position = self.calculate_player_position(joints)

            if position:
                frame_positions.append(position)

        frame_positions = self.assign_player_ids(frame_positions)
        return frame_positions

    def analyze_player_trajectories(self) -> None:
        """Simple analysis to identify the two main players."""
        for position in self.player_positions:
            player_id = position['tracked_id']
            self._player_trajectories[player_id].append(position)

        # Filter trajectories with sufficient frames
        valid_trajectories = {
            player_id: trajectory
            for player_id, trajectory in self._player_trajectories.items()
            if len(trajectory) >= 5
        }

        if len(valid_trajectories) >= 2:
            # Sort by trajectory length and take top 2
            sorted_trajectories = sorted(
                valid_trajectories.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            self._final_player_mapping = {
                sorted_trajectories[0][0]: 0,
                sorted_trajectories[1][0]: 1
            }

            # Map remaining to closest player
            for player_id in valid_trajectories:
                if player_id not in self._final_player_mapping:
                    self._final_player_mapping[player_id] = 0
        else:
            # Map all to player 0
            for player_id in valid_trajectories:
                self._final_player_mapping[player_id] = 0

    def apply_two_player_mapping(self) -> None:
        """Apply the final two-player mapping to all positions."""
        for position in self.player_positions:
            original_id = position['tracked_id']
            position['player_id'] = self._final_player_mapping.get(original_id, 0)

        self.player_positions.sort(key=lambda x: (x['frame_index'], x['player_id']))

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

            for position in positions:
                position['frame_index'] = frame_idx
                self.player_positions.append(position)

        print(f"Extracted {len(self.player_positions)} player positions")

        # Apply two-player system
        self.analyze_player_trajectories()
        self.apply_two_player_mapping()

    def save_results(self) -> None:
        """Save tracking results to JSON file."""
        frames_with_players = len(set(pos['frame_index'] for pos in self.player_positions))

        output_data = {
            'court_points': self.court_points,
            'all_court_points': self.pose_data.get('all_court_points', self.court_points),
            'video_info': self.video_info,
            'player_positions': self.player_positions,
            'tracking_method': "Weighted average positioning with simple homography transformation",
            'processing_info': {
                'total_positions': len(self.player_positions),
                'frames_with_players': frames_with_players,
                'method_details': {
                    'weighted_average': True,
                    'joint_weights': self.JOINT_WEIGHTS,
                    'homography_only': True,
                    'no_height_corrections': True,
                    'fixed_hip_height': 0.9
                }
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
        print(f"Starting weighted average tracking for: {self.video_name}")

        try:
            self.load_pose_data()
            self.calculate_homography()
            self.process_all_frames()
            self.save_results()
            print("Weighted average tracking completed successfully!")

        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python weighted_average_tracker.py <video_file_path> [--debug]")
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    tracker = WeightedAverageTracker(video_path, debug=debug)
    tracker.run()


if __name__ == "__main__":
    main()