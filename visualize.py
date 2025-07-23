#!/usr/bin/env python3
"""
Enhanced Universal Visualization Script for Complete Badminton Analysis Pipeline

Fully compatible with all pipeline outputs:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → Enhanced ankle position tracking visualization
- Stage 4: corrected_positions.json → Jump correction comparison visualization

Features:
- Frame-organized data structure support from calculate_location.py
- Enhanced ankle-only tracking visualization
- Court coordinate system correction and proper mirroring
- Jump correction visualization with trajectory comparison
- Automatic data format detection and compatibility
- Performance optimized with multithreading
"""

import sys
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import gc
import time

import numpy as np
import cv2
import json
import csv
import argparse
import logging
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    """Enhanced video processing with accurate badminton court visualization"""

    def __init__(self, video_path: str, num_threads: Optional[int] = None):
        self.video_path = video_path
        self.num_threads = num_threads or min(mp.cpu_count(), 6)

        # Get video info once
        self.video_info = self._get_video_info()

        # Official badminton court dimensions (BWF standard)
        self.court_length_m = 13.4  # Total length
        self.court_width_m = 6.1    # Total width
        self.margin_m = 2.0         # Margin around court
        self.court_scale = 80       # Scale for better visualization

        # Calculate court image dimensions
        self.court_img_h = int((self.court_length_m + 2 * self.margin_m) * self.court_scale)
        self.court_img_w = int((self.court_width_m + 2 * self.margin_m) * self.court_scale)

        # Enhanced color scheme
        self.colors = {
            'court_boundary': (220, 220, 220),      # Light gray
            'service_lines': (180, 180, 180),       # Medium gray
            'center_line': (160, 160, 160),         # Darker gray
            'net': (255, 255, 255),                 # White
            'background': (40, 50, 45),             # Dark green
            'text': (200, 200, 200),                # Light gray
            'players': [(0, 255, 100), (255, 100, 0), (100, 150, 255),
                        (255, 200, 0), (255, 0, 150), (0, 255, 255)]
        }

        # Create court template
        self.court_template = self._create_court_template()

        # Pose skeleton connections (COCO format)
        self.pose_edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
                           (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

    def _get_video_info(self) -> Dict[str, Any]:
        """Extract video information"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info

    def _create_court_template(self):
        """Create accurate badminton court template"""
        court_img = np.full((self.court_img_h, self.court_img_w, 3), self.colors['background'], dtype=np.uint8)

        # Calculate coordinates
        margin_px = int(self.margin_m * self.court_scale)
        court_width_px = int(self.court_width_m * self.court_scale)
        court_length_px = int(self.court_length_m * self.court_scale)

        # Court boundaries
        left = margin_px
        right = margin_px + court_width_px
        top = margin_px
        bottom = margin_px + court_length_px
        center_x = margin_px + court_width_px // 2
        center_y = margin_px + court_length_px // 2

        # Service dimensions
        service_line_distance_px = int(1.98 * self.court_scale)  # 1.98m from net
        back_service_distance_px = int(0.76 * self.court_scale)  # 0.76m from back
        side_service_distance_px = int(0.46 * self.court_scale)  # Singles margin

        # Service court boundaries
        front_service_top = center_y - service_line_distance_px
        front_service_bottom = center_y + service_line_distance_px
        back_service_top = top + back_service_distance_px
        back_service_bottom = bottom - back_service_distance_px

        # Singles lines
        singles_left = left + side_service_distance_px
        singles_right = right - side_service_distance_px

        # Draw court elements
        # 1. Main court boundary
        cv2.rectangle(court_img, (left, top), (right, bottom), self.colors['court_boundary'], 3)

        # 2. Singles lines
        cv2.line(court_img, (singles_left, top), (singles_left, bottom), self.colors['court_boundary'], 2)
        cv2.line(court_img, (singles_right, top), (singles_right, bottom), self.colors['court_boundary'], 2)

        # 3. Net line
        cv2.line(court_img, (left, center_y), (right, center_y), self.colors['net'], 4)

        # 4. Center line
        cv2.line(court_img, (center_x, front_service_top), (center_x, front_service_bottom), self.colors['center_line'], 2)

        # 5. Service lines
        cv2.line(court_img, (singles_left, front_service_top), (singles_right, front_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (singles_left, front_service_bottom), (singles_right, front_service_bottom),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (left, back_service_top), (right, back_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (left, back_service_bottom), (right, back_service_bottom),
                 self.colors['service_lines'], 2)

        # 6. Net posts
        cv2.circle(court_img, (left, center_y), 4, self.colors['net'], -1)
        cv2.circle(court_img, (right, center_y), 4, self.colors['net'], -1)

        # 7. Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(court_img, "NET", (center_x - 20, center_y - 10),
                    font, 0.5, self.colors['text'], 1)

        return court_img

    def get_frame_batch(self, start_frame: int, batch_size: int) -> List[Tuple[int, np.ndarray]]:
        """Get a batch of frames"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((start_frame + i, frame.copy()))

        cap.release()
        return frames

    def get_color(self, player_id: int) -> Tuple[int, int, int]:
        """Get color for player"""
        return self.colors['players'][player_id % len(self.colors['players'])]

    def world_to_court(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to court image coordinates"""
        px = int((x + self.margin_m) * self.court_scale)
        py = int((y + self.margin_m) * self.court_scale)
        return (px, py)

    def add_info_panel(self, court_img: np.ndarray, frame_idx: int, info: str) -> np.ndarray:
        """Add information panel to court visualization"""
        panel_height = 80
        panel = np.full((panel_height, court_img.shape[1], 3),
                        tuple(int(c * 0.8) for c in self.colors['background']), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"Frame: {frame_idx:06d}", (10, 25),
                    font, 0.6, self.colors['text'], 1)
        cv2.putText(panel, info, (10, 50),
                    font, 0.5, self.colors['text'], 1)

        if hasattr(self, 'video_info') and self.video_info["fps"] > 0:
            timestamp = frame_idx / self.video_info["fps"]
            cv2.putText(panel, f"Time: {timestamp:.2f}s", (200, 25),
                        font, 0.5, self.colors['text'], 1)

        return np.vstack([court_img, panel])

# Data loading utilities
def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_court_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load court points from CSV"""
    court_points = {}
    with open(csv_path, 'r') as file:
        # Check if header exists
        first_line = file.readline().strip()
        file.seek(0)

        if 'Point' in first_line and 'X' in first_line:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    court_points[row['Point']] = [float(row['X']), float(row['Y'])]
                except (ValueError, KeyError):
                    continue
        else:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 3:
                    try:
                        court_points[row[0]] = [float(row[1]), float(row[2])]
                    except (ValueError, IndexError):
                        continue
    return court_points

def organize_by_frame(data_list: List[Dict]) -> Dict[int, List[Dict]]:
    """Organize data by frame index"""
    frame_dict = defaultdict(list)
    for item in data_list:
        frame_idx = item["frame_index"]
        frame_dict[frame_idx].append(item)
    return dict(frame_dict)

def extract_corner_points(all_court_points: Dict) -> np.ndarray:
    """Extract corner points P1-P4"""
    corner_points = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_court_points:
            corner_points.append(all_court_points[point_name])

    if len(corner_points) != 4:
        # Fallback: use first 4 points
        points_list = list(all_court_points.values())[:4]
        corner_points = points_list

    return np.array(corner_points, dtype=np.float32)

# Visualization functions for each stage
def draw_court_points(frame: np.ndarray, court_points: Dict, show_labels: bool = True) -> np.ndarray:
    """Draw court points with enhanced styling"""
    frame_display = frame.copy()

    for point_name, coords in court_points.items():
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])

            # Point styling based on type
            if point_name.startswith('P') and point_name[1:].isdigit():
                color = (0, 255, 255)  # Cyan for corners
                radius = 8
                border_color = (255, 255, 255)
            elif 'Net' in point_name:
                color = (255, 100, 255)  # Magenta for net
                radius = 6
                border_color = (255, 255, 255)
            else:
                color = (255, 150, 0)  # Orange for others
                radius = 4
                border_color = (200, 200, 200)

            # Draw point with border
            cv2.circle(frame_display, (x, y), radius + 2, border_color, -1)
            cv2.circle(frame_display, (x, y), radius, color, -1)

            # Labels
            if show_labels:
                cv2.putText(frame_display, point_name, (x + 12, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame_display

def draw_court_polygon(frame: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Draw court polygon overlay"""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(frame, 0.92, overlay, 0.08, 0, frame)

        # Border
        cv2.polylines(frame, [pts], True, (0, 255, 0), 4)

    return frame

def process_stage1_batch(args):
    """Process batch for Stage 1: Court detection"""
    batch_frames, court_points, corner_points = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_court_points(frame, court_points)
        frame_display = draw_court_polygon(frame_display, corner_points)

        # Frame info
        cv2.rectangle(frame_display, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (400, 60), (0, 255, 0), 2)
        cv2.putText(frame_display, f"COURT DETECTION - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Points: {len(court_points)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage2_batch(args):
    """Process batch for Stage 2: Pose estimation"""
    batch_frames, poses_by_frame, court_points, pose_edges, colors = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_court_points(frame, court_points, show_labels=False)

        frame_poses = poses_by_frame.get(frame_idx, [])
        total_joints_in_court = 0

        for pose in frame_poses:
            human_idx = pose["human_index"]
            joints = pose["joints"]
            color = colors[human_idx % len(colors)]

            # Draw joints
            joint_positions = {}
            for joint in joints:
                if joint["confidence"] > 0.5:
                    joint_idx = joint["joint_index"]
                    x, y = int(joint["x"]), int(joint["y"])

                    if joint.get("in_court", False):
                        cv2.circle(frame_display, (x, y), 6, color, -1)
                        cv2.circle(frame_display, (x, y), 8, (255, 255, 255), 1)
                        total_joints_in_court += 1
                    else:
                        cv2.circle(frame_display, (x, y), 3, (128, 128, 128), -1)

                    joint_positions[joint_idx] = (x, y)

            # Draw skeleton
            for edge in pose_edges:
                if edge[0] in joint_positions and edge[1] in joint_positions:
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], color, 3)

            # Player label
            if joint_positions:
                head_pos = joint_positions.get(0, list(joint_positions.values())[0])
                cv2.putText(frame_display, f"Player {human_idx}", (head_pos[0] - 25, head_pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Frame info
        cv2.rectangle(frame_display, (0, 0), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (500, 80), (255, 100, 0), 2)
        cv2.putText(frame_display, f"POSE ESTIMATION - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Players: {len(frame_poses)} | Court Joints: {total_joints_in_court}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage3_batch(args):
    """Process batch for Stage 3: Position tracking"""
    (batch_frames, frame_data_dict, court_template, processor, out_h, out_w) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Frame info header
        cv2.rectangle(frame_display, (0, 0), (480, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (480, 60), (100, 150, 255), 2)
        cv2.putText(frame_display, f"POSITION TRACKING - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Get frame data
        frame_key = str(frame_idx)
        frame_players = frame_data_dict.get(frame_key, {})

        total_ankles = 0
        active_players = len(frame_players)

        # Process players in frame-organized format
        for player_key, player_data in frame_players.items():
            # Extract player ID
            if player_key.startswith('player_'):
                player_id = int(player_key.split('_')[1])
            else:
                player_id = 0

            color = processor.get_color(player_id)

            # Get ankle data
            ankles = player_data.get('ankles', [])
            center_pos = player_data.get('center_position', {})

            # Draw ankle positions
            left_ankle_pos = None
            right_ankle_pos = None

            for ankle_data in ankles:
                ankle_side = ankle_data['ankle_side']
                world_x = ankle_data['world_x']
                world_y = ankle_data['world_y']
                confidence = ankle_data['joint_confidence']

                # Skip invalid positions
                if world_x == 0.0 and world_y == 0.0:
                    continue

                px, py = processor.world_to_court(world_x, world_y)
                total_ankles += 1

                # Color based on ankle side
                if ankle_side == 'left':
                    ankle_color = (0, 220, 255)  # Cyan
                    left_ankle_pos = (px, py)
                else:
                    ankle_color = (255, 200, 0)  # Yellow
                    right_ankle_pos = (px, py)

                # Draw ankle
                radius = max(5, int(7 * confidence))
                cv2.circle(court_img, (px, py), radius + 2, (255, 255, 255), -1)
                cv2.circle(court_img, (px, py), radius, ankle_color, -1)

                # Label
                cv2.putText(court_img, f"P{player_id}{ankle_side[0].upper()}", (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ankle_color, 1)

            # Connect ankles
            if left_ankle_pos and right_ankle_pos:
                cv2.line(court_img, left_ankle_pos, right_ankle_pos, color, 2)

            # Draw center position
            if 'x' in center_pos and 'y' in center_pos:
                if not (center_pos['x'] == 0.0 and center_pos['y'] == 0.0):
                    center_px, center_py = processor.world_to_court(center_pos['x'], center_pos['y'])
                    cv2.circle(court_img, (center_px, center_py), 6, color, 2)
                    cv2.circle(court_img, (center_px, center_py), 2, (255, 255, 255), -1)
                    cv2.putText(court_img, f"P{player_id}", (center_px + 10, center_py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add info panel
        info_text = f"Players: {active_players} | Ankles: {total_ankles}"
        court_img = processor.add_info_panel(court_img, frame_idx, info_text)

        # Update frame info
        cv2.putText(frame_display, f"Players: {active_players} | Ankles: {total_ankles}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

def process_stage4_batch(args):
    """Process batch for Stage 4: Corrected positions comparison"""
    (batch_frames, corrected_data, original_data, court_template, processor, out_h, out_w, trajectory_history) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Frame info
        cv2.rectangle(frame_display, (0, 0), (550, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (550, 60), (255, 0, 150), 2)
        cv2.putText(frame_display, f"CORRECTION COMPARISON - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        corrections_in_frame = 0
        total_displacement = 0.0

        # Handle both frame-organized and legacy formats
        corrected_frame_data = None
        original_frame_data = None

        # Check if data has frame_data structure (new format)
        if 'frame_data' in corrected_data:
            frame_key = str(frame_idx)
            corrected_frame_data = corrected_data['frame_data'].get(frame_key, {})
            original_frame_data = original_data['frame_data'].get(frame_key, {})

            # Process frame-organized data
            for player_key, corrected_player_data in corrected_frame_data.items():
                if not player_key.startswith('player_'):
                    continue

                player_id = int(player_key.split('_')[1])
                color = processor.get_color(player_id)

                # Get original data for comparison
                original_player_data = original_frame_data.get(player_key, {})

                # Process corrected center position
                corrected_center = corrected_player_data.get('center_position', {})
                original_center = original_player_data.get('center_position', {})

                if ('x' in corrected_center and 'y' in corrected_center and
                        'x' in original_center and 'y' in original_center):

                    # Calculate displacement
                    displacement = np.sqrt(
                        (corrected_center['x'] - original_center['x'])**2 +
                        (corrected_center['y'] - original_center['y'])**2
                    )

                    if displacement > 0.01:  # Significant correction
                        corrections_in_frame += 1
                        total_displacement += displacement

                    # Update trajectories
                    if player_id not in trajectory_history:
                        trajectory_history[player_id] = {'original': [], 'corrected': []}

                    corr_px, corr_py = processor.world_to_court(corrected_center['x'], corrected_center['y'])
                    orig_px, orig_py = processor.world_to_court(original_center['x'], original_center['y'])

                    trajectory_history[player_id]['corrected'].append((corr_px, corr_py))
                    trajectory_history[player_id]['original'].append((orig_px, orig_py))

                    # Limit trajectory length
                    for traj_type in ['corrected', 'original']:
                        if len(trajectory_history[player_id][traj_type]) > 50:
                            trajectory_history[player_id][traj_type] = trajectory_history[player_id][traj_type][-50:]

                    # Draw positions
                    # Original position (subtle)
                    if displacement > 0.01:
                        cv2.circle(court_img, (orig_px, orig_py), 4, (120, 120, 120), -1)
                        cv2.line(court_img, (orig_px, orig_py), (corr_px, corr_py), (150, 150, 120), 2)

                    # Corrected position (prominent)
                    cv2.circle(court_img, (corr_px, corr_py), 8, (255, 255, 255), -1)
                    cv2.circle(court_img, (corr_px, corr_py), 6, color, -1)
                    cv2.putText(court_img, f"P{player_id}", (corr_px + 10, corr_py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw trajectories
        for player_id, trajectories in trajectory_history.items():
            color = processor.get_color(player_id)

            # Draw original trajectory (subtle)
            original_traj = trajectories.get('original', [])[-30:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    alpha = i / len(original_traj)
                    if alpha > 0.3:
                        cv2.line(court_img, original_traj[i-1], original_traj[i], (80, 80, 80), 1)

            # Draw corrected trajectory (prominent)
            corrected_traj = trajectories.get('corrected', [])[-30:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    alpha = i / len(corrected_traj)
                    if alpha > 0.4:
                        thickness = max(1, int(2 * alpha))
                        subtle_color = tuple(int(c * 0.7) for c in color)
                        cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], subtle_color, thickness)

        # Correction indicator
        if corrections_in_frame > 0:
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (200, 200, 100), 2)
            cv2.circle(frame_display, (frame.shape[1]-30, 30), 8, (200, 200, 100), -1)
            cv2.putText(frame_display, f"{corrections_in_frame}", (frame.shape[1]-35, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Legend
        if corrections_in_frame > 0:
            legend_x = court_img.shape[1] - 180
            legend_y = court_img.shape[0] - 80

            cv2.rectangle(court_img, (legend_x - 5, legend_y - 15),
                          (court_img.shape[1] - 5, legend_y + 35), (0, 0, 0), -1)
            cv2.rectangle(court_img, (legend_x - 5, legend_y - 15),
                          (court_img.shape[1] - 5, legend_y + 35), (60, 60, 60), 1)

            cv2.circle(court_img, (legend_x + 8, legend_y), 3, (100, 255, 100), -1)
            cv2.putText(court_img, "Corrected", (legend_x + 18, legend_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            cv2.circle(court_img, (legend_x + 8, legend_y + 15), 2, (120, 120, 120), -1)
            cv2.putText(court_img, "Original", (legend_x + 18, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # Info panel
        info_text = f"Corrections: {corrections_in_frame}"
        court_img = processor.add_info_panel(court_img, frame_idx, info_text)

        # Update frame info
        cv2.putText(frame_display, f"Corrections: {corrections_in_frame}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

# Main visualization functions
def visualize_stage1(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 1: Court detection visualization"""
    logging.info("Creating Stage 1 visualization: Court detection")

    processor = VideoProcessor(video_path, num_threads)
    court_points = load_court_csv(data_path)
    corner_points = extract_corner_points(court_points)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, processor.video_info["fps"],
                          (processor.video_info["width"], processor.video_info["height"]))

    frame_count = processor.video_info["frame_count"]
    batch_size = 16

    with ThreadPoolExecutor(max_workers=processor.num_threads) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, court_points, corner_points)
                future = executor.submit(process_stage1_batch, args)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                processed_batch = future.result()
                processed_batch.sort(key=lambda x: x[0])

                for frame_idx, processed_frame in processed_batch:
                    out.write(processed_frame)
                    pbar.update(1)

    out.release()
    logging.info(f"✓ Stage 1 visualization saved to {output_path}")

def visualize_stage2(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 2: Pose estimation visualization"""
    logging.info("Creating Stage 2 visualization: Pose estimation")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)

    # Get court points (try multiple sources)
    court_points = data.get("enlarged_court_points",
                            data.get("all_court_points",
                                     data.get("court_points", {})))

    pose_data = data["pose_data"]
    video_info = data["video_info"]
    poses_by_frame = organize_by_frame(pose_data)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"],
                          (video_info["width"], video_info["height"]))

    frame_count = video_info["frame_count"]
    batch_size = 12

    with ThreadPoolExecutor(max_workers=processor.num_threads) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, poses_by_frame, court_points, processor.pose_edges,
                        processor.colors['players'])
                future = executor.submit(process_stage2_batch, args)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                processed_batch = future.result()
                processed_batch.sort(key=lambda x: x[0])

                for frame_idx, processed_frame in processed_batch:
                    out.write(processed_frame)
                    pbar.update(1)

    out.release()
    logging.info(f"✓ Stage 2 visualization saved to {output_path}")

def visualize_stage3(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 3: Position tracking visualization"""
    logging.info("Creating Stage 3 visualization: Position tracking")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)

    # Handle frame-organized data structure
    if 'frame_data' in data:
        frame_data_dict = data['frame_data']
        video_info = data['video_info']
        tracking_summary = data.get('tracking_summary', {})

        logging.info(f"Loaded position tracking data:")
        logging.info(f"  Frames with data: {tracking_summary.get('frames_with_ankle_data', 0)}")
        logging.info(f"  Total detections: {tracking_summary.get('total_ankle_detections', 0)}")
        logging.info(f"  Method: {tracking_summary.get('primary_method', 'unknown')}")
    else:
        # Legacy format fallback
        logging.warning("Legacy data format detected")
        frame_data_dict = {}
        video_info = data.get("video_info", processor.video_info)

    # Setup video writer with side-by-side layout
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h + 80)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    frame_count = video_info.get("frame_count", processor.video_info["frame_count"])
    batch_size = 8

    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 4)) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, frame_data_dict, processor.court_template.copy(),
                        processor, out_h, out_w)
                future = executor.submit(process_stage3_batch, args)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                processed_batch = future.result()
                processed_batch.sort(key=lambda x: x[0])

                for frame_idx, combined_frame in processed_batch:
                    out.write(combined_frame)
                    pbar.update(1)

                # Periodic cleanup
                if start_frame % (batch_size * 10) == 0:
                    gc.collect()

    out.release()

    # Summary
    total_frames_with_data = len([f for f in frame_data_dict.values() if f])
    if total_frames_with_data > 0:
        logging.info(f"✓ Stage 3 visualization completed")
        logging.info(f"✓ Output: {output_path}")
        logging.info(f"✓ Coverage: {total_frames_with_data}/{frame_count} frames ({total_frames_with_data/frame_count:.1%})")
    else:
        logging.warning("No position tracking data found")

def visualize_stage4(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 4: Corrected positions comparison visualization"""
    logging.info("Creating Stage 4 visualization: Corrected positions comparison")

    processor = VideoProcessor(video_path, num_threads)
    corrected_data = load_json_data(data_path)

    # Load original data for comparison
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)
    original_path = os.path.join(result_dir, "positions.json")

    try:
        original_data = load_json_data(original_path)
    except FileNotFoundError:
        logging.error(f"Original positions file not found: {original_path}")
        return

    # Get video info
    video_info = corrected_data.get("video_info", {})
    if not video_info or "width" not in video_info:
        video_info = processor.video_info

    # Setup video writer
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h + 80)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    frame_count = video_info["frame_count"]
    batch_size = 6

    # Trajectory history (shared across batches)
    trajectory_history = defaultdict(lambda: {'original': [], 'corrected': []})

    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 3)) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, corrected_data, original_data,
                        processor.court_template.copy(), processor, out_h, out_w, trajectory_history)
                future = executor.submit(process_stage4_batch, args)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                processed_batch = future.result()
                processed_batch.sort(key=lambda x: x[0])

                for frame_idx, combined_frame in processed_batch:
                    out.write(combined_frame)
                    pbar.update(1)

                if start_frame % (batch_size * 8) == 0:
                    gc.collect()

    out.release()
    logging.info(f"✓ Stage 4 visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization for complete badminton analysis pipeline")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Stage to visualize (1=court, 2=pose, 3=positions, 4=corrected)")
    parser.add_argument("--data_path", type=str, help="Path to data file (auto-detected if not provided)")
    parser.add_argument("--output", type=str, help="Output video path (auto-generated if not provided)")
    parser.add_argument("--threads", type=int, help="Number of threads (auto-detected if not provided)")

    args = parser.parse_args()

    # Auto-detect optimal thread count
    if args.threads is None:
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        args.threads = min(cpu_count, max(2, int(memory_gb / 3)))

    logging.info(f"Using {args.threads} threads for processing")

    # Auto-detect paths
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    result_dir = os.path.join("results", base_name)

    if args.data_path is None:
        stage_data_files = {
            1: os.path.join(result_dir, "court.csv"),
            2: os.path.join(result_dir, "pose.json"),
            3: os.path.join(result_dir, "positions.json"),
            4: os.path.join(result_dir, "corrected_positions.json")
        }
        args.data_path = stage_data_files[args.stage]

    if args.output is None:
        stage_names = {1: "court", 2: "pose", 3: "positions", 4: "corrected"}
        args.output = os.path.join(result_dir, f"{base_name}_{stage_names[args.stage]}_visualization.mp4")

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        logging.error("Run the corresponding processing stage first")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Performance tracking
    start_time = time.time()
    initial_memory = psutil.virtual_memory().percent

    try:
        # Call appropriate visualization function
        if args.stage == 1:
            visualize_stage1(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 2:
            visualize_stage2(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 3:
            visualize_stage3(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 4:
            visualize_stage4(args.video_path, args.data_path, args.output, args.threads)

        # Performance summary
        end_time = time.time()
        total_time = end_time - start_time
        final_memory = psutil.virtual_memory().percent

        logging.info(f"✓ Visualization complete!")
        logging.info(f"✓ Output: {args.output}")
        logging.info(f"✓ Processing time: {total_time:.2f} seconds")
        logging.info(f"✓ Memory usage: {initial_memory}% → {final_memory}%")

        # Pipeline status check
        logging.info(f"\n📊 Pipeline Status for {base_name}:")
        pipeline_files = {
            "Court Detection": os.path.join(result_dir, "court.csv"),
            "Calibration": os.path.join(result_dir, "calibration.csv"),
            "Pose Estimation": os.path.join(result_dir, "pose.json"),
            "Position Tracking": os.path.join(result_dir, "positions.json"),
            "Jump Correction": os.path.join(result_dir, "corrected_positions.json")
        }

        for stage_name, file_path in pipeline_files.items():
            status = "✅" if os.path.exists(file_path) else "❌"
            logging.info(f"{status} {stage_name}")

    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()