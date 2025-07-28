#!/usr/bin/env python3
"""
Enhanced Universal Visualization Script for Complete Badminton Analysis Pipeline

Fully compatible with all pipeline outputs:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → Enhanced ankle position tracking visualization
- Stage 4: corrected_positions.json → Jump correction comparison visualization

Features:
- Professional, clean court visualization
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
    """Enhanced video processing with professional badminton court visualization"""

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

        # Professional color scheme - cleaner and more subtle
        self.colors = {
            'court_boundary': (255, 255, 255),      # Pure white for main lines
            'service_lines': (220, 220, 220),       # Light gray for service lines
            'center_line': (200, 200, 200),         # Medium gray for center line
            'net': (255, 255, 255),                 # White for net
            'background': (0, 0, 0),            # Professional forest green
            'text': (255, 255, 255),                # White text
            'court_fill': (0, 1, 14),            # Slightly lighter green for court area
            'players': [(255, 87, 34), (33, 150, 243), (156, 39, 176),
                        (255, 193, 7), (76, 175, 80), (244, 67, 54)]
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
        """Create professional, clean badminton court template"""
        # Start with background
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

        # Service dimensions (BWF standard)
        service_line_distance_px = int(1.98 * self.court_scale)  # 1.98m from net
        back_service_distance_px = int(0.76 * self.court_scale)  # 0.76m from back
        side_service_distance_px = int(0.46 * self.court_scale)  # Singles margin (0.46m on each side)

        # Service court boundaries
        front_service_top = center_y - service_line_distance_px
        front_service_bottom = center_y + service_line_distance_px
        back_service_top = top + back_service_distance_px
        back_service_bottom = bottom - back_service_distance_px

        # Singles lines
        singles_left = left + side_service_distance_px
        singles_right = right - side_service_distance_px

        # Fill court area with slightly different color
        cv2.rectangle(court_img, (left, top), (right, bottom), self.colors['court_fill'], -1)

        # Draw court elements with professional styling

        # 1. Main court boundary (thicker for prominence)
        cv2.rectangle(court_img, (left, top), (right, bottom), self.colors['court_boundary'], 4)

        # 2. Singles lines (clean, medium thickness)
        cv2.line(court_img, (singles_left, top), (singles_left, bottom), self.colors['court_boundary'], 3)
        cv2.line(court_img, (singles_right, top), (singles_right, bottom), self.colors['court_boundary'], 3)

        # 3. Net line (prominent, slightly thicker)
        cv2.line(court_img, (left, center_y), (right, center_y), self.colors['net'], 5)

        # 4. Center line - EXTENDS ALL THE WAY ACROSS THE COURT
        cv2.line(court_img, (center_x, top), (center_x, bottom), self.colors['center_line'], 2)

        # 5. Service lines (clean, consistent thickness)
        # Short service lines
        cv2.line(court_img, (singles_left, front_service_top), (singles_right, front_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (singles_left, front_service_bottom), (singles_right, front_service_bottom),
                 self.colors['service_lines'], 2)

        # Long service lines (doubles back boundary for doubles play)
        cv2.line(court_img, (left, back_service_top), (right, back_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (left, back_service_bottom), (right, back_service_bottom),
                 self.colors['service_lines'], 2)

        # 6. Net posts (small, clean circles)
        cv2.circle(court_img, (left, center_y), 5, self.colors['net'], -1)
        cv2.circle(court_img, (right, center_y), 5, self.colors['net'], -1)

        # 7. Clean, minimal labeling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Net label (centered and clean)
        text_size = cv2.getTextSize("NET", font, font_scale, font_thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y - 12

        # Add subtle background for text readability
        cv2.rectangle(court_img, (text_x - 5, text_y - text_size[1] - 3),
                      (text_x + text_size[0] + 5, text_y + 5),
                      self.colors['background'], -1)
        cv2.putText(court_img, "NET", (text_x, text_y), font, font_scale, self.colors['text'], font_thickness)

        # Court dimensions annotations (subtle, in corners)
        dim_font_scale = 0.4
        dim_thickness = 1
        dim_color = (180, 180, 180)

        # Length annotation
        cv2.putText(court_img, "13.4m", (left + 5, bottom - 10),
                    font, dim_font_scale, dim_color, dim_thickness)

        # Width annotation
        cv2.putText(court_img, "6.1m", (right - 40, top + 20),
                    font, dim_font_scale, dim_color, dim_thickness)

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
        """Get color for player with better contrast"""
        return self.colors['players'][player_id % len(self.colors['players'])]

    def world_to_court(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to court image coordinates"""
        px = int((x + self.margin_m) * self.court_scale)
        py = int((y + self.margin_m) * self.court_scale)
        return (px, py)

    def add_info_panel(self, court_img: np.ndarray, frame_idx: int, info: str) -> np.ndarray:
        """Add clean, professional information panel to court visualization"""
        panel_height = 80
        # Use a darker shade of the background for contrast
        panel_color = tuple(int(c * 0.7) for c in self.colors['background'])
        panel = np.full((panel_height, court_img.shape[1], 3), panel_color, dtype=np.uint8)

        # Add subtle border
        cv2.rectangle(panel, (0, 0), (court_img.shape[1]-1, panel_height-1),
                      self.colors['court_boundary'], 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Frame counter (larger, more prominent)
        cv2.putText(panel, f"Frame: {frame_idx:06d}", (15, 30),
                    font, 0.7, self.colors['text'], 2)

        # Additional info (smaller, secondary)
        cv2.putText(panel, info, (15, 55),
                    font, 0.5, (220, 220, 220), 1)

        # Timestamp (right aligned)
        if hasattr(self, 'video_info') and self.video_info["fps"] > 0:
            timestamp = frame_idx / self.video_info["fps"]
            time_text = f"Time: {timestamp:.2f}s"
            text_size = cv2.getTextSize(time_text, font, 0.5, 1)[0]
            cv2.putText(panel, time_text, (court_img.shape[1] - text_size[0] - 15, 30),
                        font, 0.5, (200, 200, 200), 1)

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

# Enhanced visualization functions with professional styling
def draw_court_points(frame: np.ndarray, court_points: Dict, show_labels: bool = True) -> np.ndarray:
    """Draw court points with professional styling"""
    frame_display = frame.copy()

    for point_name, coords in court_points.items():
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])

            # Professional point styling
            if point_name.startswith('P') and point_name[1:].isdigit():
                color = (255, 215, 0)  # Gold for corners
                radius = 6
                border_color = (255, 255, 255)
                border_thickness = 2
            elif 'Net' in point_name:
                color = (255, 20, 147)  # Deep pink for net
                radius = 5
                border_color = (255, 255, 255)
                border_thickness = 2
            else:
                color = (255, 140, 0)  # Dark orange for others
                radius = 4
                border_color = (200, 200, 200)
                border_thickness = 1

            # Draw point with clean border
            cv2.circle(frame_display, (x, y), radius + border_thickness, border_color, -1)
            cv2.circle(frame_display, (x, y), radius, color, -1)

            # Clean labels with background
            if show_labels:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1

                text_size = cv2.getTextSize(point_name, font, font_scale, text_thickness)[0]

                # Background rectangle for readability
                bg_x1, bg_y1 = x + 10, y - text_size[1] - 4
                bg_x2, bg_y2 = x + 14 + text_size[0], y + 4

                cv2.rectangle(frame_display, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(frame_display, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

                cv2.putText(frame_display, point_name, (x + 12, y),
                            font, font_scale, (255, 255, 255), text_thickness)

    return frame_display

def draw_court_polygon(frame: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Draw court polygon overlay with professional styling"""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)

        # Very subtle overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 1, 14))  # Lime green
        cv2.addWeighted(frame, 0.95, overlay, 0.05, 0, frame)

        # Clean border
        cv2.polylines(frame, [pts], True, (255, 255, 255), 3)

    return frame

def process_stage1_batch(args):
    """Process batch for Stage 1: Court detection with professional styling"""
    batch_frames, court_points, corner_points = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_court_points(frame, court_points)
        frame_display = draw_court_polygon(frame_display, corner_points)

        # Professional frame info overlay
        overlay_height = 70
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)

        # Gradient background
        for i in range(overlay_height):
            alpha = i / overlay_height
            overlay[i, :] = (int(30 * alpha), int(50 * alpha), int(30 * alpha))

        # Border
        cv2.rectangle(overlay, (0, 0), (frame.shape[1]-1, overlay_height-1), (255, 255, 255), 2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"COURT DETECTION", (15, 25),
                    font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Frame: {frame_idx:06d} | Points: {len(court_points)}", (15, 50),
                    font, 0.5, (220, 220, 220), 1)

        # Blend overlay
        result = frame_display.copy()
        result[:overlay_height] = cv2.addWeighted(result[:overlay_height], 0.7, overlay, 0.3, 0)

        processed_frames.append((frame_idx, result))

    return processed_frames

def process_stage2_batch(args):
    """Process batch for Stage 2: Pose estimation with professional styling"""
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

            # Draw joints with better styling
            joint_positions = {}
            for joint in joints:
                if joint["confidence"] > 0.5:
                    joint_idx = joint["joint_index"]
                    x, y = int(joint["x"]), int(joint["y"])

                    if joint.get("in_court", False):
                        # Prominent joint in court
                        cv2.circle(frame_display, (x, y), 8, (255, 255, 255), -1)
                        cv2.circle(frame_display, (x, y), 6, color, -1)
                        total_joints_in_court += 1
                    else:
                        # Subtle joint outside court
                        cv2.circle(frame_display, (x, y), 4, (150, 150, 150), -1)
                        cv2.circle(frame_display, (x, y), 2, (100, 100, 100), -1)

                    joint_positions[joint_idx] = (x, y)

            # Draw skeleton with better styling
            for edge in pose_edges:
                if edge[0] in joint_positions and edge[1] in joint_positions:
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], color, 2)

            # Clean player label
            if joint_positions:
                head_pos = joint_positions.get(0, list(joint_positions.values())[0])

                # Background for label
                label_text = f"P{human_idx}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(label_text, font, 0.6, 2)[0]

                bg_x1 = head_pos[0] - 5
                bg_y1 = head_pos[1] - text_size[1] - 10
                bg_x2 = head_pos[0] + text_size[0] + 5
                bg_y2 = head_pos[1] - 5

                cv2.rectangle(frame_display, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(frame_display, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

                cv2.putText(frame_display, label_text, (head_pos[0], head_pos[1] - 8),
                            font, 0.6, (255, 255, 255), 2)

        # Professional info overlay
        overlay_height = 80
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)

        # Gradient background
        for i in range(overlay_height):
            alpha = i / overlay_height
            overlay[i, :] = (int(50 * alpha), int(30 * alpha), int(70 * alpha))

        cv2.rectangle(overlay, (0, 0), (frame.shape[1]-1, overlay_height-1), (255, 100, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"POSE ESTIMATION", (15, 30),
                    font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Frame: {frame_idx:06d} | Players: {len(frame_poses)} | Court Joints: {total_joints_in_court}",
                    (15, 60), font, 0.5, (220, 220, 220), 1)

        # Blend overlay
        result = frame_display.copy()
        result[:overlay_height] = cv2.addWeighted(result[:overlay_height], 0.7, overlay, 0.3, 0)

        processed_frames.append((frame_idx, result))

    return processed_frames

def process_stage3_batch(args):
    """Process batch for Stage 3: Position tracking with professional styling"""
    (batch_frames, frame_data_dict, court_template, processor, out_h, out_w) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Professional frame info header
        overlay_height = 70
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)

        # Gradient
        for i in range(overlay_height):
            alpha = i / overlay_height
            overlay[i, :] = (int(100 * alpha), int(40 * alpha), int(150 * alpha))

        cv2.rectangle(overlay, (0, 0), (frame.shape[1]-1, overlay_height-1), (100, 150, 255), 2)

        # Get frame data
        frame_key = str(frame_idx)
        frame_players = frame_data_dict.get(frame_key, {})

        total_ankles = 0
        active_players = len(frame_players)

        # Process players with professional styling
        for player_key, player_data in frame_players.items():
            if player_key.startswith('player_'):
                player_id = int(player_key.split('_')[1])
            else:
                player_id = 0

            color = processor.get_color(player_id)

            # Get ankle data
            ankles = player_data.get('ankles', [])
            center_pos = player_data.get('center_position', {})

            # Draw ankle positions with professional styling
            left_ankle_pos = None
            right_ankle_pos = None

            for ankle_data in ankles:
                ankle_side = ankle_data['ankle_side']
                world_x = ankle_data['world_x']
                world_y = ankle_data['world_y']
                confidence = ankle_data['joint_confidence']

                if world_x == 0.0 and world_y == 0.0:
                    continue

                px, py = processor.world_to_court(world_x, world_y)
                total_ankles += 1

                # Professional ankle styling
                if ankle_side == 'left':
                    ankle_color = (255, 193, 7)  # Amber
                    left_ankle_pos = (px, py)
                else:
                    ankle_color = (33, 150, 243)  # Blue
                    right_ankle_pos = (px, py)

                # Draw ankle with confidence-based sizing
                radius = max(4, int(6 * confidence))
                cv2.circle(court_img, (px, py), radius + 3, (255, 255, 255), -1)
                cv2.circle(court_img, (px, py), radius, ankle_color, -1)

                # Clean label
                label = f"P{player_id}{ankle_side[0].upper()}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(court_img, label, (px + 10, py - 5),
                            font, 0.4, (255, 255, 255), 1)

            # Connect ankles with styled line
            if left_ankle_pos and right_ankle_pos:
                cv2.line(court_img, left_ankle_pos, right_ankle_pos, color, 3)

            # Draw center position professionally
            if 'x' in center_pos and 'y' in center_pos:
                if not (center_pos['x'] == 0.0 and center_pos['y'] == 0.0):
                    center_px, center_py = processor.world_to_court(center_pos['x'], center_pos['y'])

                    # Center marker
                    cv2.circle(court_img, (center_px, center_py), 10, (255, 255, 255), -1)
                    cv2.circle(court_img, (center_px, center_py), 8, color, -1)
                    cv2.circle(court_img, (center_px, center_py), 3, (255, 255, 255), -1)

                    # Player label
                    cv2.putText(court_img, f"P{player_id}", (center_px + 12, center_py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add professional info panel
        info_text = f"Players: {active_players} | Ankle Detections: {total_ankles}"
        court_img = processor.add_info_panel(court_img, frame_idx, info_text)

        # Frame overlay text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"POSITION TRACKING", (15, 25),
                    font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Frame: {frame_idx:06d} | {info_text}", (15, 55),
                    font, 0.5, (220, 220, 220), 1)

        # Blend overlay
        frame_display[:overlay_height] = cv2.addWeighted(frame_display[:overlay_height], 0.7, overlay, 0.3, 0)

        # Combine frames professionally
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

def process_stage4_batch(args):
    """Process batch for Stage 4: Corrected positions comparison with professional styling"""
    (batch_frames, corrected_data, original_data, court_template, processor, out_h, out_w, trajectory_history) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Professional frame info
        overlay_height = 70
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)

        # Gradient background
        for i in range(overlay_height):
            alpha = i / overlay_height
            overlay[i, :] = (int(150 * alpha), int(30 * alpha), int(100 * alpha))

        cv2.rectangle(overlay, (0, 0), (frame.shape[1]-1, overlay_height-1), (255, 0, 150), 2)

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

            # Process frame-organized data with professional styling
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

                    # Draw positions with professional styling
                    # Original position (subtle)
                    if displacement > 0.01:
                        cv2.circle(court_img, (orig_px, orig_py), 6, (150, 150, 150), -1)
                        cv2.circle(court_img, (orig_px, orig_py), 4, (100, 100, 100), -1)

                        # Correction vector
                        cv2.arrowedLine(court_img, (orig_px, orig_py), (corr_px, corr_py),
                                        (200, 200, 100), 2, tipLength=0.3)

                    # Corrected position (prominent)
                    cv2.circle(court_img, (corr_px, corr_py), 12, (255, 255, 255), -1)
                    cv2.circle(court_img, (corr_px, corr_py), 10, color, -1)
                    cv2.circle(court_img, (corr_px, corr_py), 4, (255, 255, 255), -1)

                    # Player label with background
                    label = f"P{player_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(label, font, 0.5, 2)[0]

                    bg_x1 = corr_px + 15
                    bg_y1 = corr_py - text_size[1] - 5
                    bg_x2 = corr_px + 20 + text_size[0]
                    bg_y2 = corr_py + 5

                    cv2.rectangle(court_img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.rectangle(court_img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

                    cv2.putText(court_img, label, (corr_px + 17, corr_py),
                                font, 0.5, (255, 255, 255), 2)

        # Draw trajectories with professional styling
        for player_id, trajectories in trajectory_history.items():
            color = processor.get_color(player_id)

            # Draw original trajectory (subtle, dashed effect)
            original_traj = trajectories.get('original', [])[-30:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    alpha = i / len(original_traj)
                    if alpha > 0.3 and i % 2 == 0:  # Dashed effect
                        line_color = tuple(int(c * 0.4) for c in color)
                        cv2.line(court_img, original_traj[i-1], original_traj[i], line_color, 1)

            # Draw corrected trajectory (prominent, smooth)
            corrected_traj = trajectories.get('corrected', [])[-30:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    alpha = i / len(corrected_traj)
                    if alpha > 0.4:
                        thickness = max(1, int(3 * alpha))
                        line_alpha = min(1.0, alpha + 0.3)
                        line_color = tuple(int(c * line_alpha) for c in color)
                        cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], line_color, thickness)

        # Professional correction indicator
        if corrections_in_frame > 0:
            # Frame border
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (255, 215, 0), 3)

            # Correction badge
            badge_size = 20
            badge_center = (frame.shape[1] - 40, 40)
            cv2.circle(frame_display, badge_center, badge_size, (255, 215, 0), -1)
            cv2.circle(frame_display, badge_center, badge_size, (255, 255, 255), 2)
            cv2.putText(frame_display, f"{corrections_in_frame}",
                        (badge_center[0] - 8, badge_center[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Professional legend
        if corrections_in_frame > 0:
            legend_x = court_img.shape[1] - 200
            legend_y = court_img.shape[0] - 100

            # Legend background
            cv2.rectangle(court_img, (legend_x - 10, legend_y - 25),
                          (court_img.shape[1] - 10, legend_y + 45), (0, 0, 0), -1)
            cv2.rectangle(court_img, (legend_x - 10, legend_y - 25),
                          (court_img.shape[1] - 10, legend_y + 45), (255, 255, 255), 2)

            # Legend title
            cv2.putText(court_img, "LEGEND", (legend_x - 5, legend_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Legend items
            cv2.circle(court_img, (legend_x + 8, legend_y + 5), 5, (100, 255, 100), -1)
            cv2.putText(court_img, "Corrected", (legend_x + 20, legend_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            cv2.circle(court_img, (legend_x + 8, legend_y + 25), 3, (120, 120, 120), -1)
            cv2.putText(court_img, "Original", (legend_x + 20, legend_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Info panel
        info_text = f"Corrections: {corrections_in_frame} | Avg Displacement: {total_displacement/max(1,corrections_in_frame):.3f}m"
        court_img = processor.add_info_panel(court_img, frame_idx, info_text)

        # Frame overlay text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"CORRECTION ANALYSIS", (15, 25),
                    font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Frame: {frame_idx:06d} | Corrections: {corrections_in_frame}", (15, 55),
                    font, 0.5, (220, 220, 220), 1)

        # Blend overlay
        frame_display[:overlay_height] = cv2.addWeighted(frame_display[:overlay_height], 0.7, overlay, 0.3, 0)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

# Main visualization functions with enhanced professional styling
def visualize_stage1(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 1: Court detection visualization with professional styling"""
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
    """Stage 2: Pose estimation visualization with professional styling"""
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
    """Stage 3: Position tracking visualization with professional styling"""
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
    """Stage 4: Corrected positions comparison visualization with professional styling"""
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
    parser = argparse.ArgumentParser(description="Enhanced professional visualization for complete badminton analysis pipeline")
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

        logging.info(f"✓ Professional visualization complete!")
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