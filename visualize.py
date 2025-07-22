#!/usr/bin/env python3
"""
Enhanced Universal Visualization Script for BPLFMV Structure with Ankle-Only Support

Modified to handle ankle-only tracking data without hip points:
- Stage 3: Enhanced 3D position tracking (ankle-only support)
- Stage 4: Enhanced corrected position comparison (ankle-only support)
- Backward compatible with hip-based data formats
- Clean visualization for simplified tracking results

Handles different data formats:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → 3D position tracking visualization (ankle-only compatible)
- Stage 4: corrected_positions.json → Jump correction comparison (ankle-only compatible)
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
    """Enhanced video processing with sophisticated court design"""

    def __init__(self, video_path: str, num_threads: Optional[int] = None):
        self.video_path = video_path
        self.num_threads = num_threads or min(mp.cpu_count(), 6)

        # Get video info once
        self.video_info = self._get_video_info()

        # Enhanced court parameters (official badminton court dimensions)
        self.court_length_m = 13.4  # Total length
        self.court_width_m = 6.1    # Total width
        self.margin_m = 2.0         # Margin around court
        self.court_scale = 75       # Increased scale for better detail

        # Calculate court image dimensions
        self.court_img_h = int((self.court_length_m + 2 * self.margin_m) * self.court_scale)
        self.court_img_w = int((self.court_width_m + 2 * self.margin_m) * self.court_scale)

        # Enhanced court measurements (in meters)
        self.net_height = 1.55  # Net height at posts
        self.service_line_distance = 1.98  # Distance from net to service line
        self.back_boundary_to_back_service = 0.76  # Distance from back boundary to back service line
        self.side_boundary_to_side_service = 0.46  # Distance from side boundary to side service line (singles)
        self.doubles_side_line = 0.46  # Additional width for doubles

        # Color scheme - minimalistic but clear
        self.colors = {
            'court_boundary': (220, 220, 220),      # Light gray for main boundaries
            'service_lines': (180, 180, 180),       # Medium gray for service lines
            'center_line': (160, 160, 160),         # Darker gray for center line
            'net': (255, 255, 255),                 # White for net
            'background': (40, 50, 45),             # Dark green background
            'text': (200, 200, 200),                # Light gray for text
            'players': [(0, 255, 100), (255, 100, 0), (100, 150, 255),
                        (255, 200, 0), (255, 0, 150), (0, 255, 255)]
        }

        # Create enhanced court template
        self.court_template = self._create_enhanced_court_template()

        # Pose skeleton connections
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

    def _create_enhanced_court_template(self):
        """Create sophisticated badminton court template with all proper lines"""
        # Create court image with dark background
        court_img = np.full((self.court_img_h, self.court_img_w, 3), self.colors['background'], dtype=np.uint8)

        # Calculate key coordinates
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

        # Service line positions
        service_line_distance_px = int(self.service_line_distance * self.court_scale)
        back_service_distance_px = int(self.back_boundary_to_back_service * self.court_scale)
        side_service_distance_px = int(self.side_boundary_to_side_service * self.court_scale)

        # Calculate service court boundaries
        front_service_top = center_y - service_line_distance_px
        front_service_bottom = center_y + service_line_distance_px
        back_service_top = top + back_service_distance_px
        back_service_bottom = bottom - back_service_distance_px

        # Singles side lines (inner lines)
        singles_left = left + side_service_distance_px
        singles_right = right - side_service_distance_px

        # 1. Draw court background area (subtle)
        court_bg_color = tuple(int(c * 1.1) for c in self.colors['background'])
        cv2.rectangle(court_img, (left-2, top-2), (right+2, bottom+2), court_bg_color, -1)

        # 2. Draw main court boundary (doubles court)
        cv2.rectangle(court_img, (left, top), (right, bottom), self.colors['court_boundary'], 3)

        # 3. Draw singles side lines
        cv2.line(court_img, (singles_left, top), (singles_left, bottom), self.colors['court_boundary'], 2)
        cv2.line(court_img, (singles_right, top), (singles_right, bottom), self.colors['court_boundary'], 2)

        # 4. Draw net line (center line across the court)
        cv2.line(court_img, (left, center_y), (right, center_y), self.colors['net'], 4)

        # 5. Draw center line (divides left and right service courts)
        cv2.line(court_img, (center_x, top), (center_x, bottom), self.colors['center_line'], 2)

        # 6. Draw service lines
        # Front service lines
        cv2.line(court_img, (singles_left, front_service_top), (singles_right, front_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (singles_left, front_service_bottom), (singles_right, front_service_bottom),
                 self.colors['service_lines'], 2)

        # Back service lines (for doubles)
        cv2.line(court_img, (left, back_service_top), (right, back_service_top),
                 self.colors['service_lines'], 2)
        cv2.line(court_img, (left, back_service_bottom), (right, back_service_bottom),
                 self.colors['service_lines'], 2)

        # 7. Add net posts
        net_post_radius = 4
        cv2.circle(court_img, (left, center_y), net_post_radius, self.colors['net'], -1)
        cv2.circle(court_img, (right, center_y), net_post_radius, self.colors['net'], -1)

        # 8. Add subtle court markings and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Net label
        cv2.putText(court_img, "NET", (center_x - 20, center_y - 10),
                    font, font_scale, self.colors['text'], font_thickness)

        # Service court labels (subtle)
        label_color = tuple(int(c * 0.8) for c in self.colors['text'])

        # Left service courts
        cv2.putText(court_img, "L1", (center_x - 40, front_service_top - 15),
                    font, 0.4, label_color, 1)
        cv2.putText(court_img, "L2", (center_x - 40, front_service_bottom + 25),
                    font, 0.4, label_color, 1)

        # Right service courts
        cv2.putText(court_img, "R1", (center_x + 20, front_service_top - 15),
                    font, 0.4, label_color, 1)
        cv2.putText(court_img, "R2", (center_x + 20, front_service_bottom + 25),
                    font, 0.4, label_color, 1)

        # Court dimensions (very subtle)
        dim_color = tuple(int(c * 0.6) for c in self.colors['text'])
        cv2.putText(court_img, f"{self.court_length_m}m", (margin_px + 5, bottom + 20),
                    font, 0.35, dim_color, 1)
        cv2.putText(court_img, f"{self.court_width_m}m", (right + 10, margin_px + 15),
                    font, 0.35, dim_color, 1)

        # 9. Add corner markers for reference
        corner_size = 8
        corner_color = self.colors['service_lines']

        # Corner L-shaped markers
        corners = [(left, top), (right, top), (left, bottom), (right, bottom)]
        for x, y in corners:
            cv2.line(court_img, (x-corner_size//2, y), (x+corner_size//2, y), corner_color, 2)
            cv2.line(court_img, (x, y-corner_size//2), (x, y+corner_size//2), corner_color, 2)

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
        """Get color for player with enhanced palette"""
        return self.colors['players'][player_id % len(self.colors['players'])]

    def world_to_court(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to court image coordinates"""
        px = int((x + self.margin_m) * self.court_scale)
        py = int((y + self.margin_m) * self.court_scale)
        return (px, py)

    def add_court_info_panel(self, court_img: np.ndarray, frame_idx: int, num_players: int,
                             additional_info: str = "") -> np.ndarray:
        """Add information panel to court visualization"""
        panel_height = 80
        panel = np.full((panel_height, court_img.shape[1], 3),
                        tuple(int(c * 0.8) for c in self.colors['background']), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = self.colors['text']

        # Frame info
        cv2.putText(panel, f"Frame: {frame_idx:06d}", (10, 25),
                    font, font_scale, text_color, font_thickness)

        # Player count
        cv2.putText(panel, f"Players: {num_players}", (10, 50),
                    font, font_scale, text_color, font_thickness)

        # Additional info
        if additional_info:
            cv2.putText(panel, additional_info, (200, 25),
                        font, font_scale, text_color, font_thickness)

        # Add timestamp
        if hasattr(self, 'video_info') and self.video_info["fps"] > 0:
            timestamp = frame_idx / self.video_info["fps"]
            time_str = f"Time: {timestamp:.2f}s"
            cv2.putText(panel, time_str, (200, 50),
                        font, font_scale, text_color, font_thickness)

        return np.vstack([court_img, panel])

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def organize_by_frame(data_list: List[Dict]) -> Dict[int, List[Dict]]:
    """Organize data by frame index"""
    frame_dict = defaultdict(list)
    for item in data_list:
        frame_idx = item["frame_index"]
        frame_dict[frame_idx].append(item)
    return dict(frame_dict)

def read_court_csv(csv_path: str) -> Dict[str, List[float]]:
    """Read court CSV file"""
    court_points = {}
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            point_name = row['Point']
            x_coord = float(row['X'])
            y_coord = float(row['Y'])
            court_points[point_name] = [x_coord, y_coord]
    return court_points

def extract_corner_points(all_court_points: Dict) -> np.ndarray:
    """Extract corner points"""
    corner_points = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_court_points:
            corner_points.append(all_court_points[point_name])

    if len(corner_points) != 4:
        points_list = list(all_court_points.values())[:4]
        corner_points = points_list

    return np.array(corner_points, dtype=np.float32)

def draw_enhanced_court_points(frame: np.ndarray, all_court_points: Dict, show_labels: bool = True) -> np.ndarray:
    """Draw court points with enhanced styling"""
    frame_display = frame.copy()

    for point_name, coords in all_court_points.items():
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])

            # Enhanced point styling based on type
            if point_name.startswith('P') and point_name[1:].isdigit():
                # Corner points - most important
                color = (0, 255, 255)  # Cyan
                radius = 8
                thickness = -1
                border_color = (255, 255, 255)
                border_thickness = 2
            elif 'NetPole' in point_name or 'Net' in point_name:
                # Net-related points
                color = (255, 100, 255)  # Magenta
                radius = 6
                thickness = -1
                border_color = (255, 255, 255)
                border_thickness = 2
            elif 'Service' in point_name:
                # Service line points
                color = (100, 255, 100)  # Light green
                radius = 5
                thickness = -1
                border_color = (200, 200, 200)
                border_thickness = 1
            else:
                # Other court points
                color = (255, 150, 0)  # Orange
                radius = 4
                thickness = -1
                border_color = (200, 200, 200)
                border_thickness = 1

            # Draw point with border
            cv2.circle(frame_display, (x, y), radius + border_thickness, border_color, -1)
            cv2.circle(frame_display, (x, y), radius, color, thickness)

            # Enhanced labels
            if show_labels:
                label_color = color
                font_scale = 0.5 if point_name.startswith('P') else 0.4
                font_thickness = 2 if point_name.startswith('P') else 1

                # Add text background for better readability
                text_size = cv2.getTextSize(point_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_bg_start = (x + 12, y - text_size[1] - 2)
                text_bg_end = (x + 12 + text_size[0] + 4, y + 4)
                cv2.rectangle(frame_display, text_bg_start, text_bg_end, (0, 0, 0), -1)
                cv2.rectangle(frame_display, text_bg_start, text_bg_end, label_color, 1)

                cv2.putText(frame_display, point_name, (x + 14, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)

    return frame_display

def draw_enhanced_court_polygon(frame: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Draw court polygon with enhanced styling"""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)

        # Draw filled polygon with transparency effect
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(frame, 0.92, overlay, 0.08, 0, frame)

        # Draw border
        cv2.polylines(frame, [pts], True, (0, 255, 0), 4)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

    return frame

def process_stage1_batch(args):
    """Process batch for stage 1 with enhanced court visualization"""
    batch_frames, all_court_points, corner_points = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_enhanced_court_points(frame, all_court_points)
        frame_display = draw_enhanced_court_polygon(frame_display, corner_points)

        # Enhanced frame info
        cv2.rectangle(frame_display, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (400, 60), (0, 255, 0), 2)
        cv2.putText(frame_display, f"COURT DETECTION - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Detected Points: {len(all_court_points)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage2_batch(args):
    """Process batch for stage 2 with enhanced pose visualization"""
    batch_frames, poses_by_frame, all_court_points, pose_edges, colors = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_enhanced_court_points(frame, all_court_points, show_labels=False)

        frame_poses = poses_by_frame.get(frame_idx, [])
        total_joints_in_court = 0

        for pose in frame_poses:
            human_idx = pose["human_index"]
            joints = pose["joints"]
            color = colors[human_idx % len(colors)]

            # Draw joints and collect positions
            joint_positions = {}
            in_court_count = 0

            for joint in joints:
                if joint["confidence"] > 0.5:
                    joint_idx = joint["joint_index"]
                    x, y = int(joint["x"]), int(joint["y"])

                    if joint.get("in_court", False):
                        cv2.circle(frame_display, (x, y), 6, color, -1)
                        cv2.circle(frame_display, (x, y), 8, (255, 255, 255), 1)
                        in_court_count += 1
                        total_joints_in_court += 1
                    else:
                        cv2.circle(frame_display, (x, y), 3, (128, 128, 128), -1)

                    joint_positions[joint_idx] = (x, y)

            # Enhanced skeleton drawing
            for edge in pose_edges:
                if edge[0] in joint_positions and edge[1] in joint_positions:
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], color, 3)
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], (255, 255, 255), 1)

            # Enhanced player label
            if joint_positions:
                head_pos = joint_positions.get(0, list(joint_positions.values())[0])
                label_text = f"Player {human_idx}"

                # Background for label
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_display,
                              (head_pos[0] - 30, head_pos[1] - 35),
                              (head_pos[0] + text_size[0] + 10, head_pos[1] - 10),
                              (0, 0, 0), -1)
                cv2.rectangle(frame_display,
                              (head_pos[0] - 30, head_pos[1] - 35),
                              (head_pos[0] + text_size[0] + 10, head_pos[1] - 10),
                              color, 2)

                cv2.putText(frame_display, label_text, (head_pos[0] - 25, head_pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Enhanced info panel
        cv2.rectangle(frame_display, (0, 0), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (500, 80), (255, 100, 0), 2)
        cv2.putText(frame_display, f"POSE ESTIMATION - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Players: {len(frame_poses)} | Court Joints: {total_joints_in_court}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage3_batch(args):
    """Process batch for stage 3 with enhanced ankle tracking visualization"""
    (batch_frames, frame_data_by_frame, image_points, court_template,
     processor, out_h, out_w) = args
    processed_frames = []

    def world_to_court_fixed(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to court image coordinates (fixed mirroring)"""
        # Fix mirrored coordinate system by flipping X coordinate
        flipped_x = processor.court_width_m - x
        px = int((flipped_x + processor.margin_m) * processor.court_scale)
        py = int((y + processor.margin_m) * processor.court_scale)
        return (px, py)

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Enhanced calibration points (court corners) - only if we have them
        if len(image_points) >= 4:
            for i, point in enumerate(image_points):
                cv2.circle(frame_display, (int(point[0]), int(point[1])), 6, (0, 255, 255), -1)
                cv2.circle(frame_display, (int(point[0]), int(point[1])), 8, (255, 255, 255), 2)
                cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Enhanced frame info header
        cv2.rectangle(frame_display, (0, 0), (480, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (480, 60), (100, 150, 255), 2)
        cv2.putText(frame_display, f"ANKLE TRACKING - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Process ankle positions from new format
        frame_players = frame_data_by_frame.get(str(frame_idx), {})
        active_players = len(frame_players)
        total_ankles = 0

        for player_key, player_data in frame_players.items():
            # Extract player ID from key (e.g., "player_0" -> 0)
            player_id = int(player_key.split('_')[1]) if '_' in player_key else 0
            color = processor.get_color(player_id)

            ankles = player_data.get('ankles', [])
            center_pos = player_data.get('center_position', {})

            # Draw individual ankle positions
            left_ankle_pos = None
            right_ankle_pos = None

            for ankle_data in ankles:
                ankle_side = ankle_data['ankle_side']
                world_x = ankle_data['world_x']
                world_y = ankle_data['world_y']
                confidence = ankle_data['joint_confidence']
                method = ankle_data.get('method', 'homography')

                # Skip if coordinates are 0,0 (invalid detection)
                if world_x == 0.0 and world_y == 0.0:
                    continue

                # Use fixed coordinate transformation
                px, py = world_to_court_fixed(world_x, world_y)
                total_ankles += 1

                # Correct coordinate system - from player's perspective
                if ankle_side == 'right':  # Player's right ankle
                    right_ankle_pos = (px, py)
                    ankle_color = (255, 200, 0)   # Yellow for right ankle
                    marker = "R"
                else:  # ankle_side == 'left' - Player's left ankle
                    left_ankle_pos = (px, py)
                    ankle_color = (0, 220, 255)  # Cyan for left ankle
                    marker = "L"

                # Draw ankle position with confidence-based sizing
                radius = max(5, int(7 * confidence))
                cv2.circle(court_img, (px, py), radius + 2, (255, 255, 255), -1)
                cv2.circle(court_img, (px, py), radius, ankle_color, -1)

                # Enhanced method indicator - subtle green ring
                if method == 'enhanced_homography':
                    cv2.circle(court_img, (px, py), radius + 4, (100, 255, 100), 1)

                # Clean ankle label
                label_text = f"P{player_id}{marker}"
                cv2.putText(court_img, label_text, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ankle_color, 1)

            # Connect left and right ankles with player color line
            if left_ankle_pos and right_ankle_pos:
                cv2.line(court_img, left_ankle_pos, right_ankle_pos, color, 2)

            # Draw player center position (only if not 0,0)
            if ('x' in center_pos and 'y' in center_pos and
                    not (center_pos['x'] == 0.0 and center_pos['y'] == 0.0)):
                center_x = center_pos['x']
                center_y = center_pos['y']
                center_px, center_py = world_to_court_fixed(center_x, center_y)

                # Clean center marker
                cv2.circle(court_img, (center_px, center_py), 6, color, 2)
                cv2.circle(court_img, (center_px, center_py), 2, (255, 255, 255), -1)

                # Player ID label
                cv2.putText(court_img, f"P{player_id}", (center_px + 10, center_py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Enhanced court info panel
        method_info = "Enhanced" if frame_players and any(
            any(ankle.get('method') == 'enhanced_homography' for ankle in player_data.get('ankles', []))
            for player_data in frame_players.values()
        ) else "Basic"

        info_text = f"Method: {method_info} | Ankles: {total_ankles}"
        court_img = processor.add_court_info_panel(court_img, frame_idx, active_players, info_text)

        # Update frame info with ankle count
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
    """Process batch for stage 4 with ankle-only correction visualization"""
    (batch_frames, corrected_by_frame, original_by_frame, image_points,
     court_template, processor, out_h, out_w, trajectory_history) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Enhanced calibration points
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 10, (0, 255, 255), -1)
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 12, (255, 255, 255), 2)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 15, int(point[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Enhanced frame info
        cv2.rectangle(frame_display, (0, 0), (550, 60), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (0, 0), (550, 60), (255, 0, 150), 2)
        cv2.putText(frame_display, f"ANKLE CORRECTION - Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Get frame data
        corrected_frame_players = corrected_by_frame.get(frame_idx, [])
        original_frame_players = original_by_frame.get(frame_idx, [])

        # Track corrections
        corrections_in_frame = 0
        total_displacement = 0.0

        # Update trajectory history for corrected positions
        for pos_data in corrected_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id", 0))

            # Handle both ankle-only and legacy formats
            world_x, world_y = None, None

            if "x" in pos_data and "y" in pos_data:
                # New ankle-only format - flip x for reflection
                world_x = processor.court_width_m - pos_data["x"]
                world_y = pos_data["y"]
            elif "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                # Legacy format - flip x for reflection
                world_x = processor.court_width_m - pos_data["hip_world_X"]
                world_y = pos_data["hip_world_Y"]

            if world_x is not None and world_y is not None:
                px, py = processor.world_to_court(world_x, world_y)

                # Update trajectory
                if player_id not in trajectory_history:
                    trajectory_history[player_id] = {'corrected': [], 'original': []}

                trajectory_history[player_id]['corrected'].append((px, py))
                if len(trajectory_history[player_id]['corrected']) > 50:
                    trajectory_history[player_id]['corrected'] = trajectory_history[player_id]['corrected'][-50:]

        # Process original positions for trajectory and correction detection
        for pos_data in original_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id", 0))

            # Handle both formats for original data
            orig_world_x, orig_world_y = None, None

            if "x" in pos_data and "y" in pos_data:
                # New ankle-only format - flip x for reflection
                orig_world_x = processor.court_width_m - pos_data["x"]
                orig_world_y = pos_data["y"]
            elif "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                # Legacy format - flip x for reflection
                orig_world_x = processor.court_width_m - pos_data["hip_world_X"]
                orig_world_y = pos_data["hip_world_Y"]

            if orig_world_x is not None and orig_world_y is not None:
                orig_px, orig_py = processor.world_to_court(orig_world_x, orig_world_y)

                # Update trajectory
                if player_id not in trajectory_history:
                    trajectory_history[player_id] = {'corrected': [], 'original': []}

                trajectory_history[player_id]['original'].append((orig_px, orig_py))
                if len(trajectory_history[player_id]['original']) > 50:
                    trajectory_history[player_id]['original'] = trajectory_history[player_id]['original'][-50:]

                # Check for corrections using unflipped world coordinates for displacement calculation
                corrected_pos = next((pos for pos in corrected_frame_players
                                      if pos.get("player_id", pos.get("tracked_id", 0)) == player_id), None)

                if corrected_pos:
                    # Calculate displacement using unflipped coordinates
                    if "x" in pos_data and "y" in pos_data:
                        orig_x = pos_data["x"]
                        orig_y = pos_data["y"]
                        corr_x = corrected_pos.get("x", orig_x)
                        corr_y = corrected_pos.get("y", orig_y)
                    elif "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                        orig_x = pos_data["hip_world_X"]
                        orig_y = pos_data["hip_world_Y"]
                        corr_x = corrected_pos.get("hip_world_X", orig_x)
                        corr_y = corrected_pos.get("hip_world_Y", orig_y)
                    else:
                        continue

                    displacement = np.sqrt((orig_x - corr_x)**2 + (orig_y - corr_y)**2)

                    if displacement > 0.01:  # Significant correction
                        corrections_in_frame += 1
                        total_displacement += displacement

        # Subtle correction frame highlighting
        if corrections_in_frame > 0:
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (200, 200, 100), 2)
            cv2.circle(frame_display, (frame.shape[1]-30, 30), 8, (200, 200, 100), -1)
            cv2.putText(frame_display, f"{corrections_in_frame}", (frame.shape[1]-35, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Subtle trajectory visualization
        for player_id, trajectories in trajectory_history.items():
            color = processor.get_color(player_id)

            # Draw original trajectory - subtle gray
            original_traj = trajectories['original'][-50:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    alpha = i / len(original_traj)
                    if alpha > 0.3:
                        thickness = 1
                        fade_color = (80, 80, 80)
                        cv2.line(court_img, original_traj[i-1], original_traj[i], fade_color, thickness)

            # Draw corrected trajectory - subtle but visible
            corrected_traj = trajectories['corrected'][-50:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    alpha = i / len(corrected_traj)
                    if alpha > 0.4:
                        thickness = max(1, int(2 * alpha))
                        subtle_color = tuple(int(c * 0.7) for c in color)
                        cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], subtle_color, thickness)

        # Process corrected player positions
        for pos_data in corrected_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id", 0))
            color = processor.get_color(player_id)

            # Handle both formats
            if "x" in pos_data and "y" in pos_data:
                # New ankle-only format - flip for display
                world_x = processor.court_width_m - pos_data["x"]
                world_y = pos_data["y"]
                px, py = processor.world_to_court(world_x, world_y)

                # Enhanced ankle position marker
                cv2.circle(court_img, (px, py), 10, (255, 255, 255), -1)
                cv2.circle(court_img, (px, py), 8, color, -1)

                # Method indicator for ankle tracking
                method_used = pos_data.get("method", "unknown")
                method_color = {
                    'calibrated_3d': (0, 255, 0),
                    'enhanced_homography': (255, 255, 0),
                    'basic_homography': (255, 100, 0)
                }.get(method_used, (128, 128, 128))

                cv2.circle(court_img, (px - 12, py - 12), 2, method_color, -1)

                # Label
                label_text = f"P{player_id}"
                cv2.putText(court_img, label_text, (px - 15, py - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            elif "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                # Legacy format - flip for display
                world_x = processor.court_width_m - pos_data["hip_world_X"]
                world_y = pos_data["hip_world_Y"]
                px, py = processor.world_to_court(world_x, world_y)

                # Hip position marker
                cv2.circle(court_img, (px, py), 8, (255, 255, 255), -1)
                cv2.circle(court_img, (px, py), 6, color, -1)

                # Label
                label_text = f"P{player_id}"
                cv2.putText(court_img, label_text, (px - 15, py - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw correction indicators
        for pos_data in original_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id", 0))

            corrected_pos = next((pos for pos in corrected_frame_players
                                  if pos.get("player_id", pos.get("tracked_id", 0)) == player_id), None)

            if corrected_pos:
                # Get original position (flipped for visualization)
                if "x" in pos_data and "y" in pos_data:
                    orig_x_flipped = processor.court_width_m - pos_data["x"]
                    orig_y = pos_data["y"]
                    orig_x_unflipped = pos_data["x"]
                elif "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                    orig_x_flipped = processor.court_width_m - pos_data["hip_world_X"]
                    orig_y = pos_data["hip_world_Y"]
                    orig_x_unflipped = pos_data["hip_world_X"]
                else:
                    continue

                orig_px, orig_py = processor.world_to_court(orig_x_flipped, orig_y)

                # Get corrected position (flipped for visualization)
                if "x" in corrected_pos and "y" in corrected_pos:
                    corr_x = corrected_pos["x"]
                    corr_y = corrected_pos["y"]
                elif "hip_world_X" in corrected_pos and "hip_world_Y" in corrected_pos:
                    corr_x = corrected_pos["hip_world_X"]
                    corr_y = corrected_pos["hip_world_Y"]
                else:
                    continue

                corr_x_flipped = processor.court_width_m - corr_x
                corr_px, corr_py = processor.world_to_court(corr_x_flipped, corr_y)

                # Calculate displacement using unflipped coordinates
                displacement = np.sqrt((orig_x_unflipped - corr_x)**2 + (orig_y - corr_y)**2)

                if displacement > 0.01:
                    # Subtle original position
                    cv2.circle(court_img, (orig_px, orig_py), 4, (120, 120, 120), -1)

                    # Subtle correction line
                    cv2.line(court_img, (orig_px, orig_py), (corr_px, corr_py), (150, 150, 120), 1)

                    # Small dot at corrected position
                    cv2.circle(court_img, (corr_px, corr_py), 2, (180, 180, 150), -1)

        # Compact legend for corrections
        if corrections_in_frame > 0:
            legend_x = court_img.shape[1] - 180
            legend_y = court_img.shape[0] - 80

            cv2.rectangle(court_img, (legend_x - 5, legend_y - 15),
                          (court_img.shape[1] - 5, legend_y + 35), (0, 0, 0), -1)
            cv2.rectangle(court_img, (legend_x - 5, legend_y - 15),
                          (court_img.shape[1] - 5, legend_y + 35), (60, 60, 60), 1)

            cv2.circle(court_img, (legend_x + 8, legend_y), 3, (100, 255, 100), -1)
            cv2.putText(court_img, "Current", (legend_x + 18, legend_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            cv2.circle(court_img, (legend_x + 8, legend_y + 15), 2, (120, 120, 120), -1)
            cv2.putText(court_img, "Original", (legend_x + 18, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # Court info panel with correction information
        info_text = ""
        if corrections_in_frame > 0:
            info_text = f"Corrections: {corrections_in_frame}"

        court_img = processor.add_court_info_panel(court_img, frame_idx, len(corrected_frame_players), info_text)

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

def visualize_stage1(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 1 visualization: Enhanced court detection"""
    logging.info("Creating Stage 1 visualization: Enhanced court detection")

    processor = VideoProcessor(video_path, num_threads)
    all_court_points = read_court_csv(data_path)
    corner_points = extract_corner_points(all_court_points)

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
                args = (batch_frames, all_court_points, corner_points)
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
    logging.info(f"✓ Enhanced Stage 1 visualization saved to {output_path}")

def visualize_stage2(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 2 visualization: Enhanced pose estimation"""
    logging.info("Creating Stage 2 visualization: Enhanced pose estimation")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)

    # Use enlarged court points if available, otherwise fall back to all_court_points or court_points
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
    logging.info(f"✓ Enhanced Stage 2 visualization saved to {output_path}")

def visualize_stage3(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 3 visualization: Enhanced ankle position tracking"""
    logging.info("Creating Stage 3 visualization: Enhanced ankle position tracking")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)

    # Handle the new ankle tracking format
    if 'frame_data' in data:
        frame_data_dict = data['frame_data']
        video_info = data['video_info']
        tracking_summary = data.get('tracking_summary', {})

        # Log tracking statistics
        logging.info(f"Loaded ankle tracking data:")
        logging.info(f"  Frames with data: {tracking_summary.get('frames_with_ankle_data', 0)}")
        logging.info(f"  Total ankle detections: {tracking_summary.get('total_ankle_detections', 0)}")
        logging.info(f"  Method: {tracking_summary.get('primary_method', 'unknown')}")

        # Try to load court points from calibration file
        court_points = {}
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        result_dir = os.path.join("results", base_name)
        calibration_file = os.path.join(result_dir, f"{base_name}_calibration_complete.csv")

        if os.path.exists(calibration_file):
            logging.info(f"Loading court points from calibration file...")
            try:
                import csv
                with open(calibration_file, 'r') as file:
                    csv_reader = csv.reader(file)
                    in_court_points_section = False

                    for row in csv_reader:
                        if not row or row[0].startswith('#'):
                            continue

                        key = row[0].strip()

                        if key == 'Point':
                            in_court_points_section = True
                            continue
                        elif in_court_points_section and len(row) >= 3:
                            point_name = row[0].strip()
                            try:
                                x_coord = float(row[1])
                                y_coord = float(row[2])
                                court_points[point_name] = [x_coord, y_coord]
                            except (ValueError, IndexError):
                                continue
                logging.info(f"  Loaded {len(court_points)} court points")
            except Exception as e:
                logging.warning(f"Could not load court points from calibration: {e}")

        # Create minimal court points if none found (just for visualization)
        if not court_points:
            logging.info("No court points found - visualization will show ankle tracking only")
            court_points = {}

    else:
        # Legacy format fallback
        logging.warning("Legacy data format detected - limited compatibility")
        court_points = data.get("court_points", {})
        video_info = data["video_info"]
        frame_data_dict = {}

        # Convert if possible
        if "player_positions" in data:
            for pos_data in data["player_positions"]:
                frame_idx = str(pos_data["frame_index"])
                if frame_idx not in frame_data_dict:
                    frame_data_dict[frame_idx] = {}

    # Extract corner points for display (only if we have 4+ points)
    image_points = []
    if len(court_points) >= 4:
        image_points = extract_corner_points(court_points)

    # Use actual video info from the file or get fresh info
    if not video_info.get("width") or not video_info.get("height"):
        logging.info("Getting video information...")
        processor_video_info = processor._get_video_info()
        video_info.update(processor_video_info)

    # Setup video writer with side-by-side layout
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h + 80)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    # Use actual frame count or estimate
    frame_count = video_info.get("frame_count", 0)
    if frame_count == 0:
        # Get actual frame count from video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        logging.info(f"Detected {frame_count} frames in video")

    batch_size = 8

    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 4)) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, frame_data_dict, image_points,
                        processor.court_template.copy(), processor, out_h, out_w)
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

    # Print comprehensive summary
    total_frames_with_data = len([f for f in frame_data_dict.values() if f])
    total_ankle_detections = sum(
        len(player_data['ankles'])
        for frame_data in frame_data_dict.values()
        for player_data in frame_data.values()
    )

    if total_frames_with_data > 0:
        avg_ankles_per_frame = total_ankle_detections / total_frames_with_data
        logging.info(f"✓ Enhanced Stage 3 ankle tracking visualization completed")
        logging.info(f"✓ Output saved to: {output_path}")
        logging.info(f"✓ Processed {total_frames_with_data}/{frame_count} frames ({total_frames_with_data/frame_count:.1%} coverage)")
        logging.info(f"✓ Total ankle detections: {total_ankle_detections}")
        logging.info(f"✓ Average ankles per frame: {avg_ankles_per_frame:.1f}")
    else:
        logging.warning("No ankle tracking data found in frames")
        logging.info(f"✓ Video visualization saved to: {output_path} (no tracking data overlay)")

def visualize_stage4(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 4 visualization: Enhanced ankle-only corrected position comparison"""
    logging.info("Creating Stage 4 visualization: Enhanced ankle-only corrected position comparison")

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

    # Organize data
    court_points = corrected_data.get("court_points", {})

    # Get video info from either data file or processor
    video_info = corrected_data.get("video_info", {})
    if not video_info or "width" not in video_info:
        video_info = processor.video_info

    corrected_positions = corrected_data["player_positions"]
    original_positions = original_data["player_positions"]

    corrected_by_frame = organize_by_frame(corrected_positions)
    original_by_frame = organize_by_frame(original_positions)
    image_points = extract_corner_points(court_points)

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
                args = (batch_frames, corrected_by_frame, original_by_frame, image_points,
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
    logging.info(f"✓ Enhanced Stage 4 ankle-only visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create enhanced visualizations for BPLFMV pipeline stages with ankle-only support")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Stage to visualize (1=court, 2=pose, 3=positions, 4=corrected)")
    parser.add_argument("--data_path", type=str, help="Path to data file (auto-detected if not provided)")
    parser.add_argument("--output", type=str, help="Output video path (auto-generated if not provided)")
    parser.add_argument("--threads", type=int, help="Number of threads to use (auto-detected if not provided)")

    args = parser.parse_args()

    # Auto-detect optimal thread count
    if args.threads is None:
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        args.threads = min(cpu_count, max(2, int(memory_gb / 3)))

    logging.info(f"Using {args.threads} threads for enhanced processing")

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
        stage_names = {1: "court_enhanced", 2: "pose_enhanced", 3: "ankle_enhanced", 4: "corrected_ankle_enhanced"}
        args.output = os.path.join(result_dir, f"{base_name}_{stage_names[args.stage]}_viz.mp4")

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

        logging.info(f"✓ Enhanced ankle-only visualization complete!")
        logging.info(f"✓ Output saved to: {args.output}")
        logging.info(f"✓ Total processing time: {total_time:.2f} seconds")
        logging.info(f"✓ Memory usage change: {initial_memory}% → {final_memory}%")

    except Exception as e:
        logging.error(f"Enhanced visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()