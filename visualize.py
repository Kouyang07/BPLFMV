#!/usr/bin/env python3
"""
Efficient Universal Visualization Script for BPLFMV Structure

Simplified optimizations:
1. Reduced thread overhead with smart batching
2. Removed unnecessary caching layers
3. Streamlined memory management
4. Modified stage 4 to use player_id instead of tracked_id
5. Simplified coordinate transformations
6. Reduced redundant operations

Handles different data formats:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → 3D position tracking visualization
- Stage 4: corrected_positions.json → Jump correction comparison (using player_id)
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
    """Simplified video processing with efficient batching"""

    def __init__(self, video_path: str, num_threads: Optional[int] = None):
        self.video_path = video_path
        self.num_threads = num_threads or min(mp.cpu_count(), 6)  # Conservative thread count

        # Get video info once
        self.video_info = self._get_video_info()

        # Court visualization parameters
        self.court_length_m = 13.4
        self.court_width_m = 6.1
        self.margin_m = 2.0
        self.court_scale = 65
        self.court_img_h = int((self.court_length_m + 2 * self.margin_m) * self.court_scale)
        self.court_img_w = int((self.court_width_m + 2 * self.margin_m) * self.court_scale)

        # Create court template once
        self.court_template = self._create_court_template()

        # Color palettes
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

        # Pose skeleton
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
        """Create court template once"""
        court_img = np.zeros((self.court_img_h, self.court_img_w, 3), dtype=np.uint8)

        # Calculate coordinates
        left = int(self.margin_m * self.court_scale)
        right = int((self.margin_m + self.court_width_m) * self.court_scale)
        top = int(self.margin_m * self.court_scale)
        bottom = int((self.margin_m + self.court_length_m) * self.court_scale)
        center_x = int((self.margin_m + self.court_width_m / 2) * self.court_scale)
        net_y = int((self.margin_m + self.court_length_m / 2) * self.court_scale)

        # Draw court lines
        cv2.rectangle(court_img, (left, top), (right, bottom), (255, 255, 255), 2)
        cv2.line(court_img, (left, net_y), (right, net_y), (255, 255, 255), 2)
        cv2.line(court_img, (center_x, top), (center_x, bottom), (150, 150, 150), 1)

        # Service lines
        service_front = int((self.margin_m + self.court_length_m/2 - 1.98) * self.court_scale)
        service_back = int((self.margin_m + self.court_length_m/2 + 1.98) * self.court_scale)
        cv2.line(court_img, (left, service_front), (right, service_front), (150, 150, 150), 1)
        cv2.line(court_img, (left, service_back), (right, service_back), (150, 150, 150), 1)

        # Add labels
        cv2.putText(court_img, "NET", (center_x - 20, net_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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
        return self.colors[player_id % len(self.colors)]

    def world_to_court(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to court image coordinates"""
        px = int((x + self.margin_m) * self.court_scale)
        py = int((y + self.margin_m) * self.court_scale)
        return (px, py)

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

def draw_court_points(frame: np.ndarray, all_court_points: Dict, show_labels: bool = True) -> np.ndarray:
    """Draw court points on frame"""
    frame_display = frame.copy()

    for point_name, coords in all_court_points.items():
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])

            if point_name.startswith('P') and point_name[1:].isdigit():
                color = (0, 255, 255)
                radius = 6
            elif 'NetPole' in point_name:
                color = (255, 0, 255)
                radius = 8
            else:
                color = (0, 255, 0)
                radius = 4

            cv2.circle(frame_display, (x, y), radius, color, -1)
            cv2.circle(frame_display, (x, y), radius + 2, (255, 255, 255), 2)

            if show_labels:
                cv2.putText(frame_display, point_name, (x + 10, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame_display

def draw_court_polygon(frame: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Draw court polygon"""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
    return frame

def process_stage1_batch(args):
    """Process batch for stage 1"""
    batch_frames, all_court_points, corner_points = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_court_points(frame, all_court_points)
        frame_display = draw_court_polygon(frame_display, corner_points)
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage2_batch(args):
    """Process batch for stage 2"""
    batch_frames, poses_by_frame, all_court_points, pose_edges, colors = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = draw_court_points(frame, all_court_points, show_labels=False)

        frame_poses = poses_by_frame.get(frame_idx, [])

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
                        cv2.circle(frame_display, (x, y), 5, color, -1)
                        in_court_count += 1
                    else:
                        cv2.circle(frame_display, (x, y), 3, (128, 128, 128), -1)

                    joint_positions[joint_idx] = (x, y)

            # Draw skeleton
            for edge in pose_edges:
                if edge[0] in joint_positions and edge[1] in joint_positions:
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], color, 2)

            # Add player label
            if joint_positions:
                head_pos = joint_positions.get(0, list(joint_positions.values())[0])
                cv2.putText(frame_display, f"Player {human_idx} ({in_court_count})",
                            (head_pos[0] - 20, head_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame_display, f"Frame: {frame_idx} | Players: {len(frame_poses)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def process_stage3_batch(args):
    """Process batch for stage 3"""
    (batch_frames, positions_by_frame, image_points, court_template,
     processor, out_h, out_w) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Draw calibration points
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process positions
        frame_players = positions_by_frame.get(frame_idx, [])

        for pos_data in frame_players:
            tracked_id = pos_data["tracked_id"]
            color = processor.get_color(tracked_id)

            # Draw different position types
            position_types = [
                ("hip_world_X", "hip_world_Y", "Hip", color, 5),
                ("left_hip_world_X", "left_hip_world_Y", "LH", (100, 100, 255), 3),
                ("right_hip_world_X", "right_hip_world_Y", "RH", (255, 100, 100), 3),
                ("left_ankle_world_X", "left_ankle_world_Y", "LA", (0, 255, 255), 4),
                ("right_ankle_world_X", "right_ankle_world_Y", "RA", (255, 255, 0), 4)
            ]

            for x_key, y_key, label, pos_color, radius in position_types:
                if x_key in pos_data and y_key in pos_data:
                    world_x = pos_data[x_key]
                    world_y = pos_data[y_key]
                    px, py = processor.world_to_court(world_x, world_y)

                    cv2.circle(court_img, (px, py), radius, pos_color, -1)
                    cv2.putText(court_img, f"P{tracked_id}-{label}", (px - 15, py - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, pos_color, 1)

        # Add statistics
        cv2.putText(court_img, f"Players: {len(frame_players)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(court_img, f"Frame: {frame_idx}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

def process_stage4_batch(args):
    """Process batch for stage 4 - now using player_id"""
    (batch_frames, corrected_by_frame, original_by_frame, image_points,
     court_template, processor, out_h, out_w, trajectory_history) = args
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Draw calibration points
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Get frame data
        corrected_frame_players = corrected_by_frame.get(frame_idx, [])
        original_frame_players = original_by_frame.get(frame_idx, [])

        # Check for corrections in this frame
        has_corrections = False

        # Update trajectory history and check for corrections
        for pos_data in corrected_frame_players:
            # Use player_id instead of tracked_id
            player_id = pos_data.get("player_id", pos_data.get("tracked_id"))  # Fallback for compatibility

            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = processor.world_to_court(hip_x, hip_y)

                # Update trajectory
                if player_id not in trajectory_history:
                    trajectory_history[player_id] = {'corrected': [], 'original': []}

                trajectory_history[player_id]['corrected'].append(hip_px)
                if len(trajectory_history[player_id]['corrected']) > 30:
                    trajectory_history[player_id]['corrected'] = trajectory_history[player_id]['corrected'][-30:]

        for pos_data in original_frame_players:
            # Use player_id instead of tracked_id
            player_id = pos_data.get("player_id", pos_data.get("tracked_id"))  # Fallback for compatibility

            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = processor.world_to_court(hip_x, hip_y)

                # Update trajectory
                if player_id not in trajectory_history:
                    trajectory_history[player_id] = {'corrected': [], 'original': []}

                trajectory_history[player_id]['original'].append(hip_px)
                if len(trajectory_history[player_id]['original']) > 30:
                    trajectory_history[player_id]['original'] = trajectory_history[player_id]['original'][-30:]

                # Check if this position was corrected
                corrected_pos = next((pos for pos in corrected_frame_players
                                      if pos.get("player_id", pos.get("tracked_id")) == player_id), None)

                if corrected_pos:
                    corr_x = corrected_pos.get("hip_world_X", hip_x)
                    corr_y = corrected_pos.get("hip_world_Y", hip_y)
                    if abs(hip_x - corr_x) > 0.01 or abs(hip_y - corr_y) > 0.01:
                        has_corrections = True

        # Highlight correction frames
        if has_corrections:
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 255), 8)
            cv2.putText(frame_display, "CORRECTED FRAME", (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Draw trajectories
        for player_id, trajectories in trajectory_history.items():
            color = processor.get_color(player_id)

            # Draw original trajectory (gray)
            original_traj = trajectories['original'][-30:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    cv2.line(court_img, original_traj[i-1], original_traj[i], (128, 128, 128), 2)

            # Draw corrected trajectory (colored)
            corrected_traj = trajectories['corrected'][-30:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], color, 3)

        # Draw current corrected positions
        for pos_data in corrected_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id"))
            color = processor.get_color(player_id)

            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = processor.world_to_court(hip_x, hip_y)
                cv2.circle(court_img, hip_px, 8, color, -1)
                cv2.circle(court_img, hip_px, 10, (255, 255, 255), 2)
                cv2.putText(court_img, f"P{player_id}-C", (hip_px[0] - 20, hip_px[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw original positions if different from corrected
        for pos_data in original_frame_players:
            player_id = pos_data.get("player_id", pos_data.get("tracked_id"))

            corrected_pos = next((pos for pos in corrected_frame_players
                                  if pos.get("player_id", pos.get("tracked_id")) == player_id), None)

            if corrected_pos and "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                orig_x = pos_data["hip_world_X"]
                orig_y = pos_data["hip_world_Y"]
                corr_x = corrected_pos.get("hip_world_X", orig_x)
                corr_y = corrected_pos.get("hip_world_Y", orig_y)

                if abs(orig_x - corr_x) > 0.01 or abs(orig_y - corr_y) > 0.01:
                    orig_px = processor.world_to_court(orig_x, orig_y)
                    corr_px = processor.world_to_court(corr_x, corr_y)

                    cv2.circle(court_img, orig_px, 4, (128, 128, 128), -1)
                    cv2.line(court_img, orig_px, corr_px, (255, 255, 255), 1)

        # Add legend
        legend_x = court_img.shape[1] - 200
        cv2.putText(court_img, "Corrected Position", (legend_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(court_img, "Original Position", (legend_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        cv2.putText(court_img, f"Frame: {frame_idx}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

def visualize_stage1(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 1 visualization: Court detection"""
    logging.info("Creating Stage 1 visualization: Court detection")

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
    logging.info(f"✓ Stage 1 visualization saved to {output_path}")

def visualize_stage2(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 2 visualization: Pose estimation"""
    logging.info("Creating Stage 2 visualization: Pose estimation")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)
    all_court_points = data.get("all_court_points", data.get("court_points", {}))
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
                args = (batch_frames, poses_by_frame, all_court_points, processor.pose_edges, processor.colors)
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
    """Stage 3 visualization: 3D position tracking"""
    logging.info("Creating Stage 3 visualization: 3D position tracking")

    processor = VideoProcessor(video_path, num_threads)
    data = load_json_data(data_path)
    court_points = data.get("court_points", {})
    video_info = data["video_info"]
    player_positions = data["player_positions"]

    positions_by_frame = organize_by_frame(player_positions)
    image_points = extract_corner_points(court_points)

    # Setup video writer with side-by-side layout
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    frame_count = video_info["frame_count"]
    batch_size = 8

    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 4)) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                args = (batch_frames, positions_by_frame, image_points,
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
    logging.info(f"✓ Stage 3 visualization saved to {output_path}")

def visualize_stage4(video_path: str, data_path: str, output_path: str, num_threads: int = None):
    """Stage 4 visualization: Corrected position comparison using player_id"""
    logging.info("Creating Stage 4 visualization: Corrected position comparison (using player_id)")

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
    video_info = corrected_data["video_info"]
    corrected_positions = corrected_data["player_positions"]
    original_positions = original_data["player_positions"]

    corrected_by_frame = organize_by_frame(corrected_positions)
    original_by_frame = organize_by_frame(original_positions)
    image_points = extract_corner_points(court_points)

    # Setup video writer
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h)

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

                # Periodic cleanup
                if start_frame % (batch_size * 8) == 0:
                    gc.collect()

    out.release()
    logging.info(f"✓ Stage 4 visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create efficient visualizations for BPLFMV pipeline stages")
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
        args.threads = min(cpu_count, max(2, int(memory_gb / 3)))  # Conservative

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

        logging.info(f"✓ Visualization complete!")
        logging.info(f"✓ Output saved to: {args.output}")
        logging.info(f"✓ Total processing time: {total_time:.2f} seconds")
        logging.info(f"✓ Memory usage change: {initial_memory}% → {final_memory}%")

    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()