#!/usr/bin/env python3
"""
Universal Visualization Script for BPLFMV Structure

Handles different data formats:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → 3D position tracking visualization
- Stage 4: corrected_positions.json → Jump correction comparison
"""

import sys
import os
from collections import defaultdict

import numpy as np
import cv2
import json
import csv
import argparse
import logging
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_court(court_image: np.ndarray, court_scale: int) -> None:
    """
    Draws the detailed outline of the badminton court with additional lines.

    Args:
        court_image (np.ndarray): Image on which the court will be drawn.
        court_scale (int): Scaling factor for the court dimensions.
    """
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0

    # Calculate court dimensions in pixels
    left = int(margin_m * court_scale)
    right = int((margin_m + court_width_m) * court_scale)
    top = int(margin_m * court_scale)
    bottom = int((margin_m + court_length_m) * court_scale)

    # Main court outline
    cv2.rectangle(court_image, (left, top), (right, bottom), (255, 255, 255), 2)

    # Center line (vertical)
    center_x = int((margin_m + court_width_m / 2) * court_scale)
    cv2.line(court_image, (center_x, top), (center_x, bottom), (150, 150, 150), 1)

    # Service lines - 1.98m from the net on each side
    service_line_top = int((margin_m + court_length_m/2 - 1.98) * court_scale)
    service_line_bottom = int((margin_m + court_length_m/2 + 1.98) * court_scale)

    # Net line
    net_y = int((margin_m + court_length_m / 2) * court_scale)
    cv2.line(court_image, (left, net_y), (right, net_y), (255, 255, 255), 2)
    cv2.putText(court_image, "NET", (center_x - 20, net_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.line(court_image, (left, service_line_top), (right, service_line_top), (150, 150, 150), 2)
    cv2.putText(court_image, "SERVICE LINE", (left + 10, service_line_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.line(court_image, (left, service_line_bottom), (right, service_line_bottom), (150, 150, 150), 2)
    cv2.putText(court_image, "SERVICE LINE", (left + 10, service_line_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Back lines - 0.76m from each end of the court
    back_line_top = int((margin_m + 0.76) * court_scale)
    back_line_bottom = int((margin_m + court_length_m - 0.76) * court_scale)

    cv2.line(court_image, (left, back_line_top), (right, back_line_top), (150, 150, 150), 2)
    cv2.putText(court_image, "BACK LINE", (left + 10, back_line_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.line(court_image, (left, back_line_bottom), (right, back_line_bottom), (150, 150, 150), 2)
    cv2.putText(court_image, "BACK LINE", (left + 10, back_line_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Singles sidelines - 0.46m from each side
    singles_left = int((margin_m + 0.46) * court_scale)
    singles_right = int((margin_m + court_width_m - 0.46) * court_scale)

    cv2.line(court_image, (singles_left, top), (singles_left, bottom), (150, 150, 150), 1)
    cv2.line(court_image, (singles_right, top), (singles_right, bottom), (150, 150, 150), 1)

    # Draw intersections of service courts
    cv2.line(court_image, (center_x, service_line_top), (center_x, service_line_bottom), (150, 150, 150), 1)

    # Court orientation labels
    cv2.putText(court_image, "FRONT", (center_x - 30, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(court_image, "BACK", (center_x - 25, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def world_to_court(X: float, Y: float, court_img_w: int, court_img_h: int,
                   court_scale: int, margin_m: float, court_length_m: float) -> Tuple[int, int]:
    """
    Converts (X, Y) in world meters into pixel coordinates on the overhead court image.
    Uses simple coordinate mapping without any transformations.

    Args:
        X (float): World x-coordinate (meters).
        Y (float): World y-coordinate (meters).
        court_img_w (int): Width of court image in pixels.
        court_img_h (int): Height of court image in pixels.
        court_scale (int): Scaling factor for the court dimensions.
        margin_m (float): Margin in meters.
        court_length_m (float): Court length in meters.

    Returns:
        Tuple[int, int]: Pixel coordinates (x, y) in the overhead view.
    """
    # Simple mapping from world coordinates to pixel coordinates
    px = int((X + margin_m) * court_scale)
    py = int((Y + margin_m) * court_scale)
    return (px, py)


def read_court_csv(csv_path):
    """Read court coordinates from CSV file."""
    court_points = {}

    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            point_name = row['Point']
            x_coord = float(row['X'])
            y_coord = float(row['Y'])
            court_points[point_name] = [x_coord, y_coord]

    return court_points


def get_video_info(video_path):
    """Extract basic video information."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count
    }


def extract_corner_points(all_court_points):
    """Extract P1-P4 corner points for polygon drawing."""
    corner_points = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_court_points:
            corner_points.append(all_court_points[point_name])

    if len(corner_points) != 4:
        # Use first 4 points as fallback
        points_list = list(all_court_points.values())[:4]
        corner_points = points_list

    return np.array(corner_points, dtype=np.float32)


def draw_all_court_points(frame, all_court_points, show_labels=True):
    """Draw all detected court points with different colors."""
    if not all_court_points:
        return frame

    frame_display = frame.copy()

    # Color scheme for different point types
    colors = {
        'P': (0, 255, 255),        # Yellow for main points
        'NetPole': (255, 0, 255),  # Magenta for net poles
        'default': (0, 255, 0)     # Green for others
    }

    for point_name, coords in all_court_points.items():
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])

            # Choose color based on point type
            if point_name.startswith('P') and point_name[1:].isdigit():
                color = colors['P']
                radius = 6
            elif 'NetPole' in point_name:
                color = colors['NetPole']
                radius = 8
            else:
                color = colors['default']
                radius = 4

            # Draw point
            cv2.circle(frame_display, (x, y), radius, color, -1)
            cv2.circle(frame_display, (x, y), radius + 2, (255, 255, 255), 2)  # White border

            # Draw label
            if show_labels:
                font_scale = 0.5 if len(point_name) <= 3 else 0.4
                cv2.putText(frame_display, point_name, (x + 10, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame_display


def draw_court_polygon(frame, corner_points):
    """Draw court boundary polygon."""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
    return frame


def extract_corner_points_from_data(court_points: Dict) -> np.ndarray:
    """Extract P1-P4 corner points for calibration visualization."""
    corner_points = []
    for point in ['P1', 'P2', 'P3', 'P4']:
        if point in court_points:
            corner_points.append(court_points[point])

    if len(corner_points) != 4:
        # Use available points or fallback
        all_points = list(court_points.values())
        corner_points = all_points[:4]

    return np.array(corner_points, dtype=np.float32)


def organize_positions_by_frame(player_positions: List[Dict]) -> Dict[int, List[Dict]]:
    """Organize position data by frame index."""
    positions_by_frame = {}

    for pos_data in player_positions:
        frame_idx = pos_data["frame_index"]
        if frame_idx not in positions_by_frame:
            positions_by_frame[frame_idx] = []
        positions_by_frame[frame_idx].append(pos_data)

    return positions_by_frame


def visualize_stage1_court_detection(video_path: str, data_path: str, output_path: str) -> None:
    """Visualize court detection from court.csv."""
    logging.info("Creating Stage 1 visualization: Court detection")

    # Read court points from CSV
    all_court_points = read_court_csv(data_path)
    corner_points = extract_corner_points(all_court_points)

    # Get video info
    video_info = get_video_info(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    fps = video_info["fps"]
    w = video_info["width"]
    h = video_info["height"]
    frame_count = video_info["frame_count"]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()

        # Draw all court points
        frame_display = draw_all_court_points(frame_display, all_court_points, show_labels=True)

        # Draw court boundary
        frame_display = draw_court_polygon(frame_display, corner_points)

        # Add frame info and statistics
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Court points detected: {len(all_court_points)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add legend
        legend_y = h - 120
        cv2.putText(frame_display, "LEGEND:", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame_display, (20, legend_y + 25), 6, (0, 255, 255), -1)
        cv2.putText(frame_display, "Court corners (P1-P22)", (35, legend_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(frame_display, (20, legend_y + 50), 8, (255, 0, 255), -1)
        cv2.putText(frame_display, "Net poles", (35, legend_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        out.write(frame_display)

    cap.release()
    out.release()
    logging.info(f"Stage 1 visualization saved to {output_path}")


def visualize_stage2_pose_estimation(video_path: str, data_path: str, output_path: str) -> None:
    """Visualize pose estimation from pose.json."""
    logging.info("Creating Stage 2 visualization: Pose estimation")

    with open(data_path, 'r') as f:
        data = json.load(f)

    all_court_points = data.get("all_court_points", data.get("court_points", {}))
    pose_data = data["pose_data"]
    video_info = data["video_info"]

    # Organize poses by frame
    poses_by_frame = {}
    for pose in pose_data:
        frame_idx = pose["frame_index"]
        if frame_idx not in poses_by_frame:
            poses_by_frame[frame_idx] = []
        poses_by_frame[frame_idx].append(pose)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    fps = video_info["fps"]
    w = video_info["width"]
    h = video_info["height"]
    frame_count = video_info["frame_count"]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Pose skeleton edges
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
             (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]

    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()

        # Draw court points (subtle)
        frame_display = draw_all_court_points(frame_display, all_court_points, show_labels=False)

        # Draw poses
        if frame_idx in poses_by_frame:
            for pose in poses_by_frame[frame_idx]:
                human_idx = pose["human_index"]
                joints = pose["joints"]
                color = colors[human_idx % len(colors)]

                # Draw joints
                joint_positions = {}
                in_court_count = 0

                for joint in joints:
                    if joint["confidence"] > 0.5:
                        joint_idx = joint["joint_index"]
                        x, y = int(joint["x"]), int(joint["y"])

                        # Different appearance for in-court joints
                        if joint.get("in_court", False):
                            cv2.circle(frame_display, (x, y), 5, color, -1)
                            in_court_count += 1
                        else:
                            cv2.circle(frame_display, (x, y), 3, (128, 128, 128), -1)

                        joint_positions[joint_idx] = (x, y)

                # Draw skeleton
                for edge in edges:
                    if edge[0] in joint_positions and edge[1] in joint_positions:
                        pt1 = joint_positions[edge[0]]
                        pt2 = joint_positions[edge[1]]
                        cv2.line(frame_display, pt1, pt2, color, 2)

                # Label player
                if joint_positions:
                    head_pos = joint_positions.get(0, list(joint_positions.values())[0])
                    cv2.putText(frame_display, f"Player {human_idx} ({in_court_count} joints)",
                                (head_pos[0] - 20, head_pos[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Frame info
        total_players = len(poses_by_frame.get(frame_idx, []))
        cv2.putText(frame_display, f"Frame: {frame_idx} | Players detected: {total_players}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame_display)

    cap.release()
    out.release()
    logging.info(f"Stage 2 visualization saved to {output_path}")


def visualize_stage3_positions(video_path: str, data_path: str, output_path: str) -> None:
    """
    Visualize 3D positions from positions.json.
    Creates a side-by-side view with original video and overhead court visualization.
    """
    logging.info("Creating Stage 3 visualization: 3D position tracking")

    # Load position data
    with open(data_path, 'r') as f:
        data = json.load(f)

    court_points = data.get("court_points", {})
    all_court_points = data.get("all_court_points", court_points)
    video_info = data["video_info"]
    player_positions = data["player_positions"]

    # Extract corner points for calibration visualization
    image_points = extract_corner_points_from_data(court_points)

    # Organize positions by frame
    positions_by_frame = organize_positions_by_frame(player_positions)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    fps = video_info["fps"]
    w = video_info["width"]
    h = video_info["height"]
    frame_count = video_info["frame_count"]

    # Court visualization parameters
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    court_scale = 65  # Match the compute script
    court_img_h = int((court_length_m + 2 * margin_m) * court_scale)
    court_img_w = int((court_width_m + 2 * margin_m) * court_scale)

    # Setup court template
    court_img_template = np.zeros((court_img_h, court_img_w, 3), dtype=np.uint8)
    draw_court(court_img_template, court_scale)

    # Setup output video (side-by-side)
    out_w = w + court_img_w
    out_h = max(h, court_img_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    # Color palette for players
    color_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                     (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    def get_player_color(tracked_id: int) -> Tuple[int, int, int]:
        return color_palette[tracked_id % len(color_palette)]

    # Process each frame
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()
        court_img = court_img_template.copy()

        # Draw calibration points on original frame
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw frame number
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process player positions for this frame
        frame_players = positions_by_frame.get(frame_idx, [])

        for pos_data in frame_players:
            tracked_id = pos_data["tracked_id"]
            color = get_player_color(tracked_id)

            # Draw left hip position
            if "left_hip_world_X" in pos_data and "left_hip_world_Y" in pos_data:
                lhip_x = pos_data["left_hip_world_X"]
                lhip_y = pos_data["left_hip_world_Y"]
                lhip_px = world_to_court(lhip_x, lhip_y, court_img_w, court_img_h,
                                         court_scale, margin_m, court_length_m)
                cv2.circle(court_img, lhip_px, 3, (100, 100, 255), -1)
                cv2.putText(court_img, f"P{tracked_id}-LHip", (lhip_px[0] - 15, lhip_px[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)

            # Draw right hip position
            if "right_hip_world_X" in pos_data and "right_hip_world_Y" in pos_data:
                rhip_x = pos_data["right_hip_world_X"]
                rhip_y = pos_data["right_hip_world_Y"]
                rhip_px = world_to_court(rhip_x, rhip_y, court_img_w, court_img_h,
                                         court_scale, margin_m, court_length_m)
                cv2.circle(court_img, rhip_px, 3, (255, 100, 100), -1)
                cv2.putText(court_img, f"P{tracked_id}-RHip", (rhip_px[0] - 15, rhip_px[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)

            # Draw center hip position
            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = world_to_court(hip_x, hip_y, court_img_w, court_img_h,
                                        court_scale, margin_m, court_length_m)
                cv2.circle(court_img, hip_px, 5, color, -1)
                cv2.putText(court_img, f"P{tracked_id}-Hip", (hip_px[0] - 15, hip_px[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw left ankle position
            if "left_ankle_world_X" in pos_data and "left_ankle_world_Y" in pos_data:
                la_x = pos_data["left_ankle_world_X"]
                la_y = pos_data["left_ankle_world_Y"]
                la_px = world_to_court(la_x, la_y, court_img_w, court_img_h,
                                       court_scale, margin_m, court_length_m)
                cv2.circle(court_img, la_px, 4, (0, 255, 255), -1)
                cv2.putText(court_img, f"P{tracked_id}-LAnk", (la_px[0] + 5, la_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Draw right ankle position
            if "right_ankle_world_X" in pos_data and "right_ankle_world_Y" in pos_data:
                ra_x = pos_data["right_ankle_world_X"]
                ra_y = pos_data["right_ankle_world_Y"]
                ra_px = world_to_court(ra_x, ra_y, court_img_w, court_img_h,
                                       court_scale, margin_m, court_length_m)
                cv2.circle(court_img, ra_px, 4, (255, 255, 0), -1)
                cv2.putText(court_img, f"P{tracked_id}-RAnk", (ra_px[0] + 5, ra_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Add legend to court view
        legend_y = 50
        legend_x = court_img_w - 230
        cv2.putText(court_img, "POSITION LEGEND:", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Individual hip joints
        cv2.circle(court_img, (legend_x + 20, legend_y + 20), 3, (100, 100, 255), -1)
        cv2.putText(court_img, "Left Hip", (legend_x + 30, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        cv2.circle(court_img, (legend_x + 20, legend_y + 40), 3, (255, 100, 100), -1)
        cv2.putText(court_img, "Right Hip", (legend_x + 30, legend_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

        # Center hip
        cv2.circle(court_img, (legend_x + 20, legend_y + 60), 5, (255, 255, 255), -1)
        cv2.putText(court_img, "Hip Center", (legend_x + 30, legend_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Ankle positions
        cv2.circle(court_img, (legend_x + 20, legend_y + 80), 4, (0, 255, 255), -1)
        cv2.putText(court_img, "Left Ankle", (legend_x + 30, legend_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.circle(court_img, (legend_x + 20, legend_y + 100), 4, (255, 255, 0), -1)
        cv2.putText(court_img, "Right Ankle", (legend_x + 30, legend_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Add statistics
        total_players = len(frame_players)
        cv2.putText(court_img, f"Players tracked: {total_players}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(court_img, f"Frame: {frame_idx}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Combine original frame and court view side-by-side
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        out.write(combined)

    cap.release()
    out.release()

    total_positions = len(player_positions)
    unique_players = len(set(pos["tracked_id"] for pos in player_positions))

    logging.info(f"✓ Stage 3 visualization saved to {output_path}")
    logging.info(f"✓ Total positions visualized: {total_positions}")
    logging.info(f"✓ Unique players tracked: {unique_players}")
    logging.info(f"✓ Frames with position data: {len(positions_by_frame)}")


def visualize_stage4_corrected_positions(video_path: str, data_path: str, output_path: str) -> None:
    """
    Visualize corrected positions from corrected_positions.json.
    Shows comparison between original and corrected positions with interpolated trajectories.
    """
    logging.info("Creating Stage 4 visualization: Corrected position comparison")

    # Load corrected position data
    with open(data_path, 'r') as f:
        corrected_data = json.load(f)

    # Load original position data for comparison
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)
    original_path = os.path.join(result_dir, "positions.json")

    try:
        with open(original_path, 'r') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Original positions file not found: {original_path}")
        return

    court_points = corrected_data.get("court_points", {})
    all_court_points = corrected_data.get("all_court_points", court_points)
    video_info = corrected_data["video_info"]
    corrected_positions = corrected_data["player_positions"]
    original_positions = original_data["player_positions"]

    # Extract corner points for calibration visualization
    image_points = extract_corner_points_from_data(court_points)

    # Organize positions by frame for both datasets
    corrected_by_frame = organize_positions_by_frame(corrected_positions)
    original_by_frame = organize_positions_by_frame(original_positions)

    # Find frames where corrections were made
    correction_frames = set()
    for frame_idx in corrected_by_frame:
        if frame_idx in original_by_frame:
            for corr_pos in corrected_by_frame[frame_idx]:
                for orig_pos in original_by_frame[frame_idx]:
                    if corr_pos["tracked_id"] == orig_pos["tracked_id"]:
                        # Check if positions differ significantly
                        corr_x = corr_pos.get("hip_world_X", 0)
                        corr_y = corr_pos.get("hip_world_Y", 0)
                        orig_x = orig_pos.get("hip_world_X", 0)
                        orig_y = orig_pos.get("hip_world_Y", 0)

                        if abs(corr_x - orig_x) > 0.01 or abs(corr_y - orig_y) > 0.01:
                            correction_frames.add(frame_idx)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    fps = video_info["fps"]
    w = video_info["width"]
    h = video_info["height"]
    frame_count = video_info["frame_count"]

    # Court visualization parameters
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    court_scale = 65
    court_img_h = int((court_length_m + 2 * margin_m) * court_scale)
    court_img_w = int((court_width_m + 2 * margin_m) * court_scale)

    # Setup court template
    court_img_template = np.zeros((court_img_h, court_img_w, 3), dtype=np.uint8)
    draw_court(court_img_template, court_scale)

    # Setup output video (side-by-side)
    out_w = w + court_img_w
    out_h = max(h, court_img_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    # Color palette for players
    color_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                     (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    def get_player_color(tracked_id: int) -> Tuple[int, int, int]:
        return color_palette[tracked_id % len(color_palette)]

    # Store trajectory history for visualization
    trajectory_history = defaultdict(lambda: {'original': [], 'corrected': []})

    # Process each frame
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()
        court_img = court_img_template.copy()

        # Draw calibration points on original frame
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Highlight if this is a correction frame
        is_correction_frame = frame_idx in correction_frames
        if is_correction_frame:
            cv2.rectangle(frame_display, (0, 0), (w-1, h-1), (0, 255, 255), 8)
            cv2.putText(frame_display, "CORRECTED FRAME", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Draw frame number
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process corrected positions for this frame
        corrected_frame_players = corrected_by_frame.get(frame_idx, [])
        original_frame_players = original_by_frame.get(frame_idx, [])

        # Update trajectory history
        for pos_data in corrected_frame_players:
            tracked_id = pos_data["tracked_id"]
            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = world_to_court(hip_x, hip_y, court_img_w, court_img_h,
                                        court_scale, margin_m, court_length_m)
                trajectory_history[tracked_id]['corrected'].append(hip_px)

        for pos_data in original_frame_players:
            tracked_id = pos_data["tracked_id"]
            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = world_to_court(hip_x, hip_y, court_img_w, court_img_h,
                                        court_scale, margin_m, court_length_m)
                trajectory_history[tracked_id]['original'].append(hip_px)

        # Draw trajectories (last 30 frames)
        for tracked_id, trajectories in trajectory_history.items():
            color = get_player_color(tracked_id)

            # Draw original trajectory in gray
            original_traj = trajectories['original'][-30:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    cv2.line(court_img, original_traj[i-1], original_traj[i], (128, 128, 128), 2)

            # Draw corrected trajectory in color
            corrected_traj = trajectories['corrected'][-30:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], color, 3)

        # Draw current positions
        for pos_data in corrected_frame_players:
            tracked_id = pos_data["tracked_id"]
            color = get_player_color(tracked_id)

            # Draw corrected hip position (larger)
            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = world_to_court(hip_x, hip_y, court_img_w, court_img_h,
                                        court_scale, margin_m, court_length_m)
                cv2.circle(court_img, hip_px, 8, color, -1)
                cv2.circle(court_img, hip_px, 10, (255, 255, 255), 2)
                cv2.putText(court_img, f"P{tracked_id}-Corrected", (hip_px[0] - 25, hip_px[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw original positions (smaller, if different)
        for pos_data in original_frame_players:
            tracked_id = pos_data["tracked_id"]

            # Find corresponding corrected position
            corrected_pos = None
            for corr_pos in corrected_frame_players:
                if corr_pos["tracked_id"] == tracked_id:
                    corrected_pos = corr_pos
                    break

            if corrected_pos and "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                orig_x = pos_data["hip_world_X"]
                orig_y = pos_data["hip_world_Y"]
                corr_x = corrected_pos.get("hip_world_X", orig_x)
                corr_y = corrected_pos.get("hip_world_Y", orig_y)

                # Only draw if positions differ
                if abs(orig_x - corr_x) > 0.01 or abs(orig_y - corr_y) > 0.01:
                    orig_px = world_to_court(orig_x, orig_y, court_img_w, court_img_h,
                                             court_scale, margin_m, court_length_m)
                    cv2.circle(court_img, orig_px, 4, (128, 128, 128), -1)
                    cv2.putText(court_img, f"P{tracked_id}-Original", (orig_px[0] - 25, orig_px[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

                    # Draw line between original and corrected
                    corr_px = world_to_court(corr_x, corr_y, court_img_w, court_img_h,
                                             court_scale, margin_m, court_length_m)
                    cv2.line(court_img, orig_px, corr_px, (255, 255, 255), 1)

        # Add legend to court view
        legend_y = 50
        legend_x = court_img_w - 250
        cv2.putText(court_img, "CORRECTION LEGEND:", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Corrected positions
        cv2.circle(court_img, (legend_x + 20, legend_y + 20), 8, (0, 255, 0), -1)
        cv2.putText(court_img, "Corrected Position", (legend_x + 35, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Original positions
        cv2.circle(court_img, (legend_x + 20, legend_y + 40), 4, (128, 128, 128), -1)
        cv2.putText(court_img, "Original Position", (legend_x + 35, legend_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Trajectory lines
        cv2.line(court_img, (legend_x + 15, legend_y + 60), (legend_x + 30, legend_y + 60), (0, 255, 0), 3)
        cv2.putText(court_img, "Corrected Trajectory", (legend_x + 35, legend_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.line(court_img, (legend_x + 15, legend_y + 80), (legend_x + 30, legend_y + 80), (128, 128, 128), 2)
        cv2.putText(court_img, "Original Trajectory", (legend_x + 35, legend_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Add statistics
        total_corrections = len(correction_frames)
        current_corrections = len([p for p in corrected_frame_players
                                   if any(abs(p.get("hip_world_X", 0) - op.get("hip_world_X", 0)) > 0.01 or
                                          abs(p.get("hip_world_Y", 0) - op.get("hip_world_Y", 0)) > 0.01
                                          for op in original_frame_players
                                          if op["tracked_id"] == p["tracked_id"])])

        cv2.putText(court_img, f"Total correction frames: {total_corrections}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(court_img, f"Current frame corrections: {current_corrections}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(court_img, f"Frame: {frame_idx}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Combine original frame and court view side-by-side
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        out.write(combined)

    cap.release()
    out.release()

    logging.info(f"✓ Stage 4 visualization saved to {output_path}")
    logging.info(f"✓ Total frames with corrections: {len(correction_frames)}")
    logging.info(f"✓ Corrected vs original position comparison complete")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for BPLFMV pipeline stages")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Stage to visualize (1=court, 2=pose, 3=positions, 4=corrected)")
    parser.add_argument("--data_path", type=str, help="Path to data file (auto-detected if not provided)")
    parser.add_argument("--output", type=str, help="Output video path (auto-generated if not provided)")

    args = parser.parse_args()

    # Auto-detect paths based on your structure
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

    # Call appropriate visualization function
    try:
        if args.stage == 1:
            visualize_stage1_court_detection(args.video_path, args.data_path, args.output)
        elif args.stage == 2:
            visualize_stage2_pose_estimation(args.video_path, args.data_path, args.output)
        elif args.stage == 3:
            visualize_stage3_positions(args.video_path, args.data_path, args.output)
        elif args.stage == 4:
            visualize_stage4_corrected_positions(args.video_path, args.data_path, args.output)
        logging.info(f"✓ Visualization complete! Output saved to: {args.output}")

    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()