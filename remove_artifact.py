#!/usr/bin/env python3
"""
Physics-Based Ankle-Only Jump Analysis Script with Enhanced Pipeline Integration

Integrates with the complete badminton analysis pipeline:
- Works with outputs from detect_court.py, detect_pose.py, and calculate_location.py
- Uses ML model to detect valid jumps, then matches players using court coordinates
- Corrects ankle position displacement during jump sequences using physics-based interpolation
- Enhanced compatibility with frame-organized position data

Usage:
    python physics_jump_analyzer.py samples/test.mp4
"""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch
import platform
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class PhysicsInterpolator:
    """Physics-based interpolation for realistic jump trajectories."""

    def __init__(self, gravity: float = 9.81, fps: float = 30.0):
        """
        Initialize physics interpolator.

        Args:
            gravity: Gravitational acceleration (m/s²)
            fps: Video frame rate (frames/second)
        """
        self.gravity = gravity
        self.fps = fps
        self.frame_time = 1.0 / fps  # Time per frame in seconds

    def interpolate_jump_trajectory(self, begin_data: Dict, end_data: Dict,
                                    jump_frame: int, begin_frame: int, end_frame: int,
                                    target_frame: int) -> Dict:
        """
        Physics-based interpolation using projectile motion equations.

        Args:
            begin_data: Position at jump start {x, y}
            end_data: Position at jump end {x, y}
            jump_frame: Frame number of jump peak
            begin_frame: Jump start frame
            end_frame: Jump end frame
            target_frame: Frame to interpolate

        Returns:
            Interpolated position {x, y}
        """
        if end_frame == begin_frame:
            return begin_data

        # Convert frame numbers to time
        total_time = (end_frame - begin_frame) * self.frame_time
        jump_time = (jump_frame - begin_frame) * self.frame_time
        target_time = (target_frame - begin_frame) * self.frame_time

        # Clamp target time to valid range
        target_time = max(0, min(target_time, total_time))

        # Calculate displacement
        dx = end_data['x'] - begin_data['x']
        dy = end_data['y'] - begin_data['y']

        # Horizontal motion (constant velocity)
        vx = dx / total_time if total_time > 0 else 0

        # Vertical motion with physics
        # At jump peak (jump_time): vy = 0
        # Use kinematic equations to solve for initial vertical velocity

        # For projectile motion: y = y0 + v0y*t - 0.5*g*t^2
        # At peak: vy = v0y - g*t_peak = 0, so v0y = g*t_peak
        # Final position: yf = y0 + v0y*T - 0.5*g*T^2

        if jump_time > 0 and jump_time < total_time:
            # Calculate initial vertical velocity based on jump peak timing
            # v0y such that at jump_time, vertical velocity is 0
            v0y = self.gravity * jump_time

            # Adjust v0y to ensure we reach the correct end position
            # yf = y0 + v0y*T - 0.5*g*T^2 = y0 + dy
            # Solve for adjustment
            predicted_dy = v0y * total_time - 0.5 * self.gravity * total_time**2
            velocity_adjustment = (dy - predicted_dy) / total_time if total_time > 0 else 0
            v0y += velocity_adjustment
        else:
            # Fallback: use average velocity with slight upward bias
            v0y = dy / total_time if total_time > 0 else 0
            # Add slight upward initial velocity for more realistic trajectory
            if abs(dy) < 1.0:  # If vertical movement is small, assume a small jump
                v0y += 2.0  # Small upward boost

        # Calculate position at target time
        x = begin_data['x'] + vx * target_time
        y = begin_data['y'] + v0y * target_time - 0.5 * self.gravity * target_time**2

        # Apply constraints
        # 1. Don't go below the baseline (assume court level is the minimum of begin/end y)
        min_y = min(begin_data['y'], end_data['y'])
        y = max(y, min_y)

        # 2. Ensure reasonable trajectory shape
        # If we're past the jump peak time, y should be decreasing
        if target_time > jump_time and jump_time > 0:
            # Calculate y at jump peak for reference
            y_peak = begin_data['y'] + v0y * jump_time - 0.5 * self.gravity * jump_time**2
            # Ensure we don't go higher than the peak after the peak time
            if y > y_peak:
                y = y_peak

        # 3. Smooth transition near boundaries
        boundary_margin = 0.1 * total_time  # 10% of total time
        if target_time < boundary_margin:
            # Blend with begin position
            blend_factor = target_time / boundary_margin
            x = begin_data['x'] * (1 - blend_factor) + x * blend_factor
            y = begin_data['y'] * (1 - blend_factor) + y * blend_factor
        elif target_time > (total_time - boundary_margin):
            # Blend with end position
            blend_factor = (total_time - target_time) / boundary_margin
            x = end_data['x'] * (1 - blend_factor) + x * blend_factor
            y = end_data['y'] * (1 - blend_factor) + y * blend_factor

        return {'x': float(x), 'y': float(y)}

    def calculate_jump_physics_metrics(self, begin_data: Dict, end_data: Dict,
                                       jump_data: Dict, begin_frame: int,
                                       jump_frame: int, end_frame: int) -> Dict:
        """
        Calculate physics metrics for the jump.

        Returns:
            Dictionary containing jump height, velocities, etc.
        """
        total_time = (end_frame - begin_frame) * self.frame_time
        jump_time = (jump_frame - begin_frame) * self.frame_time

        # Calculate jump height (peak y relative to start/end baseline)
        # Note: In many coordinate systems, y increases downward, so we need to handle this
        baseline_y = max(begin_data['y'], end_data['y'])  # Use max instead of min for downward y
        jump_height = baseline_y - jump_data['y']  # Reverse subtraction for downward y

        # If jump height is still negative, the coordinate system might be upward
        if jump_height < 0:
            baseline_y = min(begin_data['y'], end_data['y'])
            jump_height = jump_data['y'] - baseline_y

        # Ensure we have positive jump height (absolute value as fallback)
        if jump_height < 0:
            jump_height = abs(jump_height)

        # Calculate horizontal velocity
        dx = end_data['x'] - begin_data['x']
        vx = dx / total_time if total_time > 0 else 0

        # Calculate initial vertical velocity
        v0y = self.gravity * jump_time if jump_time > 0 else 0

        # Calculate maximum theoretical height
        max_height = (v0y**2) / (2 * self.gravity) if v0y > 0 else 0

        return {
            'jump_height': float(jump_height),
            'max_theoretical_height': float(max_height),
            'horizontal_velocity': float(vx),
            'initial_vertical_velocity': float(v0y),
            'total_time': float(total_time),
            'time_to_peak': float(jump_time),
            'baseline_y': float(baseline_y),
            'jump_peak_y': float(jump_data['y'])
        }

def detect_best_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) detected")
        return 'mps'
    print(f"⚠️  Using CPU")
    return 'cpu'

def setup_device_and_model(model_path, device=None):
    """Setup YOLO model with optimal device."""
    if device is None:
        device = detect_best_device()

    model = YOLO(model_path)
    model.to(device)
    print(f"Model loaded on {device.upper()}")
    return model, device

def detect_ml_jumps(video_path, model_path, confidence_threshold=0.5, device=None):
    """Detect valid jump frames with bounding box information."""
    model, device_used = setup_device_and_model(model_path, device)

    class_names = {
        0: 'BackhandClear', 1: 'BackhandLift', 2: 'BackhandServe',
        3: 'ForehandClear', 4: 'ForehandLift', 5: 'ReadyPosition', 6: 'Smash'
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {total_frames} frames at {fps} FPS")

    frame_detections = []
    frame_number = 0

    # Progress bar for video processing
    with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence_threshold, verbose=False)
            detections = []

            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names.get(class_id, 'Unknown')

                    # Get bounding box and bottom center
                    bbox = box.xyxy[0].cpu().numpy()
                    bottom_center_x = float((bbox[0] + bbox[2]) / 2)
                    bottom_center_y = float(bbox[3])  # Bottom edge

                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'frame': frame_number,
                        'bbox': bbox.tolist(),
                        'bottom_center_x': bottom_center_x,
                        'bottom_center_y': bottom_center_y
                    })

            frame_detections.append(detections)
            frame_number += 1
            pbar.update(1)

    cap.release()
    print(f"Finished processing {frame_number} frames")

    # Find valid smash frames
    print("Analyzing smash sequences...")
    valid_smash_frames = find_valid_smash_frames(frame_detections, window_size=10)
    return valid_smash_frames, fps

def find_valid_smash_frames(frame_detections, window_size=10):
    """Find smash frames surrounded by forehand clear detections."""
    valid_frames = []

    # Progress bar for smash analysis
    smash_frames = []
    for frame_idx, detections in enumerate(frame_detections):
        smash_detections = [det for det in detections if det['class_name'] == 'Smash']
        if smash_detections:
            smash_frames.extend([(frame_idx, det) for det in smash_detections])

    with tqdm(smash_frames, desc="Analyzing smash sequences", unit="smash") as pbar:
        for frame_idx, smash_det in pbar:
            start_idx = max(0, frame_idx - window_size)
            end_idx = min(len(frame_detections), frame_idx + window_size + 1)

            # Check for forehand clear before and after in similar spatial region
            smash_x = smash_det['bottom_center_x']
            smash_y = smash_det['bottom_center_y']
            spatial_threshold = 200

            forehand_clear_before = any(
                any(
                    det['class_name'] == 'ForehandClear' and
                    abs(det['bottom_center_x'] - smash_x) < spatial_threshold and
                    abs(det['bottom_center_y'] - smash_y) < spatial_threshold
                    for det in frame_detections[i]
                )
                for i in range(start_idx, frame_idx)
            )

            forehand_clear_after = any(
                any(
                    det['class_name'] == 'ForehandClear' and
                    abs(det['bottom_center_x'] - smash_x) < spatial_threshold and
                    abs(det['bottom_center_y'] - smash_y) < spatial_threshold
                    for det in frame_detections[i]
                )
                for i in range(frame_idx + 1, end_idx)
            )

            if forehand_clear_before and forehand_clear_after:
                valid_frames.append({
                    'frame': frame_idx,
                    'confidence': smash_det['confidence'],
                    'bottom_center_x': smash_x,
                    'bottom_center_y': smash_y,
                    'bbox': smash_det['bbox']
                })

    return valid_frames

def load_position_data(video_path):
    """Load ankle position data and extract homography matrix from court points."""
    base_name = Path(video_path).stem
    result_dir = Path("results") / base_name
    position_json_path = result_dir / "positions.json"

    try:
        with open(position_json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded ankle position data from: {position_json_path}")

        # Extract court points for homography calculation
        court_points = {}

        # Check if court_info exists in the data structure
        if 'court_info' in data:
            print("Found court_info in positions.json")

        # Try to get court points from the pose data structure or fallback to court.csv
        pose_json_path = result_dir / "pose.json"
        if pose_json_path.exists():
            with open(pose_json_path, 'r') as f:
                pose_data = json.load(f)

            # Extract court points from pose.json
            if 'court_points' in pose_data:
                court_points = pose_data['court_points']
                print("✅ Loaded court points from pose.json")
            elif 'all_court_points' in pose_data:
                court_points = pose_data['all_court_points']
                print("✅ Loaded court points from pose.json (all_court_points)")

        # If no court points found, try loading from court.csv
        if not court_points:
            court_csv_path = result_dir / "court.csv"
            if court_csv_path.exists():
                court_points = load_court_points_from_csv(court_csv_path)
                print("✅ Loaded court points from court.csv")

        if not court_points:
            raise ValueError("No court points found in any source")

        # Calculate homography matrix from court points
        homography_matrix = calculate_homography_from_court_points(court_points)

        return data, position_json_path.parent, homography_matrix
    except FileNotFoundError:
        print(f"❌ Position data not found: {position_json_path}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading position data: {e}")
        return None, None, None

def load_court_points_from_csv(csv_path):
    """Load court points from court.csv file."""
    import csv
    court_points = {}

    with open(csv_path, 'r') as f:
        # Check if header exists
        first_line = f.readline().strip()
        f.seek(0)

        if 'Point' in first_line and 'X' in first_line:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row['Point'].strip()
                    x = float(row['X'])
                    y = float(row['Y'])
                    court_points[name] = [x, y]
                except (ValueError, KeyError):
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        name = row[0].strip()
                        x = float(row[1])
                        y = float(row[2])
                        court_points[name] = [x, y]
                    except (ValueError, IndexError):
                        continue

    return court_points

def calculate_homography_from_court_points(court_points):
    """Calculate homography matrix from court corner points."""
    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    required_points = ['P1', 'P2', 'P3', 'P4']
    if not all(point in court_points for point in required_points):
        raise ValueError(f"Missing required court points: {required_points}")

    # Image points (pixels) - handle both list and dict formats
    image_points = []
    for point in required_points:
        coords = court_points[point]
        if isinstance(coords, list) and len(coords) >= 2:
            image_points.append([coords[0], coords[1]])
        elif isinstance(coords, dict) and 'x' in coords and 'y' in coords:
            image_points.append([coords['x'], coords['y']])
        else:
            raise ValueError(f"Invalid coordinate format for {point}: {coords}")

    image_points = np.array(image_points, dtype=np.float32)

    # World coordinates (meters) - standard court layout
    world_points = np.array([
        [0, 0],                    # P1: Top-left
        [0, COURT_LENGTH],         # P2: Bottom-left
        [COURT_WIDTH, COURT_LENGTH], # P3: Bottom-right
        [COURT_WIDTH, 0]           # P4: Top-right
    ], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(image_points, world_points, cv2.RANSAC)
    if homography_matrix is None:
        raise ValueError("Failed to calculate homography")

    print("✅ Homography matrix calculated successfully")
    return homography_matrix

def transform_point_to_court(pixel_point, homography_matrix):
    """Transform pixel point to court coordinates using homography."""
    point = np.array([[pixel_point]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, homography_matrix)
    return float(world_point[0][0][0]), float(world_point[0][0][1])

def get_player_court_position(position_data, frame_index, player_id, homography_matrix):
    """Get player's ankle-based court position for a specific frame."""
    # Handle frame-organized data structure from calculate_location.py
    if 'frame_data' in position_data:
        frame_key = str(frame_index)
        if frame_key in position_data['frame_data']:
            frame_data = position_data['frame_data'][frame_key]
            player_key = f"player_{player_id}"

            if player_key in frame_data and 'center_position' in frame_data[player_key]:
                center_pos = frame_data[player_key]['center_position']
                return float(center_pos['x']), float(center_pos['y'])

            # Alternative: use first available ankle position
            if player_key in frame_data and 'ankles' in frame_data[player_key]:
                ankles = frame_data[player_key]['ankles']
                if ankles:
                    return float(ankles[0]['world_x']), float(ankles[0]['world_y'])

    # Handle legacy player_positions format
    if 'player_positions' in position_data:
        for pos in position_data["player_positions"]:
            if pos['frame_index'] == frame_index and pos.get('player_id') == player_id:
                # Use ankle-only coordinates (x, y) if available
                if 'x' in pos and 'y' in pos:
                    return float(pos['x']), float(pos['y'])
                # Fallback to legacy hip coordinates if available
                elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                    return float(pos['hip_world_X']), float(pos['hip_world_Y'])

    return None, None

def extract_player_coordinates(position_data):
    """Extract player coordinate trajectories for both frame-organized and legacy data."""
    coordinates = defaultdict(lambda: {
        'frames': [], 'x': [], 'y': []
    })

    print("Extracting player coordinates...")

    # Handle frame-organized data structure from calculate_location.py
    if 'frame_data' in position_data:
        print("Processing frame-organized data structure")
        frame_data = position_data['frame_data']

        for frame_idx_str, players_data in tqdm(frame_data.items(), desc="Processing frames", unit="frame"):
            frame_idx = int(frame_idx_str)

            for player_key, player_data in players_data.items():
                # Extract player_id from player_key (e.g., "player_0" -> 0)
                if player_key.startswith('player_'):
                    player_id = int(player_key.split('_')[1])
                else:
                    continue

                # Use center position if available
                if 'center_position' in player_data:
                    center_pos = player_data['center_position']
                    x, y = center_pos['x'], center_pos['y']

                    coordinates[player_id]['frames'].append(frame_idx)
                    coordinates[player_id]['x'].append(x)
                    coordinates[player_id]['y'].append(y)

                # Alternative: use average of ankle positions
                elif 'ankles' in player_data and player_data['ankles']:
                    ankles = player_data['ankles']
                    avg_x = sum(ankle['world_x'] for ankle in ankles) / len(ankles)
                    avg_y = sum(ankle['world_y'] for ankle in ankles) / len(ankles)

                    coordinates[player_id]['frames'].append(frame_idx)
                    coordinates[player_id]['x'].append(avg_x)
                    coordinates[player_id]['y'].append(avg_y)

    # Handle legacy player_positions format
    elif 'player_positions' in position_data:
        print("Processing legacy player_positions data structure")
        for pos in tqdm(position_data['player_positions'], desc="Processing positions", unit="position"):
            player_id = pos.get('player_id')
            if player_id is None:
                continue

            frame_idx = pos['frame_index']

            # Handle ankle-only format
            if 'x' in pos and 'y' in pos:
                x = pos['x']
                y = pos['y']
            # Fallback to legacy hip format
            elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                x = pos['hip_world_X']
                y = pos['hip_world_Y']
            else:
                continue

            if x is not None and y is not None:
                coordinates[player_id]['frames'].append(frame_idx)
                coordinates[player_id]['x'].append(x)
                coordinates[player_id]['y'].append(y)

    # Convert to numpy arrays and sort by frame
    for player_id in coordinates:
        coord = coordinates[player_id]
        if len(coord['frames']) > 0:
            sort_idx = np.argsort(coord['frames'])
            for key in coord:
                coord[key] = np.array(coord[key])[sort_idx]

    print(f"Extracted coordinates for {len(coordinates)} players")
    return dict(coordinates)

def find_peaks_and_valleys(y_data, frames, prominence_threshold=0.2, min_distance=10):
    """Find peaks (jumps) and valleys (beginnings/ends)."""
    try:
        from scipy import signal
    except ImportError:
        print("❌ scipy not available. Install with: pip install scipy")
        return [], []

    if len(y_data) < 10:
        return [], []

    # Smooth data
    smoothed = np.convolve(y_data, np.ones(5)/5, mode='same')

    # Find peaks and valleys
    peak_indices, _ = signal.find_peaks(smoothed, prominence=prominence_threshold, distance=min_distance)
    peaks = [(frames[idx], y_data[idx], idx) for idx in peak_indices if 0 <= idx < len(frames)]

    inverted_data = -smoothed
    valley_indices, _ = signal.find_peaks(inverted_data, prominence=prominence_threshold*0.7, distance=min_distance//2)
    valleys = [(frames[idx], y_data[idx], idx) for idx in valley_indices if 0 <= idx < len(frames)]

    return peaks, valleys

def correlate_ml_jumps_with_players(ml_jumps, coordinates, position_data, homography_matrix,
                                    proximity_threshold=15, court_distance_threshold=1.5):
    """Correlate ML-detected jumps with specific players using court coordinates."""
    correlations = []

    if not ml_jumps:
        print("No ML jumps to correlate")
        return correlations

    print("Correlating ML jumps with players...")
    for ml_jump in tqdm(ml_jumps, desc="Correlating jumps", unit="jump"):
        ml_frame = ml_jump['frame']
        ml_bottom_x = ml_jump['bottom_center_x']
        ml_bottom_y = ml_jump['bottom_center_y']

        # Convert ML detection to court coordinates
        ml_court_x, ml_court_y = transform_point_to_court((ml_bottom_x, ml_bottom_y), homography_matrix)

        print(f"\nML jump at frame {ml_frame}: screen ({ml_bottom_x:.1f}, {ml_bottom_y:.1f}) → court ({ml_court_x:.2f}, {ml_court_y:.2f})")

        # Find closest player in court coordinates
        best_match = None
        min_distance = float('inf')
        player_distances = {}

        for player_id, coord_data in coordinates.items():
            # Get player's court position for this frame (or nearby frames)
            player_court_x, player_court_y = get_player_court_position(
                position_data, ml_frame, player_id, homography_matrix
            )

            if player_court_x is None:
                # Try nearby frames
                for offset in range(-3, 4):
                    test_frame = ml_frame + offset
                    player_court_x, player_court_y = get_player_court_position(
                        position_data, test_frame, player_id, homography_matrix
                    )
                    if player_court_x is not None:
                        break

            if player_court_x is not None:
                # Calculate court distance
                court_distance = np.sqrt(
                    (ml_court_x - player_court_x)**2 +
                    (ml_court_y - player_court_y)**2
                )

                player_distances[player_id] = court_distance

                if court_distance < min_distance:
                    min_distance = court_distance
                    best_match = player_id

        # Print player analysis
        for player_id, distance in player_distances.items():
            marker = " ← CLOSEST MATCH" if player_id == best_match else ""
            print(f"  Player {player_id}: court distance = {distance:.2f}m{marker}")

        if best_match is not None and min_distance < court_distance_threshold:
            print(f"  Selected player {best_match} for interpolation (distance: {min_distance:.2f}m)")

            # Find position-based jump for this player
            coord_data = coordinates[best_match]
            frames = coord_data['frames']
            y_coords = coord_data['y']

            if len(y_coords) >= 10:
                peaks, valleys = find_peaks_and_valleys(y_coords, frames)

                # Find closest valley to ML jump
                nearby_valleys = [(f, y, idx) for f, y, idx in valleys
                                  if abs(f - ml_frame) <= proximity_threshold]

                if nearby_valleys:
                    closest_valley = min(nearby_valleys, key=lambda x: abs(x[0] - ml_frame))
                    valley_frame, valley_y, valley_idx = closest_valley

                    # Find neighboring peaks
                    before_peaks = [(f, y, idx) for f, y, idx in peaks if f < valley_frame]
                    after_peaks = [(f, y, idx) for f, y, idx in peaks if f > valley_frame]

                    if before_peaks and after_peaks:
                        begin = max(before_peaks, key=lambda x: x[0])
                        end = min(after_peaks, key=lambda x: x[0])

                        begin_frame_idx = np.where(frames == begin[0])[0]
                        end_frame_idx = np.where(frames == end[0])[0]

                        if len(begin_frame_idx) > 0 and len(end_frame_idx) > 0:
                            x_coords = coord_data['x']
                            begin_x = x_coords[begin_frame_idx[0]]
                            end_x = x_coords[end_frame_idx[0]]

                            correlations.append({
                                'player_id': best_match,
                                'ml_frame': ml_frame,
                                'ml_confidence': ml_jump['confidence'],
                                'court_distance': min_distance,
                                'begin_frame': begin[0],
                                'begin_x': begin_x,
                                'begin_y': begin[1],
                                'jump_frame': valley_frame,
                                'jump_x': x_coords[valley_idx] if valley_idx < len(x_coords) else (begin_x + end_x) / 2,
                                'jump_y': valley_y,
                                'end_frame': end[0],
                                'end_x': end_x,
                                'end_y': end[1],
                                'distance_to_ml': abs(valley_frame - ml_frame)
                            })

                            print(f"  ✅ Found jump sequence: {begin[0]}→{valley_frame}→{end[0]}")
                        else:
                            print(f"  ❌ Could not find coordinate indices")
                    else:
                        print(f"  ❌ Missing peaks around valley")
                else:
                    print(f"  ❌ No valleys found near ML frame")
            else:
                print(f"  ❌ Insufficient position data")
        else:
            if best_match is None:
                print(f"  ❌ No players found")
            else:
                print(f"  ❌ Closest player too far ({min_distance:.2f}m > {court_distance_threshold}m)")

    return correlations

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_corrected_positions(original_position_data, correlations, coordinates, output_dir, fps):
    """Save corrected position data with physics-based interpolated jump sequences."""
    # Always copy original data first and convert numpy types
    corrected_data = convert_numpy_types(json.loads(json.dumps(original_position_data, default=str)))

    # Initialize physics interpolator
    physics_interpolator = PhysicsInterpolator(gravity=9.81, fps=fps)

    # Add metadata about the correction process
    corrected_data['correction_metadata'] = {
        'jumps_detected': len(correlations),
        'correction_applied': len(correlations) > 0,
        'tracking_method': 'ankle_only_physics_enhanced',
        'interpolation_method': 'physics_based',
        'gravity': physics_interpolator.gravity,
        'fps': fps,
        'timestamp': str(np.datetime64('now'))
    }

    if not correlations:
        print("No correlations to process - saving original data as corrected data")
        output_path = output_dir / "corrected_positions.json"
        with open(output_path, 'w') as f:
            json.dump(corrected_data, f, indent=2)
        print(f"✅ Saved uncorrected positions to: {output_path}")
        return

    total_interpolated_frames = 0
    jump_physics_stats = []

    print("Applying physics-based corrections...")
    for corr in tqdm(correlations, desc="Physics interpolating jumps", unit="jump"):
        player_id = corr['player_id']
        begin_frame = corr['begin_frame']
        end_frame = corr['end_frame']
        jump_frame = corr['jump_frame']

        print(f"\nPhysics interpolating Player {player_id}: frames {begin_frame} to {end_frame}")
        print(f"  Jump peak at frame {jump_frame}")

        # Prepare data for physics interpolation
        begin_data = {'x': corr['begin_x'], 'y': corr['begin_y']}
        end_data = {'x': corr['end_x'], 'y': corr['end_y']}
        jump_data = {'x': corr['jump_x'], 'y': corr['jump_y']}

        # Calculate jump physics metrics
        physics_metrics = physics_interpolator.calculate_jump_physics_metrics(
            begin_data, end_data, jump_data, begin_frame, jump_frame, end_frame
        )

        # Convert numpy types in physics metrics
        physics_metrics = convert_numpy_types(physics_metrics)

        jump_physics_stats.append({
            'player_id': int(player_id),
            'jump_frame': int(jump_frame),
            'ml_frame': int(corr['ml_frame']),
            'physics_metrics': physics_metrics
        })

        print(f"  📊 Jump height: {physics_metrics['jump_height']:.2f}m")
        print(f"  📊 Horizontal velocity: {physics_metrics['horizontal_velocity']:.2f}m/s")
        print(f"  📊 Initial vertical velocity: {physics_metrics['initial_vertical_velocity']:.2f}m/s")
        print(f"  📊 Time to peak: {physics_metrics['time_to_peak']:.2f}s")

        # Interpolate between begin and end frames using physics
        frames_interpolated = 0

        # Handle frame-organized data structure
        if 'frame_data' in corrected_data:
            for frame in range(begin_frame, end_frame + 1):
                interpolated = physics_interpolator.interpolate_jump_trajectory(
                    begin_data, end_data, jump_frame, begin_frame, end_frame, frame
                )

                frame_key = str(frame)
                player_key = f"player_{player_id}"

                if frame_key in corrected_data['frame_data'] and player_key in corrected_data['frame_data'][frame_key]:
                    # Update center position (ensure float conversion)
                    corrected_data['frame_data'][frame_key][player_key]['center_position']['x'] = float(interpolated['x'])
                    corrected_data['frame_data'][frame_key][player_key]['center_position']['y'] = float(interpolated['y'])

                    # Update ankle positions if they exist
                    if 'ankles' in corrected_data['frame_data'][frame_key][player_key]:
                        for ankle in corrected_data['frame_data'][frame_key][player_key]['ankles']:
                            ankle['world_x'] = float(interpolated['x'])
                            ankle['world_y'] = float(interpolated['y'])

                    frames_interpolated += 1

        # Handle legacy player_positions format
        elif 'player_positions' in corrected_data:
            position_lookup = {}
            for idx, pos in enumerate(corrected_data["player_positions"]):
                key = (pos.get('player_id'), pos['frame_index'])
                position_lookup[key] = idx

            for frame in range(begin_frame, end_frame + 1):
                interpolated = physics_interpolator.interpolate_jump_trajectory(
                    begin_data, end_data, jump_frame, begin_frame, end_frame, frame
                )

                key = (player_id, frame)
                if key in position_lookup:
                    pos_idx = position_lookup[key]
                    pos = corrected_data["player_positions"][pos_idx]

                    # Update position components (ensure float conversion)
                    if 'x' in pos and 'y' in pos:
                        pos['x'] = float(interpolated['x'])
                        pos['y'] = float(interpolated['y'])
                    elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                        pos['hip_world_X'] = float(interpolated['x'])
                        pos['hip_world_Y'] = float(interpolated['y'])

                    frames_interpolated += 1

        print(f"  ✅ Physics interpolated {frames_interpolated} frames")
        total_interpolated_frames += frames_interpolated

    # Add physics statistics to metadata (ensure JSON serializable)
    corrected_data['correction_metadata']['jump_physics_stats'] = convert_numpy_types(jump_physics_stats)

    # Save corrected data
    output_path = output_dir / "corrected_positions.json"
    with open(output_path, 'w') as f:
        json.dump(corrected_data, f, indent=2)

    # Save physics analysis report
    physics_report_path = output_dir / "jump_physics_analysis.json"
    physics_report = convert_numpy_types({
        'analysis_metadata': {
            'total_jumps_analyzed': len(correlations),
            'total_frames_corrected': int(total_interpolated_frames),
            'physics_model': {
                'gravity': float(physics_interpolator.gravity),
                'fps': float(fps),
                'interpolation_method': 'projectile_motion'
            },
            'timestamp': str(np.datetime64('now'))
        },
        'jump_analyses': jump_physics_stats
    })

    with open(physics_report_path, 'w') as f:
        json.dump(physics_report, f, indent=2)

    print(f"\n✅ Saved physics-corrected positions to: {output_path}")
    print(f"✅ Saved physics analysis report to: {physics_report_path}")
    print(f"\n📊 PHYSICS CORRECTION SUMMARY:")
    print(f"Interpolated {total_interpolated_frames} frames across {len(correlations)} jump sequences")

    if jump_physics_stats:
        avg_jump_height = np.mean([stat['physics_metrics']['jump_height'] for stat in jump_physics_stats])
        avg_horizontal_velocity = np.mean([stat['physics_metrics']['horizontal_velocity'] for stat in jump_physics_stats])
        print(f"Average jump height: {avg_jump_height:.2f}m")
        print(f"Average horizontal velocity: {avg_horizontal_velocity:.2f}m/s")

def main():
    parser = argparse.ArgumentParser(description='Physics-based ankle-only jump analysis with enhanced pipeline integration')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--model', default='resources/BLPFMV.pt', help='Path to YOLO model weights')
    parser.add_argument('--confidence', type=float, default=0.2, help='ML detection confidence threshold')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], help='Force specific device')
    parser.add_argument('--proximity', type=int, default=15, help='Frame proximity for correlating jumps')
    parser.add_argument('--court-distance', type=float, default=4, help='Court distance threshold (meters)')
    parser.add_argument('--gravity', type=float, default=9.81, help='Gravitational acceleration (m/s²)')

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        return 1

    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return 1

    print(f"System: {platform.system()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Physics model: g = {args.gravity} m/s²")

    try:
        # Step 1: ML jump detection
        print("\n" + "="*60)
        print("STEP 1: ML JUMP DETECTION")
        print("="*60)

        ml_jumps, fps = detect_ml_jumps(args.video_path, args.model, args.confidence, args.device)

        print(f"\nML detected {len(ml_jumps)} valid jump frames:")
        for i, jump in enumerate(ml_jumps, 1):
            frame = jump['frame']
            confidence = jump['confidence']
            timestamp = frame / fps
            print(f"  {i}. Frame {frame:4d} | Time: {timestamp:6.2f}s | Confidence: {confidence:.3f}")

        # Step 2: Load position data and homography
        print("\n" + "="*60)
        print("STEP 2: LOADING POSITION DATA")
        print("="*60)

        position_data, output_dir, homography_matrix = load_position_data(args.video_path)
        if not position_data:
            print("❌ Position data unavailable. Cannot proceed.")
            return 1

        coordinates = extract_player_coordinates(position_data)
        print(f"Found {len(coordinates)} players with position data")

        # Step 3: Court-based player matching (only if we have both ML jumps and homography)
        correlations = []
        if ml_jumps and homography_matrix is not None:
            print("\n" + "="*60)
            print("STEP 3: COURT-BASED PLAYER MATCHING")
            print("="*60)

            correlations = correlate_ml_jumps_with_players(
                ml_jumps, coordinates, position_data, homography_matrix,
                args.proximity, args.court_distance
            )

            print(f"\n📊 CORRELATION SUMMARY")
            print(f"Found {len(correlations)} valid correlations:")
            for i, corr in enumerate(correlations, 1):
                print(f"  {i}. Player {corr['player_id']} | "
                      f"ML: Frame {corr['ml_frame']} | "
                      f"Sequence: {corr['begin_frame']}→{corr['jump_frame']}→{corr['end_frame']} | "
                      f"Court dist: {corr['court_distance']:.2f}m")
        else:
            if not ml_jumps:
                print("\n⚠️  No ML jumps detected - skipping correlation step")
            if homography_matrix is None:
                print("\n⚠️  No homography matrix available - skipping correlation step")

        # Step 4: Physics-based position correction
        print("\n" + "="*60)
        print("STEP 4: PHYSICS-BASED POSITION CORRECTION")
        print("="*60)

        save_corrected_positions(position_data, correlations, coordinates, output_dir, fps)

        # Summary
        print("\n" + "="*60)
        print("PHYSICS-BASED PROCESSING COMPLETE")
        print("="*60)

        if correlations:
            print(f"✅ Successfully processed {len(ml_jumps)} ML detections")
            print(f"✅ Applied {len(correlations)} physics-based jump corrections")
            print(f"✅ Physics model used: projectile motion with g = {args.gravity} m/s²")
            print(f"✅ Corrected positions saved to: {output_dir / 'corrected_positions.json'}")
            print(f"✅ Physics analysis saved to: {output_dir / 'jump_physics_analysis.json'}")
        else:
            print(f"ℹ️  Processed {len(ml_jumps)} ML detections")
            print(f"ℹ️  No corrections applied (no valid correlations found)")
            print(f"✅ Original positions saved as corrected data to: {output_dir / 'corrected_positions.json'}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())