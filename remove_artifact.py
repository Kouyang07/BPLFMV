#!/usr/bin/env python3
"""
Unified Jump Analysis Script with Court-Based Player Matching

Uses ML model to detect valid jumps, then matches players using court coordinates
to find specific jumping player and correct their position displacement.

Usage:
    python unified_jump_analyzer.py samples/test.mp4
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
    """Load position data and extract homography matrix."""
    base_name = Path(video_path).stem
    result_dir = Path("results") / base_name
    position_json_path = result_dir / "positions.json"

    try:
        with open(position_json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded position data from: {position_json_path}")

        # Extract homography matrix from court points
        homography_matrix = calculate_homography_from_court_points(data.get('court_points', {}))

        return data, position_json_path.parent, homography_matrix
    except FileNotFoundError:
        print(f"❌ Position data not found: {position_json_path}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading position data: {e}")
        return None, None, None

def calculate_homography_from_court_points(court_points):
    """Calculate homography matrix from court corner points."""
    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    required_points = ['P1', 'P2', 'P3', 'P4']
    if not all(point in court_points for point in required_points):
        raise ValueError(f"Missing required court points: {required_points}")

    # Image points (pixels)
    image_points = np.array([
        court_points['P1'],  # Top-right
        court_points['P2'],  # Bottom-right
        court_points['P3'],  # Bottom-left
        court_points['P4']   # Top-left
    ], dtype=np.float32)

    # World coordinates (meters)
    world_points = np.array([
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH],
        [0, COURT_LENGTH],
        [0, 0]
    ], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(image_points, world_points, cv2.RANSAC)
    if homography_matrix is None:
        raise ValueError("Failed to calculate homography")

    print("Homography matrix calculated successfully")
    return homography_matrix

def transform_point_to_court(pixel_point, homography_matrix):
    """Transform pixel point to court coordinates using homography."""
    point = np.array([[pixel_point]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, homography_matrix)
    return float(world_point[0][0][0]), float(world_point[0][0][1])

def get_player_court_position(position_data, frame_index, player_id, homography_matrix):
    """Get player's court position for a specific frame."""
    for pos in position_data["player_positions"]:
        if pos['frame_index'] == frame_index and pos.get('player_id') == player_id:
            # Use weighted world coordinates directly if available
            hip_x = pos.get('hip_world_X')
            hip_y = pos.get('hip_world_Y')
            if hip_x is not None and hip_y is not None:
                return float(hip_x), float(hip_y)
    return None, None

def calculate_weighted_position(hip_x, hip_y, left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y):
    """Calculate weighted average position using hip and ankle coordinates."""
    positions_x = []
    positions_y = []
    weights = []

    if hip_x is not None and hip_y is not None:
        positions_x.append(hip_x)
        positions_y.append(hip_y)
        weights.append(0.7)

    if left_ankle_x is not None and left_ankle_y is not None:
        positions_x.append(left_ankle_x)
        positions_y.append(left_ankle_y)
        weights.append(0.15)

    if right_ankle_x is not None and right_ankle_y is not None:
        positions_x.append(right_ankle_x)
        positions_y.append(right_ankle_y)
        weights.append(0.15)

    if not positions_x:
        return None, None

    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        return None, None

    weighted_x = sum(pos * weight for pos, weight in zip(positions_x, normalized_weights))
    weighted_y = sum(pos * weight for pos, weight in zip(positions_y, normalized_weights))

    return weighted_x, weighted_y

def extract_player_coordinates(player_positions):
    """Extract player coordinate trajectories using weighted averaging."""
    coordinates = defaultdict(lambda: {
        'frames': [], 'weighted_x': [], 'weighted_y': [],
        'hip_x': [], 'hip_y': [], 'left_ankle_x': [], 'left_ankle_y': [],
        'right_ankle_x': [], 'right_ankle_y': []
    })

    print("Extracting player coordinates...")
    for pos in tqdm(player_positions, desc="Processing player positions", unit="position"):
        player_id = pos.get('player_id')
        if player_id is None:
            continue

        frame_idx = pos['frame_index']
        hip_x = pos.get('hip_world_X')
        hip_y = pos.get('hip_world_Y')
        left_ankle_x = pos.get('left_ankle_world_X')
        left_ankle_y = pos.get('left_ankle_world_Y')
        right_ankle_x = pos.get('right_ankle_world_X')
        right_ankle_y = pos.get('right_ankle_world_Y')

        weighted_x, weighted_y = calculate_weighted_position(
            hip_x, hip_y, left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y
        )

        if weighted_x is not None and weighted_y is not None:
            coordinates[player_id]['frames'].append(frame_idx)
            coordinates[player_id]['weighted_x'].append(weighted_x)
            coordinates[player_id]['weighted_y'].append(weighted_y)
            coordinates[player_id]['hip_x'].append(hip_x if hip_x is not None else 0)
            coordinates[player_id]['hip_y'].append(hip_y if hip_y is not None else 0)
            coordinates[player_id]['left_ankle_x'].append(left_ankle_x if left_ankle_x is not None else 0)
            coordinates[player_id]['left_ankle_y'].append(left_ankle_y if left_ankle_y is not None else 0)
            coordinates[player_id]['right_ankle_x'].append(right_ankle_x if right_ankle_x is not None else 0)
            coordinates[player_id]['right_ankle_y'].append(right_ankle_y if right_ankle_y is not None else 0)

    # Convert to numpy arrays and sort by frame
    for player_id in coordinates:
        coord = coordinates[player_id]
        sort_idx = np.argsort(coord['frames'])
        for key in coord:
            coord[key] = np.array(coord[key])[sort_idx]

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
            weighted_y = coord_data['weighted_y']

            if len(weighted_y) >= 10:
                peaks, valleys = find_peaks_and_valleys(weighted_y, frames)

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
                            weighted_x = coord_data['weighted_x']
                            begin_x = weighted_x[begin_frame_idx[0]]
                            end_x = weighted_x[end_frame_idx[0]]

                            correlations.append({
                                'player_id': best_match,
                                'ml_frame': ml_frame,
                                'ml_confidence': ml_jump['confidence'],
                                'court_distance': min_distance,
                                'begin_frame': begin[0],
                                'begin_x': begin_x,
                                'begin_y': begin[1],
                                'jump_frame': valley_frame,
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

def interpolate_positions(begin_data, end_data, frame, begin_frame, end_frame):
    """Interpolate position components between begin and end frames."""
    if end_frame == begin_frame:
        t = 0
    else:
        t = (frame - begin_frame) / (end_frame - begin_frame)

    return {
        'hip_x': float(begin_data['hip_x'] + t * (end_data['hip_x'] - begin_data['hip_x'])),
        'hip_y': float(begin_data['hip_y'] + t * (end_data['hip_y'] - begin_data['hip_y'])),
        'left_ankle_x': float(begin_data['left_ankle_x'] + t * (end_data['left_ankle_x'] - begin_data['left_ankle_x'])),
        'left_ankle_y': float(begin_data['left_ankle_y'] + t * (end_data['left_ankle_y'] - begin_data['left_ankle_y'])),
        'right_ankle_x': float(begin_data['right_ankle_x'] + t * (end_data['right_ankle_x'] - begin_data['right_ankle_x'])),
        'right_ankle_y': float(begin_data['right_ankle_y'] + t * (end_data['right_ankle_y'] - begin_data['right_ankle_y']))
    }

def save_corrected_positions(original_position_data, correlations, coordinates, output_dir):
    """Save corrected position data with interpolated jump sequences."""
    # Always copy original data first
    corrected_data = json.loads(json.dumps(original_position_data))

    # Add metadata about the correction process
    corrected_data['correction_metadata'] = {
        'jumps_detected': len(correlations),
        'correction_applied': len(correlations) > 0,
        'timestamp': str(np.datetime64('now'))
    }

    if not correlations:
        print("No correlations to process - saving original data as corrected data")
        output_path = output_dir / "corrected_positions.json"
        with open(output_path, 'w') as f:
            json.dump(corrected_data, f, indent=2)
        print(f"✅ Saved uncorrected positions to: {output_path}")
        return

    # Create lookup for quick position access
    position_lookup = {}
    for idx, pos in enumerate(corrected_data["player_positions"]):
        key = (pos.get('player_id'), pos['frame_index'])
        position_lookup[key] = idx

    total_interpolated_frames = 0

    print("Applying corrections...")
    for corr in tqdm(correlations, desc="Interpolating jumps", unit="jump"):
        player_id = corr['player_id']
        begin_frame = corr['begin_frame']
        end_frame = corr['end_frame']

        print(f"\nInterpolating Player {player_id}: frames {begin_frame} to {end_frame}")

        coord_data = coordinates[player_id]
        frames = coord_data['frames']

        # Find begin and end indices
        begin_idx = np.where(frames == begin_frame)[0]
        end_idx = np.where(frames == end_frame)[0]

        if len(begin_idx) == 0 or len(end_idx) == 0:
            print(f"  ❌ Could not find coordinate data")
            continue

        begin_idx = begin_idx[0]
        end_idx = end_idx[0]

        # Extract position data for interpolation
        begin_data = {
            'hip_x': coord_data['hip_x'][begin_idx],
            'hip_y': coord_data['hip_y'][begin_idx],
            'left_ankle_x': coord_data['left_ankle_x'][begin_idx],
            'left_ankle_y': coord_data['left_ankle_y'][begin_idx],
            'right_ankle_x': coord_data['right_ankle_x'][begin_idx],
            'right_ankle_y': coord_data['right_ankle_y'][begin_idx]
        }

        end_data = {
            'hip_x': coord_data['hip_x'][end_idx],
            'hip_y': coord_data['hip_y'][end_idx],
            'left_ankle_x': coord_data['left_ankle_x'][end_idx],
            'left_ankle_y': coord_data['left_ankle_y'][end_idx],
            'right_ankle_x': coord_data['right_ankle_x'][end_idx],
            'right_ankle_y': coord_data['right_ankle_y'][end_idx]
        }

        # Interpolate between begin and end frames
        frames_interpolated = 0
        for frame in range(begin_frame, end_frame + 1):
            interpolated = interpolate_positions(begin_data, end_data, frame, begin_frame, end_frame)

            key = (player_id, frame)
            if key in position_lookup:
                pos_idx = position_lookup[key]
                pos = corrected_data["player_positions"][pos_idx]

                # Update position components
                pos['hip_world_X'] = interpolated['hip_x']
                pos['hip_world_Y'] = interpolated['hip_y']
                pos['left_ankle_world_X'] = interpolated['left_ankle_x']
                pos['left_ankle_world_Y'] = interpolated['left_ankle_y']
                pos['right_ankle_world_X'] = interpolated['right_ankle_x']
                pos['right_ankle_world_Y'] = interpolated['right_ankle_y']

                frames_interpolated += 1

        print(f"  ✅ Interpolated {frames_interpolated} frames")
        total_interpolated_frames += frames_interpolated

    # Save corrected data
    output_path = output_dir / "corrected_positions.json"
    with open(output_path, 'w') as f:
        json.dump(corrected_data, f, indent=2)

    print(f"\n✅ Saved corrected positions to: {output_path}")
    print(f"Summary: Interpolated {total_interpolated_frames} frames across {len(correlations)} jump sequences")

def main():
    parser = argparse.ArgumentParser(description='Unified jump analysis with court-based player matching')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--model', default='resources/BLPFMV.pt', help='Path to YOLO model weights')
    parser.add_argument('--confidence', type=float, default=0.2, help='ML detection confidence threshold')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], help='Force specific device')
    parser.add_argument('--proximity', type=int, default=15, help='Frame proximity for correlating jumps')
    parser.add_argument('--court-distance', type=float, default=4, help='Court distance threshold (meters)')

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        return 1

    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return 1

    print(f"System: {platform.system()}")
    print(f"PyTorch: {torch.__version__}")

    try:
        # Step 1: ML jump detection
        print("\n" + "="*50)
        print("STEP 1: ML JUMP DETECTION")
        print("="*50)

        ml_jumps, fps = detect_ml_jumps(args.video_path, args.model, args.confidence, args.device)

        print(f"\nML detected {len(ml_jumps)} valid jump frames:")
        for i, jump in enumerate(ml_jumps, 1):
            frame = jump['frame']
            confidence = jump['confidence']
            timestamp = frame / fps
            print(f"  {i}. Frame {frame:4d} | Time: {timestamp:6.2f}s | Confidence: {confidence:.3f}")

        # Step 2: Load position data and homography
        print("\n" + "="*50)
        print("STEP 2: LOADING POSITION DATA")
        print("="*50)

        position_data, output_dir, homography_matrix = load_position_data(args.video_path)
        if not position_data:
            print("❌ Position data unavailable. Cannot proceed.")
            return 1

        coordinates = extract_player_coordinates(position_data["player_positions"])
        print(f"Found {len(coordinates)} players with position data")

        # Step 3: Court-based player matching (only if we have both ML jumps and homography)
        correlations = []
        if ml_jumps and homography_matrix is not None:
            print("\n" + "="*50)
            print("STEP 3: COURT-BASED PLAYER MATCHING")
            print("="*50)

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

        # Step 4: Always save corrected positions (even if no corrections applied)
        print("\n" + "="*50)
        print("STEP 4: SAVING CORRECTED POSITIONS")
        print("="*50)

        save_corrected_positions(position_data, correlations, coordinates, output_dir)

        # Summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)

        if correlations:
            print(f"✅ Successfully processed {len(ml_jumps)} ML detections")
            print(f"✅ Applied {len(correlations)} jump corrections")
            print(f"✅ Corrected positions saved to: {output_dir / 'corrected_positions.json'}")
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