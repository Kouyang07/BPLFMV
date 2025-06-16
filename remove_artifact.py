#!/usr/bin/env python3
"""
Unified Jump Analysis Script

Uses ML model to detect valid jumps, then cross-references with position data
to find complete sequences and save corrected positions with linear interpolation.
Now uses weighted average of hip and ankle positions for more robust tracking.

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

def detect_best_device():
    """Auto-detect best available device."""
    print("Detecting available devices...")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"✅ CUDA GPU detected: {gpu_count} device(s) - {gpu_name}")
        return 'cuda'

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) detected")
        return 'mps'

    cpu_info = platform.processor() or "Unknown"
    print(f"⚠️  Using CPU: {cpu_info}")
    return 'cpu'

def setup_device_and_model(model_path, device=None):
    """Setup YOLO model with optimal device."""
    if device is None:
        device = detect_best_device()

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    print(f"Model moved to {device.upper()} device")

    return model, device

def detect_ml_jumps(video_path, model_path, confidence_threshold=0.5, device=None):
    """Use ML model to detect valid jump frames."""
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

    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Using device: {device_used.upper()}")

    frame_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence_threshold, verbose=False)

            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names.get(class_id, 'Unknown')

                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'frame': frame_number
                    })

            frame_detections.append(detections)
            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

    finally:
        cap.release()

    print(f"Finished processing all {frame_number} frames")

    # Find valid smash frames surrounded by forehand clear
    valid_smash_frames = find_valid_smash_frames(frame_detections, window_size=10)

    return valid_smash_frames, fps

def find_valid_smash_frames(frame_detections, window_size=10):
    """Find smash frames surrounded by forehand clear detections."""
    valid_frames = []

    for frame_idx, detections in enumerate(frame_detections):
        has_smash = any(det['class_name'] == 'Smash' for det in detections)

        if has_smash:
            start_idx = max(0, frame_idx - window_size)
            end_idx = min(len(frame_detections), frame_idx + window_size + 1)

            # Check for forehand clear before and after
            forehand_clear_before = any(
                any(det['class_name'] == 'ForehandClear' for det in frame_detections[i])
                for i in range(start_idx, frame_idx)
            )

            forehand_clear_after = any(
                any(det['class_name'] == 'ForehandClear' for det in frame_detections[i])
                for i in range(frame_idx + 1, end_idx)
            )

            if forehand_clear_before and forehand_clear_after:
                smash_detections = [det for det in detections if det['class_name'] == 'Smash']
                for smash_det in smash_detections:
                    valid_frames.append({
                        'frame': frame_idx,
                        'confidence': smash_det['confidence']
                    })

    return valid_frames

def load_position_data(video_path):
    """Load position data for the video."""
    base_name = Path(video_path).stem
    result_dir = Path("results") / base_name
    position_json_path = result_dir / "positions.json"

    try:
        with open(position_json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded position data from: {position_json_path}")
        return data, position_json_path.parent
    except FileNotFoundError:
        print(f"❌ Position data not found: {position_json_path}")
        print("Run position analysis first")
        return None, None
    except Exception as e:
        print(f"❌ Error loading position data: {e}")
        return None, None

def calculate_weighted_position(hip_x, hip_y, left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y):
    """
    Calculate weighted average position using hip and ankle coordinates.
    Uses the same weighting scheme as the position tracker: hip=0.7, each ankle=0.15
    """
    positions_x = []
    positions_y = []
    weights = []

    # Hip position (primary weight)
    if hip_x is not None and hip_y is not None:
        positions_x.append(hip_x)
        positions_y.append(hip_y)
        weights.append(0.7)

    # Left ankle
    if left_ankle_x is not None and left_ankle_y is not None:
        positions_x.append(left_ankle_x)
        positions_y.append(left_ankle_y)
        weights.append(0.15)

    # Right ankle
    if right_ankle_x is not None and right_ankle_y is not None:
        positions_x.append(right_ankle_x)
        positions_y.append(right_ankle_y)
        weights.append(0.15)

    if not positions_x:
        return None, None

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        return None, None

    # Calculate weighted average
    weighted_x = sum(pos * weight for pos, weight in zip(positions_x, normalized_weights))
    weighted_y = sum(pos * weight for pos, weight in zip(positions_y, normalized_weights))

    return weighted_x, weighted_y

def extract_player_coordinates(player_positions):
    """Extract player coordinate trajectories using weighted average of hip and ankles."""
    coordinates = defaultdict(lambda: {'frames': [], 'weighted_x': [], 'weighted_y': [],
                                       'hip_x': [], 'hip_y': [], 'left_ankle_x': [], 'left_ankle_y': [],
                                       'right_ankle_x': [], 'right_ankle_y': []})

    for pos in player_positions:
        tracked_id = pos.get('tracked_id')
        if tracked_id is None:
            continue

        frame_idx = pos['frame_index']
        hip_x = pos.get('hip_world_X')
        hip_y = pos.get('hip_world_Y')
        left_ankle_x = pos.get('left_ankle_world_X')
        left_ankle_y = pos.get('left_ankle_world_Y')
        right_ankle_x = pos.get('right_ankle_world_X')
        right_ankle_y = pos.get('right_ankle_world_Y')

        # Calculate weighted position
        weighted_x, weighted_y = calculate_weighted_position(
            hip_x, hip_y, left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y
        )

        if weighted_x is not None and weighted_y is not None:
            coordinates[tracked_id]['frames'].append(frame_idx)
            coordinates[tracked_id]['weighted_x'].append(weighted_x)
            coordinates[tracked_id]['weighted_y'].append(weighted_y)

            # Store individual components for interpolation
            coordinates[tracked_id]['hip_x'].append(hip_x if hip_x is not None else 0)
            coordinates[tracked_id]['hip_y'].append(hip_y if hip_y is not None else 0)
            coordinates[tracked_id]['left_ankle_x'].append(left_ankle_x if left_ankle_x is not None else 0)
            coordinates[tracked_id]['left_ankle_y'].append(left_ankle_y if left_ankle_y is not None else 0)
            coordinates[tracked_id]['right_ankle_x'].append(right_ankle_x if right_ankle_x is not None else 0)
            coordinates[tracked_id]['right_ankle_y'].append(right_ankle_y if right_ankle_y is not None else 0)

    # Convert to numpy arrays and sort
    for tracked_id in coordinates:
        coord = coordinates[tracked_id]
        sort_idx = np.argsort(coord['frames'])

        for key in coord:
            coord[key] = np.array(coord[key])[sort_idx]

    return dict(coordinates)

def find_peaks_and_valleys(y_data, frames, prominence_threshold=0.3, min_distance=10):
    """Find peaks (jumps) and valleys (beginnings/ends)."""
    try:
        from scipy import signal
    except ImportError:
        print("❌ Error: scipy not available. Install with: pip install scipy")
        return [], []

    if len(y_data) < 10:
        return [], []

    # Smooth data
    smoothed = np.convolve(y_data, np.ones(5)/5, mode='same')

    # Find peaks (jumps)
    peak_indices, _ = signal.find_peaks(smoothed, prominence=prominence_threshold, distance=min_distance)
    peaks = [(frames[idx], y_data[idx], idx) for idx in peak_indices if 0 <= idx < len(frames)]

    # Find valleys (beginnings/ends)
    inverted_data = -smoothed
    valley_indices, _ = signal.find_peaks(inverted_data, prominence=prominence_threshold*0.7, distance=min_distance//2)
    valleys = [(frames[idx], y_data[idx], idx) for idx in valley_indices if 0 <= idx < len(frames)]

    return peaks, valleys

def correlate_ml_jumps_with_position(ml_jumps, coordinates, proximity_threshold=15):
    """Correlate ML-detected jumps with position data jumps using weighted coordinates."""
    correlations = []

    for ml_jump in ml_jumps:
        ml_frame = ml_jump['frame']

        # Find jumps in position data near ML jump
        for player_id, coord_data in coordinates.items():
            frames = coord_data['frames']
            weighted_x = coord_data['weighted_x']
            weighted_y = coord_data['weighted_y']

            if len(weighted_y) < 10:
                continue

            # Use weighted Y coordinates for peak/valley detection
            peaks, valleys = find_peaks_and_valleys(weighted_y, frames)

            # Find closest valley (minimum) to ML jump - jumps occur at minima
            nearby_valleys = [(f, y, idx) for f, y, idx in valleys
                              if abs(f - ml_frame) <= proximity_threshold]

            if nearby_valleys:
                closest_valley = min(nearby_valleys, key=lambda x: abs(x[0] - ml_frame))
                valley_frame, valley_y, valley_idx = closest_valley

                # Find neighboring peaks as beginning and end
                before_peaks = [(f, y, idx) for f, y, idx in peaks if f < valley_frame]
                after_peaks = [(f, y, idx) for f, y, idx in peaks if f > valley_frame]

                if before_peaks and after_peaks:
                    begin = max(before_peaks, key=lambda x: x[0])  # Latest peak before
                    end = min(after_peaks, key=lambda x: x[0])     # Earliest peak after

                    # Get weighted x coordinates for begin and end frames
                    begin_frame_idx = np.where(frames == begin[0])[0]
                    end_frame_idx = np.where(frames == end[0])[0]

                    if len(begin_frame_idx) > 0 and len(end_frame_idx) > 0:
                        begin_x = weighted_x[begin_frame_idx[0]]
                        end_x = weighted_x[end_frame_idx[0]]

                        correlations.append({
                            'player_id': player_id,
                            'ml_frame': ml_frame,
                            'ml_confidence': ml_jump['confidence'],
                            'begin_frame': begin[0],
                            'begin_x': begin_x,
                            'begin_y': begin[1],
                            'jump_frame': valley_frame,  # Jump is at the valley
                            'jump_y': valley_y,
                            'end_frame': end[0],
                            'end_x': end_x,
                            'end_y': end[1],
                            'distance_to_ml': abs(valley_frame - ml_frame)
                        })

    # Deduplicate - keep best ML confidence for each unique jump sequence
    unique_sequences = {}
    for corr in correlations:
        # Create key for unique sequence (player + jump frame)
        key = (corr['player_id'], corr['jump_frame'])

        if key not in unique_sequences or corr['ml_confidence'] > unique_sequences[key]['ml_confidence']:
            unique_sequences[key] = corr

    deduped_correlations = list(unique_sequences.values())

    print(f"Before deduplication: {len(correlations)} correlations")
    print(f"After deduplication: {len(deduped_correlations)} unique sequences")

    return deduped_correlations

def interpolate_weighted_positions(begin_data, end_data, frame, begin_frame, end_frame):
    """
    Interpolate all position components and recalculate weighted position.
    This maintains consistency with the original weighting scheme.
    """
    if end_frame == begin_frame:
        t = 0
    else:
        t = (frame - begin_frame) / (end_frame - begin_frame)

    # Interpolate all components
    interp_hip_x = begin_data['hip_x'] + t * (end_data['hip_x'] - begin_data['hip_x'])
    interp_hip_y = begin_data['hip_y'] + t * (end_data['hip_y'] - begin_data['hip_y'])
    interp_left_ankle_x = begin_data['left_ankle_x'] + t * (end_data['left_ankle_x'] - begin_data['left_ankle_x'])
    interp_left_ankle_y = begin_data['left_ankle_y'] + t * (end_data['left_ankle_y'] - begin_data['left_ankle_y'])
    interp_right_ankle_x = begin_data['right_ankle_x'] + t * (end_data['right_ankle_x'] - begin_data['right_ankle_x'])
    interp_right_ankle_y = begin_data['right_ankle_y'] + t * (end_data['right_ankle_y'] - begin_data['right_ankle_y'])

    return {
        'hip_x': float(interp_hip_x),
        'hip_y': float(interp_hip_y),
        'left_ankle_x': float(interp_left_ankle_x),
        'left_ankle_y': float(interp_left_ankle_y),
        'right_ankle_x': float(interp_right_ankle_x),
        'right_ankle_y': float(interp_right_ankle_y)
    }

def save_corrected_positions(original_position_data, correlations, coordinates, output_dir):
    """Save corrected position data with interpolated jump sequences using weighted averaging."""
    if not correlations:
        print("No correlations to process")
        return

    # Create a copy of the original data
    corrected_data = json.loads(json.dumps(original_position_data))

    # Create a mapping of (player_id, frame) -> position index for quick lookup
    position_lookup = {}
    for idx, pos in enumerate(corrected_data["player_positions"]):
        key = (pos.get('tracked_id'), pos['frame_index'])
        position_lookup[key] = idx

    # Process each correlation (jump sequence)
    for corr in correlations:
        player_id = corr['player_id']
        begin_frame = corr['begin_frame']
        end_frame = corr['end_frame']

        print(f"Interpolating Player {player_id}: frames {begin_frame} to {end_frame}")

        # Get coordinate data for this player
        coord_data = coordinates[player_id]
        frames = coord_data['frames']

        # Find begin and end frame indices in coordinate data
        begin_idx = np.where(frames == begin_frame)[0]
        end_idx = np.where(frames == end_frame)[0]

        if len(begin_idx) == 0 or len(end_idx) == 0:
            print(f"Warning: Could not find coordinate data for player {player_id} frames {begin_frame}-{end_frame}")
            continue

        begin_idx = begin_idx[0]
        end_idx = end_idx[0]

        # Extract begin and end position data
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

        # Interpolate for all frames between begin and end (inclusive)
        for frame in range(begin_frame, end_frame + 1):
            # Calculate interpolated position components
            interpolated = interpolate_weighted_positions(begin_data, end_data, frame, begin_frame, end_frame)

            # Update the position data if this frame exists
            key = (player_id, frame)
            if key in position_lookup:
                pos_idx = position_lookup[key]
                pos = corrected_data["player_positions"][pos_idx]

                # Update all position components
                pos['hip_world_X'] = interpolated['hip_x']
                pos['hip_world_Y'] = interpolated['hip_y']
                pos['left_ankle_world_X'] = interpolated['left_ankle_x']
                pos['left_ankle_world_Y'] = interpolated['left_ankle_y']
                pos['right_ankle_world_X'] = interpolated['right_ankle_x']
                pos['right_ankle_world_Y'] = interpolated['right_ankle_y']

    # Save corrected data
    output_path = output_dir / "corrected_positions.json"
    with open(output_path, 'w') as f:
        json.dump(corrected_data, f, indent=2)

    print(f"✅ Saved corrected positions to: {output_path}")

    # Print summary
    total_interpolated_frames = sum(corr['end_frame'] - corr['begin_frame'] + 1 for corr in correlations)
    print(f"Summary: Interpolated {total_interpolated_frames} frames across {len(correlations)} jump sequences")
    print("All position components (hip and ankles) were interpolated to maintain weighted averaging consistency")

def main():
    parser = argparse.ArgumentParser(description='Unified jump analysis using ML detection and weighted position data')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--model', default='roboflow/runs/detect/badminton_poses/weights/best.pt',
                        help='Path to YOLO model weights')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Confidence threshold for ML detections')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                        help='Force specific device')
    parser.add_argument('--proximity', type=int, default=15,
                        help='Frame proximity for correlating ML and position jumps')

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        return 1

    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return 1

    print(f"System: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")

    try:
        # Step 1: Use ML model to detect valid jumps
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

        if not ml_jumps:
            print("No ML jumps detected. Exiting.")
            return 0

        # Step 2: Load position data
        print("\n" + "="*60)
        print("STEP 2: WEIGHTED POSITION DATA ANALYSIS")
        print("="*60)

        position_data, output_dir = load_position_data(args.video_path)
        if not position_data or "player_positions" not in position_data:
            print("Position data unavailable. Run position analysis first.")
            return 1

        coordinates = extract_player_coordinates(position_data["player_positions"])
        print(f"Found {len(coordinates)} players with weighted position data")

        # Step 3: Correlate ML jumps with position data using weighted coordinates
        print("\n" + "="*60)
        print("STEP 3: CORRELATION USING WEIGHTED POSITIONS")
        print("="*60)

        correlations = correlate_ml_jumps_with_position(ml_jumps, coordinates, args.proximity)

        print(f"Found {len(correlations)} correlations:")
        for i, corr in enumerate(correlations, 1):
            print(f"  {i}. Player {corr['player_id']} | "
                  f"ML: Frame {corr['ml_frame']} | "
                  f"Weighted Position Jump: Frame {corr['jump_frame']} | "
                  f"Sequence: {corr['begin_frame']}→{corr['jump_frame']}→{corr['end_frame']} | "
                  f"Distance: ±{corr['distance_to_ml']} frames")

        # Step 4: Save corrected positions with weighted interpolation
        if correlations:
            print("\n" + "="*60)
            print("STEP 4: SAVING CORRECTED POSITIONS WITH WEIGHTED INTERPOLATION")
            print("="*60)
            save_corrected_positions(position_data, correlations, coordinates, output_dir)
        else:
            print("No correlations found. No corrections to apply.")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())