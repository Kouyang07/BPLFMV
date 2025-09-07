#!/usr/bin/env python3

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


class PhysicsHeightEstimator:
    def __init__(self, gravity=9.81, fps=30.0):
        self.gravity = gravity
        self.fps = fps
        self.frame_time = 1.0 / fps

    def estimate_player_height(self, begin_frame, jump_frame, end_frame, target_frame, max_jump_height=1.0):
        if target_frame < begin_frame or target_frame > end_frame:
            return 0.0

        if begin_frame == end_frame:
            return 0.0

        total_time = (end_frame - begin_frame) * self.frame_time
        jump_time = (jump_frame - begin_frame) * self.frame_time
        target_time = (target_frame - begin_frame) * self.frame_time

        if jump_time <= 0 or jump_time >= total_time:
            return 0.0

        v0 = np.sqrt(2 * self.gravity * max_jump_height)
        height = v0 * target_time - 0.5 * self.gravity * target_time**2
        height = max(0.0, min(height, max_jump_height))

        expected_peak_time = v0 / self.gravity
        if abs(expected_peak_time - jump_time) > 0.1:
            v0_adjusted = self.gravity * jump_time
            height = v0_adjusted * target_time - 0.5 * self.gravity * target_time**2
            height = max(0.0, min(height, max_jump_height))

        return float(height)

    def project_elevated_player_to_ground(self, pixel_x, pixel_y, player_height, camera_params):
        if player_height <= 0.0:
            return pixel_x, pixel_y

        camera_height = camera_params.get('height_m', 0.0)
        fx = camera_params.get('fx', 1000.0)
        fy = camera_params.get('fy', 1000.0)
        cx = camera_params.get('cx', 0.0)
        cy = camera_params.get('cy', 0.0)

        if camera_height <= 0:
            return pixel_x, pixel_y

        height_ratio = player_height / camera_height
        corrected_x = cx + (pixel_x - cx) * (1 + height_ratio)
        corrected_y = cy + (pixel_y - cy) * (1 + height_ratio)

        return float(corrected_x), float(corrected_y)

    def calculate_jump_physics_metrics(self, begin_data, end_data, jump_data, begin_frame, jump_frame, end_frame):
        total_time = (end_frame - begin_frame) * self.frame_time
        jump_time = (jump_frame - begin_frame) * self.frame_time

        baseline_y = max(begin_data['y'], end_data['y'])
        jump_height = baseline_y - jump_data['y']

        if jump_height < 0:
            baseline_y = min(begin_data['y'], end_data['y'])
            jump_height = jump_data['y'] - baseline_y

        if jump_height < 0:
            jump_height = abs(jump_height)

        dx = end_data['x'] - begin_data['x']
        vx = dx / total_time if total_time > 0 else 0
        v0y = self.gravity * jump_time if jump_time > 0 else 0
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


def get_device():
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS")
        return 'mps'
    print("Using CPU")
    return 'cpu'


def load_model(model_path, device=None):
    if device is None:
        device = get_device()
    model = YOLO(model_path)
    model.to(device)
    return model, device


def detect_jumps(video_path, model_path, conf_thresh=0.5, device=None):
    model, device_used = load_model(model_path, device)

    classes = {
        0: 'BackhandClear', 1: 'BackhandLift', 2: 'BackhandServe',
        3: 'ForehandClear', 4: 'ForehandLift', 5: 'ReadyPosition', 6: 'Smash'
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Processing {total_frames} frames, {fps} FPS")

    detections = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Processing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thresh, verbose=False)
        frame_dets = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = classes.get(class_id, 'Unknown')

                bbox = box.xyxy[0].cpu().numpy()
                bottom_x = float((bbox[0] + bbox[2]) / 2)
                bottom_y = float(bbox[3])

                frame_dets.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'frame': frame_idx,
                    'bbox': bbox.tolist(),
                    'bottom_center_x': bottom_x,
                    'bottom_center_y': bottom_y
                })

        detections.append(frame_dets)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    valid_smashes = find_smash_sequences(detections)
    return valid_smashes, fps


def find_smash_sequences(detections, window=10):
    valid = []

    smash_frames = []
    for i, dets in enumerate(detections):
        for det in dets:
            if det['class_name'] == 'Smash':
                smash_frames.append((i, det))

    print(f"Found {len(smash_frames)} smash detections")

    for frame_idx, smash in tqdm(smash_frames, desc="Checking sequences"):
        start = max(0, frame_idx - window)
        end = min(len(detections), frame_idx + window + 1)

        smash_x = smash['bottom_center_x']
        smash_y = smash['bottom_center_y']
        thresh = 200

        before = any(
            any(
                d['class_name'] == 'ForehandClear' and
                abs(d['bottom_center_x'] - smash_x) < thresh and
                abs(d['bottom_center_y'] - smash_y) < thresh
                for d in detections[i]
            )
            for i in range(start, frame_idx)
        )

        after = any(
            any(
                d['class_name'] == 'ForehandClear' and
                abs(d['bottom_center_x'] - smash_x) < thresh and
                abs(d['bottom_center_y'] - smash_y) < thresh
                for d in detections[i]
            )
            for i in range(frame_idx + 1, end)
        )

        if before and after:
            valid.append({
                'frame': frame_idx,
                'confidence': smash['confidence'],
                'bottom_center_x': smash_x,
                'bottom_center_y': smash_y,
                'bbox': smash['bbox']
            })

    print(f"Valid sequences: {len(valid)}")
    return valid


def load_positions(video_path):
    base_name = Path(video_path).stem
    result_dir = Path("results") / base_name
    pos_file = result_dir / "positions.json"

    try:
        with open(pos_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded positions from {pos_file}")

        court_points = {}
        pose_file = result_dir / "pose.json"

        if pose_file.exists():
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
            if 'court_points' in pose_data:
                court_points = pose_data['court_points']
            elif 'all_court_points' in pose_data:
                court_points = pose_data['all_court_points']

        if not court_points:
            court_csv = result_dir / "court.csv"
            if court_csv.exists():
                court_points = load_court_csv(court_csv)

        homography = calc_homography(court_points) if court_points else None
        return data, result_dir, homography

    except Exception as e:
        print(f"Error loading positions: {e}")
        return None, None, None


def load_camera_params(video_path):
    base_name = Path(video_path).stem
    result_dir = Path("results") / base_name
    cal_file = result_dir / "calibration.csv"

    params = {}
    try:
        import csv
        with open(cal_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row['parameter']
                val = row['value']

                if val.startswith('[') and val.endswith(']'):
                    val = val[1:-1]

                try:
                    params[param] = float(val)
                except:
                    params[param] = val

        print(f"Camera height: {params.get('height_m', 'N/A')}m")
        return params

    except:
        print("No camera calibration found")
        return {}


def load_court_csv(csv_path):
    import csv
    points = {}

    with open(csv_path, 'r') as f:
        first = f.readline().strip()
        f.seek(0)

        if 'Point' in first:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row['Point'].strip()
                    x = float(row['X'])
                    y = float(row['Y'])
                    points[name] = [x, y]
                except:
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        points[row[0].strip()] = [float(row[1]), float(row[2])]
                    except:
                        continue

    return points


def calc_homography(court_points):
    COURT_W = 6.1
    COURT_L = 13.4

    req_points = ['P1', 'P2', 'P3', 'P4']
    if not all(p in court_points for p in req_points):
        raise ValueError(f"Missing points: {req_points}")

    img_pts = []
    for pt in req_points:
        coords = court_points[pt]
        if isinstance(coords, list):
            img_pts.append([coords[0], coords[1]])
        else:
            img_pts.append([coords['x'], coords['y']])

    img_pts = np.array(img_pts, dtype=np.float32)

    world_pts = np.array([
        [0, 0], [0, COURT_L], [COURT_W, COURT_L], [COURT_W, 0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
    if H is None:
        raise ValueError("Homography failed")

    return H


def pixel_to_court(pixel_point, H):
    pt = np.array([[pixel_point]], dtype=np.float32)
    world_pt = cv2.perspectiveTransform(pt, H)
    return float(world_pt[0][0][0]), float(world_pt[0][0][1])


def get_player_pos(pos_data, frame, player_id, H):
    if 'frame_data' in pos_data:
        frame_key = str(frame)
        if frame_key in pos_data['frame_data']:
            frame_data = pos_data['frame_data'][frame_key]
            player_key = f"player_{player_id}"

            if player_key in frame_data and 'center_position' in frame_data[player_key]:
                center = frame_data[player_key]['center_position']
                return float(center['x']), float(center['y'])

            if player_key in frame_data and 'ankles' in frame_data[player_key]:
                ankles = frame_data[player_key]['ankles']
                if ankles:
                    return float(ankles[0]['world_x']), float(ankles[0]['world_y'])

    if 'player_positions' in pos_data:
        for pos in pos_data["player_positions"]:
            if pos['frame_index'] == frame and pos.get('player_id') == player_id:
                if 'x' in pos and 'y' in pos:
                    return float(pos['x']), float(pos['y'])
                elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                    return float(pos['hip_world_X']), float(pos['hip_world_Y'])

    return None, None


def extract_coords(pos_data):
    coords = defaultdict(lambda: {'frames': [], 'x': [], 'y': []})

    if 'frame_data' in pos_data:
        for frame_str, players in tqdm(pos_data['frame_data'].items()):
            frame_idx = int(frame_str)

            for player_key, player_data in players.items():
                if not player_key.startswith('player_'):
                    continue

                player_id = int(player_key.split('_')[1])

                if 'center_position' in player_data:
                    center = player_data['center_position']
                    x, y = center['x'], center['y']
                elif 'ankles' in player_data and player_data['ankles']:
                    ankles = player_data['ankles']
                    x = sum(a['world_x'] for a in ankles) / len(ankles)
                    y = sum(a['world_y'] for a in ankles) / len(ankles)
                else:
                    continue

                coords[player_id]['frames'].append(frame_idx)
                coords[player_id]['x'].append(x)
                coords[player_id]['y'].append(y)

    elif 'player_positions' in pos_data:
        for pos in tqdm(pos_data['player_positions']):
            player_id = pos.get('player_id')
            if player_id is None:
                continue

            frame_idx = pos['frame_index']

            if 'x' in pos and 'y' in pos:
                x, y = pos['x'], pos['y']
            elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                x, y = pos['hip_world_X'], pos['hip_world_Y']
            else:
                continue

            if x is not None and y is not None:
                coords[player_id]['frames'].append(frame_idx)
                coords[player_id]['x'].append(x)
                coords[player_id]['y'].append(y)

    for player_id in coords:
        coord = coords[player_id]
        if len(coord['frames']) > 0:
            sort_idx = np.argsort(coord['frames'])
            for key in coord:
                coord[key] = np.array(coord[key])[sort_idx]

    return dict(coords)


def find_peaks_valleys(y_data, frames, prom=0.2, min_dist=10):
    try:
        from scipy import signal
    except ImportError:
        print("Need scipy")
        return [], []

    if len(y_data) < 10:
        return [], []

    smoothed = np.convolve(y_data, np.ones(5)/5, mode='same')
    peak_idx, _ = signal.find_peaks(smoothed, prominence=prom, distance=min_dist)
    peaks = [(frames[i], y_data[i], i) for i in peak_idx if 0 <= i < len(frames)]

    valley_idx, _ = signal.find_peaks(-smoothed, prominence=prom*0.7, distance=min_dist//2)
    valleys = [(frames[i], y_data[i], i) for i in valley_idx if 0 <= i < len(frames)]

    return peaks, valleys


def correlate_jumps(ml_jumps, coords, pos_data, H, prox_thresh=15, dist_thresh=1.5):
    correlations = []

    if not ml_jumps:
        return correlations

    print("Correlating jumps...")
    for ml_jump in tqdm(ml_jumps):
        ml_frame = ml_jump['frame']
        ml_x = ml_jump['bottom_center_x']
        ml_y = ml_jump['bottom_center_y']

        ml_court_x, ml_court_y = pixel_to_court((ml_x, ml_y), H)
        print(f"\nML jump frame {ml_frame}: ({ml_x:.1f}, {ml_y:.1f}) -> court ({ml_court_x:.2f}, {ml_court_y:.2f})")

        best_player = None
        min_dist = float('inf')
        player_dists = {}

        for player_id, coord_data in coords.items():
            px, py = get_player_pos(pos_data, ml_frame, player_id, H)

            if px is None:
                for offset in range(-3, 4):
                    test_frame = ml_frame + offset
                    px, py = get_player_pos(pos_data, test_frame, player_id, H)
                    if px is not None:
                        break

            if px is not None:
                court_dist = np.sqrt((ml_court_x - px)**2 + (ml_court_y - py)**2)
                player_dists[player_id] = court_dist

                if court_dist < min_dist:
                    min_dist = court_dist
                    best_player = player_id

        for pid, dist in player_dists.items():
            marker = " <- BEST" if pid == best_player else ""
            print(f"  Player {pid}: {dist:.2f}m{marker}")

        if best_player is not None and min_dist < dist_thresh:
            print(f"  Using player {best_player} (dist: {min_dist:.2f}m)")

            coord_data = coords[best_player]
            frames = coord_data['frames']
            y_coords = coord_data['y']

            if len(y_coords) >= 10:
                peaks, valleys = find_peaks_valleys(y_coords, frames)

                nearby_valleys = [(f, y, idx) for f, y, idx in valleys
                                  if abs(f - ml_frame) <= prox_thresh]

                if nearby_valleys:
                    closest_valley = min(nearby_valleys, key=lambda x: abs(x[0] - ml_frame))
                    valley_frame, valley_y, valley_idx = closest_valley

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

                            ext_begin = max(0, begin[0] - 5)
                            ext_end = end[0] + 5

                            correlations.append({
                                'player_id': best_player,
                                'ml_frame': ml_frame,
                                'ml_confidence': ml_jump['confidence'],
                                'court_distance': min_dist,
                                'begin_frame': ext_begin,
                                'begin_x': begin_x,
                                'begin_y': begin[1],
                                'jump_frame': valley_frame,
                                'jump_x': x_coords[valley_idx] if valley_idx < len(x_coords) else (begin_x + end_x) / 2,
                                'jump_y': valley_y,
                                'end_frame': ext_end,
                                'end_x': end_x,
                                'end_y': end[1],
                                'distance_to_ml': abs(valley_frame - ml_frame)
                            })

                            print(f"  Jump sequence: {begin[0]}->{valley_frame}->{end[0]} (ext: {ext_begin}->{ext_end})")

    return correlations


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


def save_corrected(orig_data, correlations, coords, output_dir, fps, camera_params=None):
    corrected = convert_numpy(json.loads(json.dumps(orig_data, default=str)))

    physics = PhysicsHeightEstimator(gravity=9.81, fps=fps)

    corrected['correction_metadata'] = {
        'jumps_detected': len(correlations),
        'correction_applied': len(correlations) > 0,
        'tracking_method': 'ankle_only_height_corrected',
        'correction_method': 'projectile_motion_height_estimation',
        'camera_calibration_available': bool(camera_params),
        'gravity': physics.gravity,
        'fps': fps,
        'timestamp': str(np.datetime64('now'))
    }

    if not correlations:
        print("No correlations - saving original")
        out_path = output_dir / "corrected_positions.json"
        with open(out_path, 'w') as f:
            json.dump(corrected, f, indent=2)
        return

    total_corrected = 0
    jump_stats = []

    print("Applying corrections...")
    for corr in tqdm(correlations):
        player_id = corr['player_id']
        begin_frame = corr['begin_frame']
        end_frame = corr['end_frame']
        jump_frame = corr['jump_frame']

        print(f"\nPlayer {player_id}: frames {begin_frame}-{end_frame}, peak {jump_frame}")

        begin_data = {'x': corr['begin_x'], 'y': corr['begin_y']}
        end_data = {'x': corr['end_x'], 'y': corr['end_y']}
        jump_data = {'x': corr['jump_x'], 'y': corr['jump_y']}

        metrics = physics.calculate_jump_physics_metrics(
            begin_data, end_data, jump_data, begin_frame, jump_frame, end_frame
        )

        metrics = convert_numpy(metrics)
        max_height = metrics.get('jump_height', 0.5)
        max_height = max(0.2, min(max_height, 1.5))

        ml_frame = corr.get('ml_frame')

        jump_stats.append({
            'player_id': int(player_id),
            'jump_frame': int(jump_frame),
            'ml_frame': int(ml_frame) if ml_frame else None,
            'trajectory_weight': float(corr.get('trajectory_weight', 0.0)),
            'ml_weight': float(corr.get('ml_weight', 0.0)),
            'physics_metrics': metrics,
            'max_estimated_height': float(max_height)
        })

        print(f"  Max height: {max_height:.2f}m")

        frames_fixed = 0

        if 'frame_data' in corrected:
            for frame in range(begin_frame, end_frame + 1):
                height = physics.estimate_player_height(
                    begin_frame, jump_frame, end_frame, frame, max_height
                )

                frame_key = str(frame)
                player_key = f"player_{player_id}"

                if frame_key in corrected['frame_data'] and player_key in corrected['frame_data'][frame_key]:
                    player_data = corrected['frame_data'][frame_key][player_key]

                    current_pos = player_data.get('center_position', {})
                    if 'x' in current_pos and 'y' in current_pos:
                        if camera_params and height > 0.01:
                            frame_ratio = (frame - begin_frame) / max(1, end_frame - begin_frame)
                            ground_x = begin_data['x'] + frame_ratio * (end_data['x'] - begin_data['x'])
                            ground_y = begin_data['y'] + frame_ratio * (end_data['y'] - begin_data['y'])

                            corrected['frame_data'][frame_key][player_key]['center_position']['x'] = float(ground_x)
                            corrected['frame_data'][frame_key][player_key]['center_position']['y'] = float(ground_y)

                        player_data['estimated_height'] = float(height)

                        if 'ankles' in player_data:
                            for ankle in player_data['ankles']:
                                if height > 0.01 and camera_params:
                                    frame_ratio = (frame - begin_frame) / max(1, end_frame - begin_frame)
                                    ground_x = begin_data['x'] + frame_ratio * (end_data['x'] - begin_data['x'])
                                    ground_y = begin_data['y'] + frame_ratio * (end_data['y'] - begin_data['y'])
                                    ankle['world_x'] = float(ground_x)
                                    ankle['world_y'] = float(ground_y)
                                ankle['estimated_height'] = float(height)

                        frames_fixed += 1

        elif 'player_positions' in corrected:
            pos_lookup = {}
            for idx, pos in enumerate(corrected["player_positions"]):
                key = (pos.get('player_id'), pos['frame_index'])
                pos_lookup[key] = idx

            for frame in range(begin_frame, end_frame + 1):
                height = physics.estimate_player_height(
                    begin_frame, jump_frame, end_frame, frame, max_height
                )

                key = (player_id, frame)
                if key in pos_lookup:
                    pos_idx = pos_lookup[key]
                    pos = corrected["player_positions"][pos_idx]

                    if height > 0.01 and camera_params:
                        frame_ratio = (frame - begin_frame) / max(1, end_frame - begin_frame)
                        ground_x = begin_data['x'] + frame_ratio * (end_data['x'] - begin_data['x'])
                        ground_y = begin_data['y'] + frame_ratio * (end_data['y'] - begin_data['y'])

                        if 'x' in pos and 'y' in pos:
                            pos['x'] = float(ground_x)
                            pos['y'] = float(ground_y)
                        elif 'hip_world_X' in pos and 'hip_world_Y' in pos:
                            pos['hip_world_X'] = float(ground_x)
                            pos['hip_world_Y'] = float(ground_y)

                    pos['estimated_height'] = float(height)
                    frames_fixed += 1

        print(f"  Fixed {frames_fixed} frames")
        total_corrected += frames_fixed

    corrected['correction_metadata']['jump_physics_stats'] = convert_numpy(jump_stats)

    out_path = output_dir / "corrected_positions.json"
    with open(out_path, 'w') as f:
        json.dump(corrected, f, indent=2)

    physics_path = output_dir / "jump_physics_analysis.json"
    physics_report = convert_numpy({
        'analysis_metadata': {
            'total_jumps_analyzed': len(correlations),
            'total_frames_corrected': int(total_corrected),
            'physics_model': {
                'gravity': float(physics.gravity),
                'fps': float(fps),
                'correction_method': 'projectile_motion_height_estimation'
            },
            'camera_calibration': camera_params if camera_params else 'not_available',
            'timestamp': str(np.datetime64('now'))
        },
        'jump_analyses': jump_stats
    })

    with open(physics_path, 'w') as f:
        json.dump(physics_report, f, indent=2)

    print(f"\nSaved to: {out_path}")
    print(f"Physics report: {physics_path}")
    print(f"Corrected {total_corrected} frames across {len(correlations)} jumps")

    if jump_stats:
        avg_height = np.mean([s['physics_metrics']['jump_height'] for s in jump_stats])
        avg_vel = np.mean([s['physics_metrics']['horizontal_velocity'] for s in jump_stats])
        print(f"Avg jump height: {avg_height:.2f}m")
        print(f"Avg horizontal vel: {avg_vel:.2f}m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--model', default='resources/BLPFMV.pt')
    parser.add_argument('--confidence', type=float, default=0.2)
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--proximity', type=int, default=15)
    parser.add_argument('--court-distance', type=float, default=4)
    parser.add_argument('--gravity', type=float, default=9.81)

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"Video not found: {args.video_path}")
        return 1

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return 1

    print(f"System: {platform.system()}, PyTorch: {torch.__version__}")
    print(f"Gravity: {args.gravity} m/s²")

    try:
        print("\n" + "="*40)
        print("ML JUMP DETECTION")
        print("="*40)

        jumps, fps = detect_jumps(args.video_path, args.model, args.confidence, args.device)

        print(f"\nDetected {len(jumps)} jumps:")
        for i, jump in enumerate(jumps, 1):
            frame = jump['frame']
            conf = jump['confidence']
            time = frame / fps
            print(f"  {i}. Frame {frame:4d} | {time:6.2f}s | {conf:.3f}")

        print("\n" + "="*40)
        print("LOADING DATA")
        print("="*40)

        pos_data, out_dir, H = load_positions(args.video_path)
        if not pos_data:
            print("No position data")
            return 1

        coords = extract_coords(pos_data)
        print(f"Players: {len(coords)}")

        camera_params = load_camera_params(args.video_path)

        correlations = []
        if H is not None:
            print("\n" + "="*40)
            print("CORRELATING")
            print("="*40)

            correlations = correlate_jumps(
                jumps, coords, pos_data, H, args.proximity, args.court_distance
            )

            print(f"\nCorrelated {len(correlations)} jumps:")
            for i, corr in enumerate(correlations, 1):
                print(f"  {i}. Player {corr['player_id']} | "
                      f"{corr['begin_frame']}->{corr['jump_frame']}->{corr['end_frame']} | "
                      f"ML: {corr['ml_frame']} | Dist: {corr['court_distance']:.2f}m")

        print("\n" + "="*40)
        print("CORRECTIONS")
        print("="*40)

        save_corrected(pos_data, correlations, coords, out_dir, fps, camera_params)

        print("\n" + "="*40)
        print("DONE")
        print("="*40)

        if correlations:
            print(f"Processed {len(jumps)} ML detections")
            print(f"Applied {len(correlations)} corrections")
            print(f"Physics: projectile motion, g = {args.gravity}")
            print(f"Camera cal: {'Yes' if camera_params else 'No'}")
        else:
            print(f"Processed {len(jumps)} detections, no corrections")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())