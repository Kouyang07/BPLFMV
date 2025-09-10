#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import json
import sys
import os
import csv
import logging
from tqdm import tqdm
from ultralytics import YOLO
import warnings
import math

warnings.filterwarnings("ignore")
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
MAX_JUMP_HEIGHT = 0.75
ANKLE_KNEE_INDICES = [13, 14, 15, 16]
MIN_ANKLE_KNEE_JOINTS = 2
CONFIDENCE_THRESHOLD = 0.5

def read_calibration_csv(csv_path):
    data = {'camera_matrix': None, 'dist_coeffs': None, 'rvec': None, 'tvec': None, 'reprojection_error': None, 'strategy': None}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if len(row) < 2:
                continue
            key, value = row[0].strip(), row[1].strip()

            try:
                def parse_float(val):
                    val = val.strip('[]')
                    return float(val)

                if key == 'fx':
                    data['camera_matrix'] = np.zeros((3, 3), dtype=np.float32)
                    data['camera_matrix'][2, 2] = 1.0
                    data['camera_matrix'][0, 0] = parse_float(value)
                elif key == 'fy':
                    data['camera_matrix'][1, 1] = parse_float(value)
                elif key == 'cx':
                    data['camera_matrix'][0, 2] = parse_float(value)
                elif key == 'cy':
                    data['camera_matrix'][1, 2] = parse_float(value)
                elif key.startswith('dist_'):
                    if data['dist_coeffs'] is None:
                        data['dist_coeffs'] = np.zeros(5, dtype=np.float32)
                    idx = int(key.split('_')[1])
                    if idx < 5:
                        data['dist_coeffs'][idx] = parse_float(value)
                elif key == 'rx':
                    data['rvec'] = np.zeros(3, dtype=np.float32)
                    data['rvec'][0] = parse_float(value)
                elif key == 'ry':
                    data['rvec'][1] = parse_float(value)
                elif key == 'rz':
                    data['rvec'][2] = parse_float(value)
                elif key == 'tx':
                    data['tvec'] = np.zeros(3, dtype=np.float32)
                    data['tvec'][0] = parse_float(value)
                elif key == 'ty':
                    data['tvec'][1] = parse_float(value)
                elif key == 'tz':
                    data['tvec'][2] = parse_float(value)
                elif key == 'error_px':
                    data['reprojection_error'] = parse_float(value)
                elif key == 'strategy':
                    data['strategy'] = value
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse row {row}: {e}")

    print(f"Loaded calibration - Camera: {'OK' if data['camera_matrix'] is not None else 'Missing'}, "
          f"Distortion: {'OK' if data['dist_coeffs'] is not None else 'Missing'}, "
          f"Pose: {'OK' if data['rvec'] is not None else 'Missing'}")
    if data['reprojection_error']:
        print(f"Error: {data['reprojection_error']:.2f}px, Strategy: {data['strategy']}")
    return data

def read_detections_csv(csv_path):
    points = {}
    if not os.path.exists(csv_path):
        print(f"Detections CSV not found: {csv_path}")
        return points

    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)

        if 'Point' in first_line and 'X' in first_line:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    points[row['Point'].strip()] = [float(row['X']), float(row['Y'])]
                except (ValueError, KeyError):
                    pass
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        points[row[0].strip()] = [float(row[1]), float(row[2])]
                    except (ValueError, IndexError):
                        pass

    print(f"Loaded {len(points)} court points")
    return points

def extract_corners(all_points):
    corners = {}
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_points:
            corners[point_name] = all_points[point_name]

    if len(corners) == 4:
        return corners

    point_items = list(all_points.items())[:4]
    return {f"P{i+1}": coords for i, (_, coords) in enumerate(point_items)}

def calc_elevation_angle(rvec, tvec):
    try:
        rmat, _ = cv2.Rodrigues(rvec)
        cam_pos = -rmat.T @ tvec
        h = abs(cam_pos[2])
        court_center = np.array([COURT_WIDTH/2, COURT_LENGTH/2, 0])
        d = np.linalg.norm(cam_pos[:2] - court_center[:2])
        theta = np.arctan(h / max(d, 0.1))
        return theta, h, d
    except:
        return np.radians(30), 5.0, 10.0

def create_perspective_boundary(corners_2d, rvec, tvec):
    try:
        theta, h_cam, d_court = calc_elevation_angle(rvec, tvec)
        rmat, _ = cv2.Rodrigues(rvec)
        cam_pos = -rmat.T @ tvec

        regions = {
            'top': np.array([COURT_WIDTH/2, 0.0, 0.0]),
            'bottom': np.array([COURT_WIDTH/2, COURT_LENGTH, 0.0])
        }

        extensions = {}
        for region, center_3d in regions.items():
            distance = np.linalg.norm(cam_pos[:2] - center_3d[:2])
            ext_dist = (MAX_JUMP_HEIGHT / np.tan(theta)) * (distance / d_court)

            court_px = np.linalg.norm(np.array(corners_2d['P2']) - np.array(corners_2d['P1']))
            px_per_m = court_px / COURT_LENGTH
            ext_px = ext_dist * px_per_m
            extensions[region] = min(ext_px, court_px * 0.3)

        enlarged = {}
        center_x = np.mean([corners_2d[f'P{i}'][0] for i in range(1, 5)])
        center_y = np.mean([corners_2d[f'P{i}'][1] for i in range(1, 5)])

        for i in range(1, 5):
            point_name = f'P{i}'
            if point_name not in corners_2d:
                continue

            orig = np.array(corners_2d[point_name])
            direction = orig - np.array([center_x, center_y])
            norm = np.linalg.norm(direction)

            if norm > 0:
                unit = direction / norm
                ext = extensions['top'] if i in [1, 4] else extensions['bottom']
                extended = orig + unit * ext
                enlarged[point_name] = [float(extended[0]), float(extended[1])]
            else:
                enlarged[point_name] = corners_2d[point_name]

        return enlarged
    except Exception as e:
        print(f"Perspective enlargement failed: {e}")
        return None

def enlarge_uniform(corners, factor=0.3):
    if len(corners) != 4:
        return corners

    cx = sum(corners[f"P{i}"][0] for i in range(1, 5)) / 4
    cy = sum(corners[f"P{i}"][1] for i in range(1, 5)) / 4

    enlarged = {}
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name not in corners:
            continue
        point = corners[point_name]
        dx = point[0] - cx
        dy = point[1] - cy
        enlarged[point_name] = [float(cx + dx * (1 + factor)), float(cy + dy * (1 + factor))]

    return enlarged

def point_in_polygon(point, corners, enlarged=None):
    x, y = point
    boundary = enlarged if enlarged is not None else corners

    polygon = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in boundary:
            polygon.append(boundary[point_name])

    if len(polygon) != 4:
        return True

    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def validate_pose(keypoints, enlarged, original):
    if keypoints.xy is None or keypoints.conf is None:
        return False, []

    xy = keypoints.xy[0]
    conf = keypoints.conf[0]
    valid_joints = []

    for joint_idx in ANKLE_KNEE_INDICES:
        if joint_idx < len(conf) and conf[joint_idx].item() > CONFIDENCE_THRESHOLD:
            x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
            if point_in_polygon((x, y), original, enlarged):
                valid_joints.append(joint_idx)

    return len(valid_joints) >= MIN_ANKLE_KNEE_JOINTS, valid_joints

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

class PoseDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self._setup_model()

    def _setup_model(self):
        print(f"Loading YOLOv11x-pose on {self.device}")

        model_paths = ['samples/yolo11x-pose.pt', 'yolo11x-pose.pt', './yolo11x-pose.pt']
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            model_path = 'yolo11x-pose.pt'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import contextlib
            import io
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.model = YOLO(model_path, verbose=False)
                self.model.to(self.device)

    def detect(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model(frame, verbose=False)

    def process_video(self, video_path, calib_data, court_points, output_path):
        corners = extract_corners(court_points)
        if len(corners) < 4:
            raise ValueError(f"Need 4 corners, got {len(corners)}")

        K = calib_data['camera_matrix']
        rvec = calib_data['rvec']
        tvec = calib_data['tvec']

        enlarged_court = None
        method = "uniform_fallback"

        if K is not None and rvec is not None and tvec is not None:
            enlarged_court = create_perspective_boundary(corners, rvec, tvec)
            if enlarged_court is not None:
                method = "perspective_aware_calibrated"

        if enlarged_court is None:
            enlarged_court = enlarge_uniform(corners, factor=0.3)

        video_info = get_video_info(video_path)

        print(f"Processing {video_info['frame_count']} frames with {method}")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        all_poses = []

        with tqdm(total=frame_count, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process all frames
                results = self.detect(frame)

                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints
                    for human_idx, person_kp in enumerate(keypoints):
                        is_valid, valid_joints = validate_pose(person_kp, enlarged_court, corners)

                        if is_valid:
                            xy = person_kp.xy[0]
                            conf = person_kp.conf[0]

                            joints = []
                            for joint_idx in range(xy.shape[0]):
                                x, y = float(xy[joint_idx, 0].item()), float(xy[joint_idx, 1].item())
                                confidence = float(conf[joint_idx].item())

                                in_court = False
                                if confidence > CONFIDENCE_THRESHOLD:
                                    in_court = point_in_polygon((x, y), corners, enlarged_court)

                                joints.append({
                                    "joint_index": joint_idx,
                                    "x": x,
                                    "y": y,
                                    "confidence": confidence,
                                    "in_court": in_court,
                                    "is_ankle_knee": joint_idx in ANKLE_KNEE_INDICES
                                })

                            pose_data = {
                                "frame_index": frame_idx,
                                "human_index": human_idx,
                                "joints": joints,
                                "valid_ankle_knee_joints": len(valid_joints),
                                "validation_joints": valid_joints
                            }
                            all_poses.append(pose_data)

                frame_idx += 1
                pbar.update(1)

        cap.release()

        output_data = {
            "court_points": corners,
            "enlarged_court_points": enlarged_court,
            "all_court_points": court_points,
            "video_info": video_info,
            "pose_data": all_poses,
            "camera_calibration": {
                "camera_matrix": calib_data['camera_matrix'].tolist() if calib_data['camera_matrix'] is not None else None,
                "distortion_coefficients": calib_data['dist_coeffs'].tolist() if calib_data['dist_coeffs'] is not None else None,
                "rotation_vector": calib_data['rvec'].tolist() if calib_data['rvec'] is not None else None,
                "translation_vector": calib_data['tvec'].tolist() if calib_data['tvec'] is not None else None,
                "reprojection_error": calib_data['reprojection_error'],
                "calibration_strategy": calib_data['strategy']
            },
            "processing_info": {
                "total_poses_detected": len(all_poses),
                "total_court_points": len(court_points),
                "corner_points_used": list(corners.keys()),
                "enlargement_method": method,
                "max_jump_height_considered": f"{MAX_JUMP_HEIGHT}m",
                "validation_criteria": f"At least {MIN_ANKLE_KNEE_JOINTS} ankle/knee joints in enlarged boundary",
                "ankle_knee_joint_indices": ANKLE_KNEE_INDICES,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "model_device": self.device,
                "model_type": "YOLOv11x-pose",
                "court_dimensions": f"{COURT_LENGTH}m x {COURT_WIDTH}m"
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Detected {len(all_poses)} valid poses using {method}")
        print(f"Saved to {output_path}")
        return output_path

def main(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)

    calib_path = os.path.join(result_dir, "calibration.csv")
    detect_path = os.path.join(result_dir, "detections.csv")
    output_path = os.path.join(result_dir, "pose.json")

    if not os.path.exists(calib_path):
        print(f"Missing calibration: {calib_path}")
        print("Run court detection first")
        return None

    if not os.path.exists(detect_path):
        print(f"Missing detections: {detect_path}")
        print("Run court detection first")
        return None

    try:
        calib_data = read_calibration_csv(calib_path)
        court_points = read_detections_csv(detect_path)

        if len(court_points) == 0:
            raise ValueError("No court points found")

        detector = PoseDetector()
        result = detector.process_video(video_path, calib_data, court_points, output_path)
        del detector
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_pose.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    result = main(video_path)
    if result:
        print("Done.")
    else:
        print("Failed.")
        sys.exit(1)