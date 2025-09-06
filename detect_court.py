import argparse
import os
import subprocess
import csv
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# BWF court dimensions
COURT_LENGTH = 13.40
COURT_WIDTH = 6.10
SINGLES_WIDTH = 5.18
SERVICE_LENGTH = 3.96
NET_HEIGHT_POSTS = 1.55
NET_HEIGHT_CENTER = 1.524

def make_court_points():
    net_pos = COURT_LENGTH / 2.0
    service_upper = SERVICE_LENGTH
    service_lower = COURT_LENGTH - SERVICE_LENGTH
    singles_margin = (COURT_WIDTH - SINGLES_WIDTH) / 2.0

    points = {
        'P1': [0.0, 0.0, 0.0], 'P2': [0.0, COURT_LENGTH, 0.0],
        'P3': [COURT_WIDTH, COURT_LENGTH, 0.0], 'P4': [COURT_WIDTH, 0.0, 0.0],
        'P5': [singles_margin, 0.0, 0.0], 'P6': [singles_margin, COURT_LENGTH, 0.0],
        'P7': [COURT_WIDTH - singles_margin, COURT_LENGTH, 0.0], 'P8': [COURT_WIDTH - singles_margin, 0.0, 0.0],
        'P9': [0.0, service_upper, 0.0], 'P10': [COURT_WIDTH, service_upper, 0.0],
        'P11': [0.0, service_lower, 0.0], 'P12': [COURT_WIDTH, service_lower, 0.0],
        'P13': [COURT_WIDTH/2.0, service_upper, 0.0], 'P14': [COURT_WIDTH/2.0, service_lower, 0.0],
        'P15': [0.0, net_pos, 0.0], 'P16': [COURT_WIDTH, net_pos, 0.0],
        'P17': [singles_margin, service_upper, 0.0], 'P18': [COURT_WIDTH - singles_margin, service_upper, 0.0],
        'P19': [singles_margin, service_lower, 0.0], 'P20': [COURT_WIDTH - singles_margin, service_lower, 0.0],
        'P21': [COURT_WIDTH/2.0, 0.0, 0.0], 'P22': [COURT_WIDTH/2.0, COURT_LENGTH, 0.0],
        'NetPole1': [0.0, net_pos, NET_HEIGHT_POSTS], 'NetPole2': [COURT_WIDTH, net_pos, NET_HEIGHT_POSTS],
        'NetCenter': [COURT_WIDTH/2.0, net_pos, NET_HEIGHT_CENTER],
    }
    return {k: np.array(v) for k, v in points.items()}

def load_detections(csv_path):
    points = {}
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)

        if 'Point' in first_line and 'X' in first_line:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    points[row['Point'].strip()] = np.array([float(row['X']), float(row['Y'])])
                except: pass
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        points[row[0].strip()] = np.array([float(row[1]), float(row[2])])
                    except: pass

    print(f"Loaded {len(points)} detections")
    return points

def fix_orientation(points, img_size):
    corners = ['P1', 'P2', 'P3', 'P4']
    if not all(p in points for p in corners): return points

    def check_orientation(pts):
        p1, p2, p3, p4 = pts['P1'], pts['P2'], pts['P3'], pts['P4']
        return (p1[1] < p2[1] and p4[1] < p3[1] and p1[0] < p4[0] and p2[0] < p3[0])

    if check_orientation(points): return points

    # Try y-flip
    flipped = {name: np.array([pt[0], img_size[1] - pt[1]]) for name, pt in points.items()}
    if check_orientation(flipped):
        print("Applied Y-flip")
        return flipped

    # Try left-right swap
    swap_map = {'P1': 'P4', 'P4': 'P1', 'P2': 'P3', 'P3': 'P2', 'P5': 'P8', 'P8': 'P5',
                'P6': 'P7', 'P7': 'P6', 'P9': 'P10', 'P10': 'P9', 'P11': 'P12', 'P12': 'P11'}
    swapped = points.copy()
    for a, b in swap_map.items():
        if a in swapped and b in swapped:
            swapped[a], swapped[b] = swapped[b], swapped[a]
    if check_orientation(swapped):
        print("Applied left-right swap")
        return swapped

    print("Could not fix orientation")
    return points

def get_euler_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)

def validate_result(K, rvec, tvec, error, n_inliers):
    valid = True
    issues = []

    height = abs(tvec[2])
    if tvec[2] > 0:
        valid = False
        issues.append(f"Camera below court (z={tvec[2].item():.2f})")

    roll, pitch, yaw = get_euler_angles(rvec)
    if pitch > 10:
        valid = False
        issues.append(f"Camera looking up (pitch={pitch:.1f})")

    if error > 20:
        valid = False
        issues.append(f"High error ({error:.1f}px)")

    if n_inliers < 6:
        valid = False
        issues.append(f"Too few inliers ({n_inliers})")

    return valid, issues, {'height': height, 'roll': roll, 'pitch': pitch, 'yaw': yaw}

def try_calibration_strategy(obj_pts, img_pts, img_size, strategy):
    focal_guess = max(img_size) * 0.8
    K_init = np.array([[focal_guess, 0, img_size[0]/2], [0, focal_guess, img_size[1]/2], [0, 0, 1]], dtype=np.float32)

    flags_map = {
        'no_dist': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'k1_only': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'radial': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'full': cv2.CALIB_USE_INTRINSIC_GUESS
    }

    try:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            np.array([obj_pts], dtype=np.float32),
            np.array([img_pts], dtype=np.float32),
            img_size, K_init.copy(), np.zeros(5), flags=flags_map[strategy]
        )
        return ret, K, dist, rvecs[0], tvecs[0]
    except:
        return False, None, None, None, None

def ransac_calibration(obj_pts, img_pts, img_size, max_iter=500):
    court_3d = make_court_points()

    best_result = None
    best_score = -1
    results_log = []

    for i in range(max_iter):
        sample_size = min(12, max(8, len(obj_pts) // 2))
        sample_idx = random.sample(range(len(obj_pts)), sample_size)

        sample_obj = [obj_pts[j] for j in sample_idx]
        sample_img = [img_pts[j] for j in sample_idx]

        for strategy in ['no_dist', 'k1_only', 'radial', 'full']:
            ret, K, dist, rvec, tvec = try_calibration_strategy(sample_obj, sample_img, img_size, strategy)
            if not ret: continue

            # Test on all points
            proj_pts, _ = cv2.projectPoints(np.array([obj_pts], dtype=np.float32)[0], rvec, tvec, K, dist)
            proj_pts = proj_pts.reshape(-1, 2)
            errors = np.sqrt(np.sum((np.array(img_pts) - proj_pts)**2, axis=1))

            inliers = errors < 15.0
            n_inliers = np.sum(inliers)
            mean_error = np.mean(errors[inliers]) if n_inliers > 0 else 999

            valid, issues, metrics = validate_result(K, rvec, tvec, mean_error, n_inliers)

            score = n_inliers / max(1, mean_error) * (2 if valid else 0.5)

            result = {
                'K': K, 'dist': dist, 'rvec': rvec, 'tvec': tvec,
                'strategy': strategy, 'error': mean_error, 'inliers': n_inliers,
                'valid': valid, 'issues': issues, 'metrics': metrics, 'score': score
            }
            results_log.append(result)

            if score > best_score:
                best_score = score
                best_result = result

        if i % 100 == 0:
            print(f"Iteration {i}, best: {best_result['inliers']} inliers, {best_result['error']:.2f}px")

    print(f"RANSAC done: {best_result['strategy']}, {best_result['inliers']}/{len(obj_pts)} inliers")
    return best_result, results_log

def analyze_results(results_log):
    print("\n=== CALIBRATION ANALYSIS ===")

    strategies = {}
    for r in results_log:
        s = r['strategy']
        if s not in strategies: strategies[s] = []
        strategies[s].append(r)

    for strategy, results in strategies.items():
        valid_results = [r for r in results if r['valid']]
        print(f"\n{strategy.upper()}:")
        print(f"  Total attempts: {len(results)}")
        print(f"  Valid results: {len(valid_results)}")
        if valid_results:
            errors = [r['error'] for r in valid_results]
            inliers = [r['inliers'] for r in valid_results]
            print(f"  Error range: {min(errors):.1f} - {max(errors):.1f}px")
            print(f"  Inlier range: {min(inliers)} - {max(inliers)}")

def plot_analysis(results_log, output_dir):
    if not results_log: return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    strategies = ['no_dist', 'k1_only', 'radial', 'full']
    colors = ['red', 'blue', 'green', 'orange']

    for i, strategy in enumerate(strategies):
        results = [r for r in results_log if r['strategy'] == strategy and r['valid']]
        if not results: continue

        errors = [r['error'] for r in results]
        inliers = [r['inliers'] for r in results]
        heights = [r['metrics']['height'].item() for r in results]
        pitches = [r['metrics']['pitch'] for r in results]

        ax1.scatter(inliers, errors, c=colors[i], label=strategy, alpha=0.6)
        ax2.hist(heights, bins=20, alpha=0.5, color=colors[i], label=strategy)
        ax3.hist(pitches, bins=20, alpha=0.5, color=colors[i])
        ax4.scatter(heights, errors, c=colors[i], alpha=0.6)

    ax1.set_xlabel('Inliers'), ax1.set_ylabel('Error (px)'), ax1.legend(), ax1.set_title('Quality vs Inliers')
    ax2.set_xlabel('Height (m)'), ax2.set_ylabel('Count'), ax2.legend(), ax2.set_title('Camera Heights')
    ax3.set_xlabel('Pitch (deg)'), ax3.set_ylabel('Count'), ax3.set_title('Camera Pitch Angles')
    ax4.set_xlabel('Height (m)'), ax4.set_ylabel('Error (px)'), ax4.set_title('Height vs Error')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_analysis.png'), dpi=150)
    plt.close()

def save_results(result, output_dir, img_size, point_names):
    os.makedirs(output_dir, exist_ok=True)

    # Main results CSV only
    with open(os.path.join(output_dir, 'calibration.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value'])

        K = result['K']
        w.writerow(['fx', K[0,0]]), w.writerow(['fy', K[1,1]])
        w.writerow(['cx', K[0,2]]), w.writerow(['cy', K[1,2]])

        for i, d in enumerate(result['dist'].flatten()):
            w.writerow([f'dist_{i}', d])

        w.writerow(['rx', result['rvec'][0]]), w.writerow(['ry', result['rvec'][1]]), w.writerow(['rz', result['rvec'][2]])
        w.writerow(['tx', result['tvec'][0]]), w.writerow(['ty', result['tvec'][1]]), w.writerow(['tz', result['tvec'][2]])

        w.writerow(['strategy', result['strategy']])
        w.writerow(['error_px', result['error']])
        w.writerow(['inliers', result['inliers']])
        w.writerow(['valid', result['valid']])
        w.writerow(['height_m', result['metrics']['height']])
        w.writerow(['pitch_deg', result['metrics']['pitch']])
        w.writerow(['roll_deg', result['metrics']['roll']])
        w.writerow(['yaw_deg', result['metrics']['yaw']])

    print(f"Results saved to {output_dir}")

def run_detection(video_path, output_path):
    result = subprocess.run(f'./resources/detect {video_path} {output_path}',
                            shell=True, capture_output=True, text=True)
    return result.returncode == 0

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {w}x{h}, {frames} frames @ {fps:.1f}fps")
    return (w, h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Generate analysis plots')
    parser.add_argument('--iterations', type=int, default=500, help='RANSAC iterations')
    args = parser.parse_args()

    print("Badminton Camera Calibration Script")
    print("=" * 50)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join("results", video_name)
    detections_csv = os.path.join(output_dir, "detections.csv")

    print(f"Video: {args.video}")
    print(f"Output: {output_dir}")

    # Detection
    print("\nRunning court detection...")
    os.makedirs(output_dir, exist_ok=True)
    if not run_detection(args.video, detections_csv):
        raise RuntimeError("Detection failed")

    # Load data
    img_size = get_video_info(args.video)
    detections = load_detections(detections_csv)
    detections = fix_orientation(detections, img_size)

    # Prepare calibration data
    court_3d = make_court_points()
    obj_pts, img_pts, point_names = [], [], []
    for name in court_3d:
        if name in detections:
            obj_pts.append(court_3d[name])
            img_pts.append(detections[name])
            point_names.append(name)

    print(f"Using {len(obj_pts)} point correspondences")
    if len(obj_pts) < 8:
        raise ValueError("Need at least 8 points")

    # Calibration
    print(f"\nRunning RANSAC calibration ({args.iterations} iterations)...")
    best_result, all_results = ransac_calibration(obj_pts, img_pts, img_size, args.iterations)

    # Analysis
    analyze_results(all_results)

    if args.plot:
        print("Generating analysis plots...")
        plot_analysis(all_results, output_dir)

    # Results
    print(f"\n=== BEST RESULT ===")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Reprojection error: {best_result['error']:.2f}px")
    print(f"Inliers: {best_result['inliers']}/{len(obj_pts)}")
    print(f"Camera height: {best_result['metrics']['height'].item():.1f}m")
    print(f"Camera angles: roll={best_result['metrics']['roll']:.1f}°, pitch={best_result['metrics']['pitch']:.1f}°, yaw={best_result['metrics']['yaw']:.1f}°")
    print(f"Valid: {best_result['valid']}")
    if best_result['issues']:
        print(f"Issues: {', '.join(best_result['issues'])}")

    save_results(best_result, output_dir, img_size, point_names)

if __name__ == "__main__":
    main()