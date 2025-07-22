import argparse
import logging
import os
import subprocess
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple


class SimplifiedBadmintonCalibrator:
    """
    Simplified badminton court calibration with fixed coordinate system alignment.
    Uses the same coordinate system that the tracking expects: P1=top-left origin.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.court_3d_coordinates = self._initialize_court_3d_points()

    def _initialize_court_3d_points(self) -> Dict[str, np.ndarray]:
        """Initialize 3D coordinates matching the tracking system's expectations."""
        court_length = 13.40
        court_width = 6.10
        singles_width = 5.18
        service_length = 3.96

        net_position = court_length / 2.0
        service_line_upper = net_position - service_length
        service_line_lower = net_position + service_length
        singles_margin = (court_width - singles_width) / 2.0

        # BWF standard net heights
        net_height_posts = 1.55
        net_height_center = 1.524

        # FIXED COORDINATE SYSTEM - matches tracking expectations
        # P1 = top-left (0,0), P2 = bottom-left (0,13.4), etc.
        court_points_3d = {
            # Main corners - CRITICAL: these must match tracking system
            'P1': np.array([0.0, 0.0, 0.0]),                    # Top-left corner (ORIGIN)
            'P2': np.array([0.0, court_length, 0.0]),           # Bottom-left corner
            'P3': np.array([court_width, court_length, 0.0]),   # Bottom-right corner
            'P4': np.array([court_width, 0.0, 0.0]),            # Top-right corner

            # All other points follow the same coordinate system
            'P5': np.array([singles_margin, 0.0, 0.0]),
            'P8': np.array([court_width - singles_margin, 0.0, 0.0]),
            'P21': np.array([court_width/2.0, 0.0, 0.0]),
            'P6': np.array([singles_margin, court_length, 0.0]),
            'P7': np.array([court_width - singles_margin, court_length, 0.0]),
            'P22': np.array([court_width/2.0, court_length, 0.0]),
            'P9': np.array([0.0, service_line_upper, 0.0]),
            'P10': np.array([court_width, service_line_upper, 0.0]),
            'P11': np.array([0.0, service_line_lower, 0.0]),
            'P12': np.array([court_width, service_line_lower, 0.0]),
            'P13': np.array([court_width/2.0, service_line_upper, 0.0]),
            'P14': np.array([court_width/2.0, service_line_lower, 0.0]),
            'P17': np.array([singles_margin, service_line_upper, 0.0]),
            'P18': np.array([court_width - singles_margin, service_line_upper, 0.0]),
            'P19': np.array([singles_margin, service_line_lower, 0.0]),
            'P20': np.array([court_width - singles_margin, service_line_lower, 0.0]),
            'P15': np.array([0.0, net_position, 0.0]),
            'P16': np.array([court_width, net_position, 0.0]),

            # Net poles (elevated points)
            'NetPole1': np.array([0.0, net_position, net_height_posts]),
            'NetPole2': np.array([court_width, net_position, net_height_posts]),
            'NetCenter': np.array([court_width/2.0, net_position, net_height_center]),
        }

        return court_points_3d

    def load_court_points_from_csv(self, csv_path: str) -> Dict[str, List[float]]:
        """Load court points from CSV file."""
        court_points = {}

        with open(csv_path, 'r') as f:
            f.seek(0)
            sample_line = f.readline().strip()
            f.seek(0)

            if 'Point' in sample_line and 'X' in sample_line and 'Y' in sample_line:
                print("📋 CSV has header row")
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        point_name = row['Point'].strip()
                        x_coord = float(row['X'])
                        y_coord = float(row['Y'])
                        court_points[point_name] = [x_coord, y_coord]
                        if self.debug:
                            print(f"   {point_name}: ({x_coord:.1f}, {y_coord:.1f})")
                    except (ValueError, KeyError) as e:
                        if self.debug:
                            print(f"⚠️  Skipping invalid row: {e}")
                        continue
            else:
                print("📋 CSV without header, using positional parsing")
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            point_name = row[0].strip()
                            x_coord = float(row[1])
                            y_coord = float(row[2])
                            court_points[point_name] = [x_coord, y_coord]
                            if self.debug:
                                print(f"   {point_name}: ({x_coord:.1f}, {y_coord:.1f})")
                        except (ValueError, IndexError) as e:
                            if self.debug:
                                print(f"⚠️  Skipping invalid row: {e}")
                            continue

        print(f"✅ Loaded {len(court_points)} court points")
        return court_points

    def calibrate_camera(self, detected_points: Dict[str, List[float]],
                         image_size: Tuple[int, int]) -> Dict:
        """Simplified camera calibration with fixed coordinate system."""

        print(f"🎯 Starting calibration with {len(detected_points)} points")
        print(f"📐 Image size: {image_size[0]}x{image_size[1]}")

        # Filter points that exist in both detected and 3D coordinate systems
        valid_points = {}
        for name, coords in detected_points.items():
            if name in self.court_3d_coordinates:
                valid_points[name] = coords

        if len(valid_points) < 4:
            raise ValueError(f"Need at least 4 valid points, got {len(valid_points)}")

        print(f"📊 Using {len(valid_points)} valid court points for calibration")

        # Prepare point correspondences
        object_points = []
        image_points = []
        point_names = []

        for name, image_coords in valid_points.items():
            object_points.append(self.court_3d_coordinates[name])
            image_points.append(image_coords)
            point_names.append(name)

        object_points = np.array([object_points], dtype=np.float32)
        image_points = np.array([image_points], dtype=np.float32)

        # Initial camera matrix estimate
        focal_estimate = max(image_size) * 0.9
        camera_matrix_init = np.array([
            [focal_estimate, 0, image_size[0]/2],
            [0, focal_estimate, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)

        print(f"🔍 Initial focal length estimate: {focal_estimate:.0f}")

        # Determine if we have elevated points for distortion modeling
        has_elevated_points = any(self.court_3d_coordinates[name][2] > 0 for name in valid_points.keys())

        if has_elevated_points:
            print("🔧 Using distortion model (elevated points detected)")
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                     cv2.CALIB_RATIONAL_MODEL |
                     cv2.CALIB_FIX_PRINCIPAL_POINT |
                     cv2.CALIB_FIX_ASPECT_RATIO)
            dist_coeffs_init = np.zeros(8, dtype=np.float32)
        else:
            print("🔧 Using no distortion model (ground points only)")
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                     cv2.CALIB_FIX_PRINCIPAL_POINT |
                     cv2.CALIB_FIX_ASPECT_RATIO |
                     cv2.CALIB_ZERO_TANGENT_DIST |
                     cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                     cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
            dist_coeffs_init = np.zeros(8, dtype=np.float32)

        # Try multiple calibration approaches
        calibration_attempts = []

        # Method 1: Standard calibration
        try:
            print("📐 Method 1: Standard calibration...")
            ret1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(
                object_points, image_points, image_size,
                camera_matrix_init, dist_coeffs_init, flags=flags
            )

            if ret1:
                # Calculate reprojection error
                projected_points1, _ = cv2.projectPoints(
                    object_points[0], rvecs1[0], tvecs1[0], camera_matrix1, dist_coeffs1
                )
                projected_points1 = projected_points1.reshape(-1, 2)
                errors1 = np.sqrt(np.sum((image_points[0] - projected_points1)**2, axis=1))
                mean_error1 = np.mean(errors1)
                camera_height1 = float(tvecs1[0][2][0])

                print(f"📏 Standard - Height: {camera_height1:.2f}m, Error: {mean_error1:.2f}px")

                calibration_attempts.append({
                    'method': 'standard',
                    'camera_matrix': camera_matrix1, 'dist_coeffs': dist_coeffs1,
                    'rvecs': rvecs1, 'tvecs': tvecs1, 'error': mean_error1, 'height': camera_height1,
                    'object_points_used': object_points
                })

        except Exception as e:
            print(f"⚠️  Standard calibration failed: {e}")

        # Method 2: Y-flipped coordinates (common issue with image coordinate systems)
        try:
            print("📐 Method 2: Y-axis flipped calibration...")
            object_points_y_flip = object_points.copy()
            max_y = np.max(object_points_y_flip[0, :, 1])
            object_points_y_flip[0, :, 1] = max_y - object_points_y_flip[0, :, 1]

            ret2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(
                object_points_y_flip, image_points, image_size,
                camera_matrix_init, dist_coeffs_init, flags=flags
            )

            if ret2:
                projected_points2, _ = cv2.projectPoints(
                    object_points_y_flip[0], rvecs2[0], tvecs2[0], camera_matrix2, dist_coeffs2
                )
                projected_points2 = projected_points2.reshape(-1, 2)
                errors2 = np.sqrt(np.sum((image_points[0] - projected_points2)**2, axis=1))
                mean_error2 = np.mean(errors2)
                camera_height2 = float(tvecs2[0][2][0])

                print(f"📏 Y-flipped - Height: {camera_height2:.2f}m, Error: {mean_error2:.2f}px")

                calibration_attempts.append({
                    'method': 'y_flipped',
                    'camera_matrix': camera_matrix2, 'dist_coeffs': dist_coeffs2,
                    'rvecs': rvecs2, 'tvecs': tvecs2, 'error': mean_error2, 'height': camera_height2,
                    'object_points_used': object_points_y_flip
                })

        except Exception as e:
            print(f"⚠️  Y-flipped calibration failed: {e}")

        # Method 3: Minimal calibration (fallback)
        if not calibration_attempts:
            try:
                print("📐 Method 3: Minimal calibration (fallback)...")
                flags_minimal = (cv2.CALIB_USE_INTRINSIC_GUESS |
                                 cv2.CALIB_FIX_PRINCIPAL_POINT |
                                 cv2.CALIB_FIX_ASPECT_RATIO |
                                 cv2.CALIB_ZERO_TANGENT_DIST |
                                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

                ret3, camera_matrix3, dist_coeffs3, rvecs3, tvecs3 = cv2.calibrateCamera(
                    object_points, image_points, image_size,
                    camera_matrix_init, np.zeros(5, dtype=np.float32), flags=flags_minimal
                )

                if ret3:
                    projected_points3, _ = cv2.projectPoints(
                        object_points[0], rvecs3[0], tvecs3[0], camera_matrix3, dist_coeffs3
                    )
                    projected_points3 = projected_points3.reshape(-1, 2)
                    errors3 = np.sqrt(np.sum((image_points[0] - projected_points3)**2, axis=1))
                    mean_error3 = np.mean(errors3)
                    camera_height3 = float(tvecs3[0][2][0])

                    print(f"📏 Minimal - Height: {camera_height3:.2f}m, Error: {mean_error3:.2f}px")

                    calibration_attempts.append({
                        'method': 'minimal',
                        'camera_matrix': camera_matrix3, 'dist_coeffs': dist_coeffs3,
                        'rvecs': rvecs3, 'tvecs': tvecs3, 'error': mean_error3, 'height': camera_height3,
                        'object_points_used': object_points
                    })

            except Exception as e:
                print(f"⚠️  Minimal calibration failed: {e}")

        if not calibration_attempts:
            raise ValueError("All calibration methods failed")

        # Select best calibration: prefer reasonable heights, then lowest error
        print(f"\n🔍 Comparing {len(calibration_attempts)} calibration attempts...")

        reasonable_attempts = [attempt for attempt in calibration_attempts
                               if 3 < attempt['height'] < 50]

        if reasonable_attempts:
            best_attempt = min(reasonable_attempts, key=lambda x: x['error'])
            print(f"✅ Selected {best_attempt['method']} method (reasonable height + lowest error)")
        else:
            best_attempt = min(calibration_attempts, key=lambda x: x['error'])
            print(f"⚠️  No reasonable heights found, selected {best_attempt['method']} method (lowest error)")

        # Extract results
        camera_matrix = best_attempt['camera_matrix']
        dist_coeffs = best_attempt['dist_coeffs']
        rvecs = best_attempt['rvecs']
        tvecs = best_attempt['tvecs']
        method_used = best_attempt['method']

        # Calculate final errors
        object_points_final = best_attempt['object_points_used']
        projected_points, _ = cv2.projectPoints(
            object_points_final[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        errors = np.sqrt(np.sum((image_points[0] - projected_points)**2, axis=1))
        mean_error = np.mean(errors)

        print(f"✅ Calibration completed using {method_used} method!")
        print(f"📊 Final reprojection error: {mean_error:.2f} pixels")
        print(f"📏 Final camera height: {tvecs[0][2][0]:.2f} meters")
        print(f"🔍 Final focal length: {camera_matrix[0,0]:.0f} pixels")

        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvec': rvecs[0].flatten(),
            'tvec': tvecs[0].flatten(),
            'reprojection_error': float(mean_error),
            'point_names': point_names,
            'point_errors': {name: float(error) for name, error in zip(point_names, errors)},
            'detected_points': detected_points,
            'image_size': image_size,
            'calibration_method': method_used,
            'y_axis_flipped': method_used == 'y_flipped',  # Track if Y-axis was flipped
            'object_points_used': object_points_final      # Store the actual 3D coordinates used
        }

    def save_complete_calibration_csv(self, calibration_results: Dict, output_path: str) -> None:
        """Save calibration results to CSV file."""

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['# Badminton Court Calibration Results'])
            writer.writerow(['# Generated by Simplified BWF Court Calibrator'])
            method_used = calibration_results.get('calibration_method', 'unknown')
            writer.writerow([f'# Calibration method: {method_used}'])
            writer.writerow([f'# Coordinate system: Fixed P1=top-left origin (matches tracking)'])
            writer.writerow([''])

            # Camera intrinsics
            writer.writerow(['# Camera Intrinsic Matrix (3x3)'])
            writer.writerow(['intrinsic_fx', calibration_results['camera_matrix'][0, 0]])
            writer.writerow(['intrinsic_fy', calibration_results['camera_matrix'][1, 1]])
            writer.writerow(['intrinsic_cx', calibration_results['camera_matrix'][0, 2]])
            writer.writerow(['intrinsic_cy', calibration_results['camera_matrix'][1, 2]])
            writer.writerow([''])

            # Distortion coefficients
            writer.writerow(['# Distortion Coefficients'])
            for i, coeff in enumerate(calibration_results['dist_coeffs'].flatten()):
                writer.writerow([f'distortion_k{i+1}' if i < 3 else f'distortion_p{i-2}', coeff])
            writer.writerow([''])

            # Extrinsics
            writer.writerow(['# Camera Extrinsic Parameters'])
            writer.writerow(['# Rotation Vector (Rodrigues)'])
            for i, val in enumerate(calibration_results['rvec']):
                writer.writerow([f'rotation_{["x", "y", "z"][i]}', val])
            writer.writerow([''])

            writer.writerow(['# Translation Vector'])
            for i, val in enumerate(calibration_results['tvec']):
                writer.writerow([f'translation_{["x", "y", "z"][i]}', val])
            writer.writerow([''])

            # Quality metrics
            writer.writerow(['# Calibration Quality'])
            writer.writerow(['reprojection_error_pixels', calibration_results['reprojection_error']])
            writer.writerow(['camera_height_meters', calibration_results['tvec'][2]])
            writer.writerow(['image_width', calibration_results['image_size'][0]])
            writer.writerow(['image_height', calibration_results['image_size'][1]])
            writer.writerow(['num_points_used', len(calibration_results['point_names'])])
            writer.writerow(['calibration_method', method_used])
            writer.writerow(['coordinate_system', 'fixed_top_left_origin'])
            writer.writerow(['y_axis_flipped', calibration_results.get('y_axis_flipped', False)])
            writer.writerow([''])

            # Court points with errors
            writer.writerow(['# Court Points with Reprojection Errors'])
            writer.writerow(['Point', 'Image_X', 'Image_Y', 'World_X', 'World_Y', 'World_Z', 'Error_Pixels'])

            # Use the correct 3D coordinates - if Y-flipped was used, we need to show the flipped coordinates
            y_flipped = calibration_results.get('y_axis_flipped', False)

            for point_name in calibration_results['point_names']:
                if point_name in calibration_results['detected_points']:
                    img_coords = calibration_results['detected_points'][point_name]

                    # Get the original world coordinates
                    world_coords = self.court_3d_coordinates[point_name].copy()

                    # If Y-axis was flipped during calibration, show the flipped coordinates
                    if y_flipped:
                        max_y = 13.4  # Court length
                        world_coords[1] = max_y - world_coords[1]

                    error = calibration_results['point_errors'][point_name]

                    writer.writerow([
                        point_name,
                        f"{img_coords[0]:.2f}",
                        f"{img_coords[1]:.2f}",
                        f"{world_coords[0]:.3f}",
                        f"{world_coords[1]:.3f}",
                        f"{world_coords[2]:.3f}",
                        f"{error:.3f}"
                    ])


def run_detect_script(video_path, output_path):
    """Run the court detection script."""
    result = subprocess.run(f'./resources/detect {video_path} {output_path}',
                            shell=True, capture_output=True, text=True)
    return result.returncode == 0


def get_video_frame_size(video_path: str) -> Tuple[int, int]:
    """Get video frame size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    print(f"📐 Video specs: {width}x{height}, {frame_count} frames @ {fps:.1f}fps")
    return (width, height)


def main(video_path, debug=False):
    """Main function for simplified calibration."""
    print("🚀 Starting Simplified Badminton Court Calibration")
    print("="*60)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results/", f"{base_name}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"🎬 Processing video: {video_path}")
    print(f"💾 Results directory: {result_dir}")

    # Run court detection
    court_csv_path = os.path.join(result_dir, "court.csv")

    print("\n" + "="*60)
    print("STEP 1: COURT DETECTION")
    print("="*60)

    if not run_detect_script(video_path, court_csv_path):
        raise RuntimeError("Court detection failed")

    if not os.path.exists(court_csv_path):
        raise RuntimeError(f"Court detection output not found: {court_csv_path}")

    print("✅ Court detection completed")

    # Get video info
    print("\n" + "="*60)
    print("STEP 2: VIDEO ANALYSIS")
    print("="*60)

    image_size = get_video_frame_size(video_path)

    # Initialize calibrator
    print("\n" + "="*60)
    print("STEP 3: CALIBRATOR SETUP")
    print("="*60)

    calibrator = SimplifiedBadmintonCalibrator(debug=debug)
    print("✅ Calibrator initialized with fixed coordinate system (P1=top-left origin)")

    # Load points and calibrate
    print("\n" + "="*60)
    print("STEP 4: POINT LOADING")
    print("="*60)

    detected_court_points = calibrator.load_court_points_from_csv(court_csv_path)

    if len(detected_court_points) < 4:
        raise RuntimeError(f"Insufficient court points detected: {len(detected_court_points)}")

    print("\n" + "="*60)
    print("STEP 5: CAMERA CALIBRATION")
    print("="*60)

    calibration_results = calibrator.calibrate_camera(detected_court_points, image_size)

    # Save results
    print("\n" + "="*60)
    print("STEP 6: RESULTS")
    print("="*60)

    output_csv_path = os.path.join(result_dir, f"{base_name}_calibration_complete.csv")
    calibrator.save_complete_calibration_csv(calibration_results, output_csv_path)

    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)

    reprojection_error = calibration_results['reprojection_error']
    camera_height = calibration_results['tvec'][2]
    num_points = len(calibration_results['point_names'])
    focal_length = calibration_results['camera_matrix'][0,0]
    method_used = calibration_results.get('calibration_method', 'unknown')

    print(f"✅ Complete calibration saved to: {output_csv_path}")
    print(f"📊 Reprojection error: {reprojection_error:.2f} pixels")
    print(f"📏 Camera height: {camera_height:.1f} meters")
    print(f"🎯 Points used: {num_points}")
    print(f"🔍 Focal length: {focal_length:.0f} pixels")
    print(f"🔧 Method used: {method_used}")
    print(f"🎯 Coordinate system: Fixed (P1=top-left origin)")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified badminton court camera calibration")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    try:
        success = main(args.video_path, debug=args.debug)
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
