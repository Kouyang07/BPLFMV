import argparse
import logging
import os
import subprocess
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple


class StreamlinedBadmintonCalibrator:
    """
    Streamlined badminton court calibration that outputs everything to a single CSV file.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.court_3d_coordinates = self._initialize_court_3d_points()

    def _initialize_court_3d_points(self) -> Dict[str, np.ndarray]:
        """Initialize 3D coordinates for all badminton court points in meters."""
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

        court_points_3d = {
            # Ground points (z=0)
            'P1': np.array([0.0, 0.0, 0.0]),
            'P4': np.array([court_width, 0.0, 0.0]),
            'P5': np.array([singles_margin, 0.0, 0.0]),
            'P8': np.array([court_width - singles_margin, 0.0, 0.0]),
            'P21': np.array([court_width/2.0, 0.0, 0.0]),
            'P2': np.array([0.0, court_length, 0.0]),
            'P3': np.array([court_width, court_length, 0.0]),
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
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        point_name = row['Point'].strip()
                        x_coord = float(row['X'])
                        y_coord = float(row['Y'])
                        court_points[point_name] = [x_coord, y_coord]
                    except (ValueError, KeyError):
                        continue
            else:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            point_name = row[0].strip()
                            x_coord = float(row[1])
                            y_coord = float(row[2])
                            court_points[point_name] = [x_coord, y_coord]
                        except (ValueError, IndexError):
                            continue

        return court_points

    def calibrate_camera(self, detected_points: Dict[str, List[float]],
                         image_size: Tuple[int, int]) -> Dict:
        """Enhanced camera calibration with coordinate system correction."""

        # Separate coplanar and non-coplanar points
        coplanar_points = {}
        noncoplanar_points = {}

        for name, coords in detected_points.items():
            if name in self.court_3d_coordinates:
                if self.court_3d_coordinates[name][2] == 0.0:
                    coplanar_points[name] = coords
                else:
                    noncoplanar_points[name] = coords

        all_points = {**coplanar_points, **noncoplanar_points}

        if len(all_points) < 6:
            if len(all_points) < 4:
                raise ValueError(f"Need at least 4 points, got {len(all_points)}")

        # Prepare point correspondences
        object_points = []
        image_points = []
        point_names = []

        for name, coords in all_points.items():
            object_points.append(self.court_3d_coordinates[name])
            image_points.append(coords)
            point_names.append(name)

        object_points = np.array([object_points], dtype=np.float32)
        image_points = np.array([image_points], dtype=np.float32)

        # Initial camera matrix
        focal_estimate = max(image_size) * 0.9
        camera_matrix_init = np.array([
            [focal_estimate, 0, image_size[0]/2],
            [0, focal_estimate, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Determine if non-coplanar points are present
        has_noncoplanar = len(noncoplanar_points) > 0

        if has_noncoplanar:
            # Estimate with some distortion for non-coplanar cases
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                     cv2.CALIB_RATIONAL_MODEL |
                     cv2.CALIB_FIX_PRINCIPAL_POINT |
                     cv2.CALIB_FIX_ASPECT_RATIO)
            dist_coeffs_init = np.zeros(8, dtype=np.float32)
        else:
            # Restricted model for coplanar points: no distortion, fix principal point and aspect ratio
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                     cv2.CALIB_FIX_PRINCIPAL_POINT |
                     cv2.CALIB_FIX_ASPECT_RATIO |
                     cv2.CALIB_ZERO_TANGENT_DIST |
                     cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                     cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
            dist_coeffs_init = np.zeros(8, dtype=np.float32)

        # Try calibration with coordinate system correction
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size,
                camera_matrix_init, dist_coeffs_init, flags=flags
            )

            camera_height = float(tvecs[0][2][0])

            # If camera height is unreasonable, try Y-axis flip
            if camera_height < 0 or camera_height > 50:
                object_points_y_flip = object_points.copy()
                max_y = np.max(object_points_y_flip[0, :, 1])
                object_points_y_flip[0, :, 1] = max_y - object_points_y_flip[0, :, 1]

                ret_y, camera_matrix_y, dist_coeffs_y, rvecs_y, tvecs_y = cv2.calibrateCamera(
                    object_points_y_flip, image_points, image_size,
                    camera_matrix_init, dist_coeffs_init, flags=flags
                )

                camera_height_y = float(tvecs_y[0][2][0])

                if 0 < camera_height_y < 30:
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = ret_y, camera_matrix_y, dist_coeffs_y, rvecs_y, tvecs_y

        except Exception:
            # Fallback to basic calibration with restricted flags
            flags_fallback = (cv2.CALIB_USE_INTRINSIC_GUESS |
                              cv2.CALIB_FIX_PRINCIPAL_POINT |
                              cv2.CALIB_FIX_ASPECT_RATIO |
                              cv2.CALIB_ZERO_TANGENT_DIST |
                              cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size,
                camera_matrix_init, None, flags=flags_fallback
            )

        if not ret:
            raise ValueError("Camera calibration failed")

        # Calculate reprojection errors
        projected_points, _ = cv2.projectPoints(
            object_points[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        errors = np.sqrt(np.sum((image_points[0] - projected_points)**2, axis=1))
        mean_error = np.mean(errors)

        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvec': rvecs[0].flatten(),
            'tvec': tvecs[0].flatten(),
            'reprojection_error': float(mean_error),
            'point_names': point_names,
            'point_errors': {name: float(error) for name, error in zip(point_names, errors)},
            'detected_points': detected_points,
            'image_size': image_size
        }

    def save_complete_calibration_csv(self, calibration_results: Dict, output_path: str) -> None:
        """Save all calibration data to a single CSV file."""

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header section
            writer.writerow(['# Badminton Court Calibration Results'])
            writer.writerow(['# Generated by Enhanced BWF Court Calibrator'])
            writer.writerow([''])

            # Camera intrinsic parameters
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

            # Extrinsic parameters
            writer.writerow(['# Camera Extrinsic Parameters'])
            writer.writerow(['# Rotation Vector (Rodrigues)'])
            for i, val in enumerate(calibration_results['rvec']):
                writer.writerow([f'rotation_{["x", "y", "z"][i]}', val])
            writer.writerow([''])

            writer.writerow(['# Translation Vector'])
            for i, val in enumerate(calibration_results['tvec']):
                writer.writerow([f'translation_{["x", "y", "z"][i]}', val])
            writer.writerow([''])

            # Calibration quality metrics
            writer.writerow(['# Calibration Quality'])
            writer.writerow(['reprojection_error_pixels', calibration_results['reprojection_error']])
            writer.writerow(['camera_height_meters', calibration_results['tvec'][2]])
            writer.writerow(['image_width', calibration_results['image_size'][0]])
            writer.writerow(['image_height', calibration_results['image_size'][1]])
            writer.writerow(['num_points_used', len(calibration_results['point_names'])])
            writer.writerow([''])

            # Detected court points with errors
            writer.writerow(['# Detected Court Points with Reprojection Errors'])
            writer.writerow(['Point', 'Image_X', 'Image_Y', 'World_X', 'World_Y', 'World_Z', 'Error_Pixels'])

            for point_name in calibration_results['point_names']:
                if point_name in calibration_results['detected_points']:
                    img_coords = calibration_results['detected_points'][point_name]
                    world_coords = self.court_3d_coordinates[point_name]
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
    """Get the frame size of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return (width, height)


def main(video_path, debug=False):
    """Main function for streamlined badminton court camera calibration."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results/", f"{base_name}")
    os.makedirs(result_dir, exist_ok=True)

    # Run court detection
    court_csv_path = os.path.join(result_dir, "court.csv")

    if not run_detect_script(video_path, court_csv_path):
        raise RuntimeError("Court detection failed")

    if not os.path.exists(court_csv_path):
        raise RuntimeError(f"Court detection output not found: {court_csv_path}")

    # Get video frame size
    image_size = get_video_frame_size(video_path)

    # Initialize calibrator
    calibrator = StreamlinedBadmintonCalibrator(debug=debug)

    # Load detected court points
    detected_court_points = calibrator.load_court_points_from_csv(court_csv_path)

    if len(detected_court_points) < 4:
        raise RuntimeError(f"Insufficient court points detected: {len(detected_court_points)}")

    # Perform camera calibration
    calibration_results = calibrator.calibrate_camera(detected_court_points, image_size)

    # Save complete results to single CSV
    output_csv_path = os.path.join(result_dir, f"{base_name}_calibration_complete.csv")
    calibrator.save_complete_calibration_csv(calibration_results, output_csv_path)

    print(f"✅ Complete calibration saved to: {output_csv_path}")
    print(f"📊 Reprojection error: {calibration_results['reprojection_error']:.2f} pixels")
    print(f"📏 Camera height: {calibration_results['tvec'][2]:.1f} meters")
    print(f"🎯 Points used: {len(calibration_results['point_names'])}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined badminton court camera calibration")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    try:
        success = main(args.video_path, debug=args.debug)
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)