import argparse
import os
import subprocess
import csv
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import random


class CoordinateSystemFixer:
    """
    Automatically detects and fixes coordinate system mismatches between
    C++ detection and Python calibration while preserving output format.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def detect_coordinate_system_issues(self, points: Dict[str, np.ndarray],
                                        image_size: Tuple[int, int]) -> Dict[str, bool]:
        """
        Detect coordinate system mismatches by analyzing point relationships.
        """
        issues = {
            'y_axis_flipped': False,
            'point_labeling_swapped': False,
            'coordinate_system_rotated': False,
            'missing_netcenter': False
        }

        if self.debug:
            print("🔍 Analyzing coordinate system...")

        # Check if we have the basic corner points
        corners = ['P1', 'P2', 'P3', 'P4']
        if not all(p in points for p in corners):
            if self.debug:
                print("⚠️  Missing corner points for coordinate system analysis")
            return issues

        # Expected relationships in correct coordinate system:
        # P1 (top-left): smallest Y
        # P2 (bottom-left): largest Y, smallest X
        # P3 (bottom-right): largest Y, largest X
        # P4 (top-right): smallest Y, largest X

        p1, p2, p3, p4 = points['P1'], points['P2'], points['P3'], points['P4']

        # Check Y-axis orientation
        # In correct system: P1.y < P2.y and P4.y < P3.y
        if p1[1] > p2[1] or p4[1] > p3[1]:
            issues['y_axis_flipped'] = True
            if self.debug:
                print("⚠️  Y-axis appears flipped (detection uses bottom-up coordinates)")

        # Check point labeling
        # P1 should be left of P4, P2 should be left of P3
        if p1[0] > p4[0] or p2[0] > p3[0]:
            issues['point_labeling_swapped'] = True
            if self.debug:
                print("⚠️  Point labeling appears swapped")

        # Check for missing NetCenter
        if 'NetPole1' in points and 'NetPole2' in points and 'NetCenter' not in points:
            issues['missing_netcenter'] = True
            if self.debug:
                print("⚠️  NetCenter point missing")

        if self.debug:
            print(f"   Y-axis flipped: {issues['y_axis_flipped']}")
            print(f"   Point labeling swapped: {issues['point_labeling_swapped']}")
            print(f"   Missing NetCenter: {issues['missing_netcenter']}")

        return issues

    def fix_coordinate_system(self, points: Dict[str, np.ndarray],
                              image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Automatically fix detected coordinate system issues.
        """
        issues = self.detect_coordinate_system_issues(points, image_size)
        fixed_points = points.copy()

        # Fix Y-axis flip (C++ uses bottom-up, Python expects top-down)
        if issues['y_axis_flipped']:
            if self.debug:
                print("🔧 Fixing Y-axis flip...")
            for name, point in fixed_points.items():
                fixed_points[name] = np.array([point[0], image_size[1] - point[1]])

        # Fix point labeling swap
        if issues['point_labeling_swapped']:
            if self.debug:
                print("🔧 Fixing point labeling swap...")
            # Swap left-right pairs
            swap_pairs = [
                ('P1', 'P4'), ('P2', 'P3'),
                ('P5', 'P8'), ('P6', 'P7'),
                ('P9', 'P10'), ('P11', 'P12'),
                ('P15', 'P16'), ('P17', 'P18'),
                ('P19', 'P20'), ('NetPole1', 'NetPole2')
            ]

            for pair1, pair2 in swap_pairs:
                if pair1 in fixed_points and pair2 in fixed_points:
                    fixed_points[pair1], fixed_points[pair2] = fixed_points[pair2], fixed_points[pair1]

        # Add missing NetCenter
        if issues['missing_netcenter']:
            if self.debug:
                print("🔧 Synthesizing missing NetCenter...")
            if 'NetPole1' in fixed_points and 'NetPole2' in fixed_points:
                fixed_points['NetCenter'] = (fixed_points['NetPole1'] + fixed_points['NetPole2']) / 2.0

        # Verify the fix worked
        if self.debug:
            print("✅ Coordinate system corrections applied")
            self._validate_fixed_coordinates(fixed_points)

        return fixed_points

    def _validate_fixed_coordinates(self, points: Dict[str, np.ndarray]):
        """Validate that coordinate fixes worked correctly."""
        corners = ['P1', 'P2', 'P3', 'P4']
        if all(p in points for p in corners):
            p1, p2, p3, p4 = points['P1'], points['P2'], points['P3'], points['P4']

            # Check expected relationships
            y_correct = p1[1] < p2[1] and p4[1] < p3[1]  # Top points above bottom points
            x_correct = p1[0] < p4[0] and p2[0] < p3[0]  # Left points left of right points

            if y_correct and x_correct:
                print("✅ Coordinate system validation passed")
            else:
                print("⚠️  Coordinate system may still have issues after correction")


class EnhancedBadmintonCalibratorWithFix:
    """
    Enhanced badminton court calibrator with automatic coordinate system fixing.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.court_3d_coordinates = self._initialize_court_3d()
        self.coord_fixer = CoordinateSystemFixer(debug=debug)

    def _initialize_court_3d(self) -> Dict[str, np.ndarray]:
        """Initialize 3D coordinates - UNCHANGED final output coordinate system."""

        # BWF standard court dimensions
        court_length = 13.40  # meters
        court_width = 6.10    # meters
        singles_width = 5.18  # meters
        service_length = 3.96 # meters

        net_position = court_length / 2.0  # 6.70m from baseline
        service_upper = service_length     # 3.96m from P1
        service_lower = court_length - service_length  # 9.44m from P1
        singles_margin = (court_width - singles_width) / 2.0  # 0.46m

        # Net heights (above court surface)
        net_height_posts = 1.55   # meters
        net_height_center = 1.524 # meters

        return {
            # Main court corners (on court surface) - KEEP SAME OUTPUT COORDINATE SYSTEM
            'P1': np.array([0.0, 0.0, 0.0]),                    # Top-left origin
            'P2': np.array([0.0, court_length, 0.0]),           # Bottom-left
            'P3': np.array([court_width, court_length, 0.0]),   # Bottom-right
            'P4': np.array([court_width, 0.0, 0.0]),            # Top-right

            # Singles sidelines
            'P5': np.array([singles_margin, 0.0, 0.0]),                      # Top singles left
            'P6': np.array([singles_margin, court_length, 0.0]),             # Bottom singles left
            'P7': np.array([court_width - singles_margin, court_length, 0.0]), # Bottom singles right
            'P8': np.array([court_width - singles_margin, 0.0, 0.0]),        # Top singles right

            # Service line intersections
            'P9': np.array([0.0, service_upper, 0.0]),           # Left sideline + upper service
            'P10': np.array([court_width, service_upper, 0.0]),  # Right sideline + upper service
            'P11': np.array([0.0, service_lower, 0.0]),          # Left sideline + lower service
            'P12': np.array([court_width, service_lower, 0.0]),  # Right sideline + lower service

            # Center service line
            'P13': np.array([court_width/2.0, service_upper, 0.0]),  # Center + upper service
            'P14': np.array([court_width/2.0, service_lower, 0.0]),  # Center + lower service
            'P21': np.array([court_width/2.0, 0.0, 0.0]),            # Center + top baseline
            'P22': np.array([court_width/2.0, court_length, 0.0]),   # Center + bottom baseline

            # Net line (at court surface)
            'P15': np.array([0.0, net_position, 0.0]),           # Left sideline + net
            'P16': np.array([court_width, net_position, 0.0]),   # Right sideline + net

            # Singles service intersections
            'P17': np.array([singles_margin, service_upper, 0.0]),                    # Left singles + upper
            'P18': np.array([court_width - singles_margin, service_upper, 0.0]),      # Right singles + upper
            'P19': np.array([singles_margin, service_lower, 0.0]),                    # Left singles + lower
            'P20': np.array([court_width - singles_margin, service_lower, 0.0]),      # Right singles + lower

            # Net poles (elevated above court surface)
            'NetPole1': np.array([0.0, net_position, net_height_posts]),
            'NetPole2': np.array([court_width, net_position, net_height_posts]),
            'NetCenter': np.array([court_width/2.0, net_position, net_height_center]),
        }

    def load_detected_points(self, csv_path: str, image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Load court points from detection CSV with automatic coordinate system fixing."""
        points = {}

        with open(csv_path, 'r') as f:
            # Check if header exists
            first_line = f.readline().strip()
            f.seek(0)

            if 'Point' in first_line and 'X' in first_line:
                if self.debug:
                    print("📋 CSV has header row")
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        name = row['Point'].strip()
                        x = float(row['X'])
                        y = float(row['Y'])
                        points[name] = np.array([x, y])
                        if self.debug:
                            print(f"   Raw {name}: ({x:.1f}, {y:.1f})")
                    except (ValueError, KeyError) as e:
                        if self.debug:
                            print(f"⚠️  Skipping invalid row: {e}")
            else:
                if self.debug:
                    print("📋 CSV without header")
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            name = row[0].strip()
                            x = float(row[1])
                            y = float(row[2])
                            points[name] = np.array([x, y])
                            if self.debug:
                                print(f"   Raw {name}: ({x:.1f}, {y:.1f})")
                        except (ValueError, IndexError):
                            continue

        print(f"✅ Loaded {len(points)} raw points from detection")

        # AUTOMATIC COORDINATE SYSTEM FIXING
        print(f"🔧 Applying automatic coordinate system corrections...")
        fixed_points = self.coord_fixer.fix_coordinate_system(points, image_size)

        # Show coordinate corrections applied
        corrections_applied = []
        for name in points.keys():
            if name in fixed_points:
                raw_point = points[name]
                fixed_point = fixed_points[name]
                if not np.allclose(raw_point, fixed_point, atol=1e-6):
                    corrections_applied.append(name)

        if corrections_applied:
            print(f"🔄 Applied corrections to {len(corrections_applied)} points")
            if self.debug:
                print(f"   Corrected points: {corrections_applied}")
        else:
            print("✅ No coordinate system corrections needed")

        # Validate point detection quality
        self._validate_point_detection(fixed_points)

        return fixed_points

    def _validate_point_detection(self, points: Dict[str, np.ndarray]) -> None:
        """Validate detected points for geometric plausibility."""
        if len(points) < 10:
            print(f"⚠️  Warning: Only {len(points)} points detected - may be insufficient")
            return

        # Check for basic geometric relationships
        validation_errors = []

        # Check if corners form a reasonable quadrilateral
        corners = ['P1', 'P2', 'P3', 'P4']
        corner_points = [points[p] for p in corners if p in points]

        if len(corner_points) == 4:
            # Check if quadrilateral is convex and reasonable
            area = self._compute_quadrilateral_area(corner_points)
            if area <= 0:
                validation_errors.append("Court corners don't form a convex quadrilateral")

        # Check parallel lines should have similar slopes in image
        parallel_checks = [
            (['P1', 'P4'], ['P2', 'P3']),  # Top and bottom baselines
            (['P1', 'P2'], ['P4', 'P3']),  # Left and right sidelines
            (['P5', 'P8'], ['P6', 'P7']),  # Singles lines
        ]

        for line1_points, line2_points in parallel_checks:
            if all(p in points for p in line1_points + line2_points):
                slope1 = self._compute_line_slope(points[line1_points[0]], points[line1_points[1]])
                slope2 = self._compute_line_slope(points[line2_points[0]], points[line2_points[1]])
                if abs(slope1 - slope2) > 0.5:  # Slopes should be similar for parallel lines
                    validation_errors.append(f"Lines {line1_points} and {line2_points} don't appear parallel")

        if validation_errors:
            print("⚠️  Point detection validation warnings:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("✅ Point detection passed basic geometric validation")

    def _compute_quadrilateral_area(self, points: List[np.ndarray]) -> float:
        """Compute signed area of quadrilateral using shoelace formula."""
        if len(points) != 4:
            return 0
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, 3)))

    def _compute_line_slope(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute slope of line between two points."""
        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            return float('inf')
        return (p2[1] - p1[1]) / dx

    def get_mirrored_points(self, detected_points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create a mirrored version of points by swapping left-right pairs."""
        mirror_map = {
            'P1': 'P4', 'P4': 'P1',
            'P2': 'P3', 'P3': 'P2',
            'P5': 'P8', 'P8': 'P5',
            'P6': 'P7', 'P7': 'P6',
            'P9': 'P10', 'P10': 'P9',
            'P11': 'P12', 'P12': 'P11',
            'P15': 'P16', 'P16': 'P15',
            'P17': 'P18', 'P18': 'P17',
            'P19': 'P20', 'P20': 'P19',
            'NetPole1': 'NetPole2', 'NetPole2': 'NetPole1',
        }

        mirrored = {}
        for label, mirror_label in mirror_map.items():
            if mirror_label in detected_points:
                mirrored[label] = detected_points[mirror_label]

        # Centers stay the same
        centers = ['P13', 'P14', 'P21', 'P22', 'NetCenter']
        for center in centers:
            if center in detected_points:
                mirrored[center] = detected_points[center]

        if self.debug:
            print("🔄 Created mirrored points")

        return mirrored

    def _progressive_camera_calibration(self, object_points: np.ndarray, image_points: np.ndarray,
                                        image_size: Tuple[int, int], camera_matrix_init: np.ndarray) -> Dict:
        """Progressive calibration: start flexible, then add constraints if needed."""
        dist_coeffs_init = np.zeros(5, dtype=np.float32)
        best_result = None
        best_error = float('inf')

        # Define calibration strategies from flexible to restrictive
        strategies = [
            {
                'name': 'Flexible',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS,
                'description': 'Full parameter estimation'
            },
            {
                'name': 'Fix Principal Point',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT,
                'description': 'Principal point at image center'
            },
            {
                'name': 'Fix K3',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3,
                'description': 'No third-order radial distortion'
            },
            {
                'name': 'Fix Tangential',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST,
                'description': 'No tangential distortion'
            }
        ]

        if self.debug:
            print("🔄 Testing calibration strategies:")

        for strategy in strategies:
            try:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    object_points, image_points, image_size,
                    camera_matrix_init.copy(), dist_coeffs_init.copy(),
                    flags=strategy['flags']
                )

                if not ret:
                    continue

                # Calculate reprojection error
                projected_points, _ = cv2.projectPoints(
                    object_points[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
                errors = np.sqrt(np.sum((image_points[0] - projected_points)**2, axis=1))
                mean_error = np.mean(errors)

                if self.debug:
                    print(f"   {strategy['name']}: {mean_error:.2f}px - {strategy['description']}")

                # Keep track of best result
                if mean_error < best_error:
                    best_error = mean_error
                    best_result = {
                        'camera_matrix': camera_matrix,
                        'dist_coeffs': dist_coeffs,
                        'rvecs': rvecs,
                        'tvecs': tvecs,
                        'mean_error': mean_error,
                        'errors': errors,
                        'strategy': strategy['name']
                    }

                # If we get a very good result, don't need to try more restrictive models
                if mean_error < 2.0:
                    if self.debug:
                        print(f"   Excellent result achieved, stopping at {strategy['name']}")
                    break

            except Exception as e:
                if self.debug:
                    print(f"   {strategy['name']}: Failed - {e}")
                continue

        if best_result is None:
            raise RuntimeError("All calibration strategies failed")

        if self.debug:
            print(f"✅ Selected strategy: {best_result['strategy']} ({best_result['mean_error']:.2f}px)")

        return best_result

    def _ransac_calibration(self, points_2d: Dict[str, np.ndarray], image_size: Tuple[int, int],
                            max_iterations: int = 1000, threshold: float = 15.0) -> Dict:
        """RANSAC-based robust calibration to handle outliers."""
        # Get all available point pairs
        all_point_names = []
        all_object_points = []
        all_image_points = []

        # Use comprehensive point set
        all_available_points = [
            'P1', 'P2', 'P3', 'P4',           # Corners (4)
            'P5', 'P6', 'P7', 'P8',           # Singles lines (4)
            'P9', 'P10', 'P11', 'P12',        # Service intersections (4)
            'P13', 'P14',                     # Center service (2)
            'P15', 'P16',                     # Net line (2)
            'P17', 'P18', 'P19', 'P20',       # Singles-service intersections (4)
            'P21', 'P22',                     # Center baselines (2)
            'NetPole1', 'NetPole2', 'NetCenter'  # Net points (3)
        ]

        for point_name in all_available_points:
            if point_name in points_2d and point_name in self.court_3d_coordinates:
                all_point_names.append(point_name)
                all_object_points.append(self.court_3d_coordinates[point_name])
                all_image_points.append(points_2d[point_name])

        if len(all_object_points) < 8:
            raise ValueError(f"Need at least 8 points for RANSAC. Have {len(all_object_points)}")

        print(f"🎯 RANSAC calibration with {len(all_object_points)} total points")
        if self.debug:
            print(f"   Available points: {all_point_names}")

        # Initial camera matrix estimate
        focal_estimate = max(image_size) * 0.8
        camera_matrix_init = np.array([
            [focal_estimate, 0, image_size[0]/2],
            [0, focal_estimate, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)

        best_inliers = []
        best_error = float('inf')
        best_calibration = None

        for iteration in range(max_iterations):
            # Randomly sample minimum points needed (8 for calibration)
            sample_size = max(8, min(12, len(all_object_points) // 2))
            sample_indices = random.sample(range(len(all_object_points)), sample_size)

            sample_object_points = [all_object_points[i] for i in sample_indices]
            sample_image_points = [all_image_points[i] for i in sample_indices]
            sample_names = [all_point_names[i] for i in sample_indices]

            try:
                # Prepare arrays for OpenCV
                object_points = np.array([sample_object_points], dtype=np.float32)
                image_points = np.array([sample_image_points], dtype=np.float32)

                # Calibrate with progressive strategy
                result = self._progressive_camera_calibration(
                    object_points, image_points, image_size, camera_matrix_init
                )

                camera_matrix = result['camera_matrix']
                dist_coeffs = result['dist_coeffs']
                rvecs = result['rvecs']
                tvecs = result['tvecs']

                # Test all points to find inliers
                all_object_array = np.array([all_object_points], dtype=np.float32)
                projected_points, _ = cv2.projectPoints(
                    all_object_array[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)

                # Compute errors for all points
                all_image_array = np.array(all_image_points)
                errors = np.sqrt(np.sum((all_image_array - projected_points)**2, axis=1))

                # Find inliers
                inlier_mask = errors < threshold
                current_inliers = [all_point_names[i] for i in range(len(errors)) if inlier_mask[i]]
                inlier_error = np.mean(errors[inlier_mask]) if np.any(inlier_mask) else float('inf')

                # Update best if this is better
                if len(current_inliers) > len(best_inliers) or \
                        (len(current_inliers) == len(best_inliers) and inlier_error < best_error):
                    best_inliers = current_inliers
                    best_error = inlier_error
                    best_calibration = result.copy()
                    best_calibration['inlier_points'] = current_inliers
                    best_calibration['all_errors'] = errors
                    best_calibration['inlier_mask'] = inlier_mask

                # Early termination if we have enough good inliers
                if len(current_inliers) >= min(20, int(0.8 * len(all_object_points))) and inlier_error < 5.0:
                    if self.debug:
                        print(f"   Early termination at iteration {iteration + 1}")
                    break

            except Exception as e:
                if self.debug and iteration < 5:  # Only show first few errors
                    print(f"   Iteration {iteration + 1} failed: {e}")
                continue

            # Progress reporting
            if self.debug and (iteration + 1) % 200 == 0:
                print(f"   Iteration {iteration + 1}: Best has {len(best_inliers)} inliers ({best_error:.2f}px)")

        if best_calibration is None:
            raise RuntimeError("RANSAC calibration failed - no valid solution found")

        # Final refinement with all inliers
        if len(best_inliers) >= 8:
            try:
                inlier_object_points = []
                inlier_image_points = []

                for i, name in enumerate(all_point_names):
                    if name in best_inliers:
                        inlier_object_points.append(all_object_points[i])
                        inlier_image_points.append(all_image_points[i])

                object_points_final = np.array([inlier_object_points], dtype=np.float32)
                image_points_final = np.array([inlier_image_points], dtype=np.float32)

                refined_result = self._progressive_camera_calibration(
                    object_points_final, image_points_final, image_size, camera_matrix_init
                )

                # Update best calibration with refined result
                best_calibration.update(refined_result)

                if self.debug:
                    print(f"✅ Refined calibration with {len(best_inliers)} inliers")

            except Exception as e:
                if self.debug:
                    print(f"⚠️  Refinement failed, using RANSAC result: {e}")

        print(f"🎯 RANSAC result: {len(best_inliers)} inliers, {best_error:.2f}px error")
        print(f"   Inlier points: {best_inliers}")

        outliers = [name for name in all_point_names if name not in best_inliers]
        if outliers:
            print(f"   Outlier points: {outliers}")

        return best_calibration

    def calibrate_camera(self, points_2d: Dict[str, np.ndarray], image_size: Tuple[int, int]) -> Dict:
        """Enhanced camera calibration with RANSAC and validation."""
        print(f"🎯 Starting enhanced camera calibration")
        print(f"📐 Image size: {image_size[0]}x{image_size[1]}")

        # Use RANSAC for robust calibration
        calibration_result = self._ransac_calibration(points_2d, image_size)

        camera_matrix = calibration_result['camera_matrix']
        dist_coeffs = calibration_result['dist_coeffs']
        rvecs = calibration_result['rvecs']
        tvecs = calibration_result['tvecs']
        inlier_points = calibration_result.get('inlier_points', [])

        # Extract parameters
        focal_length = float(camera_matrix[0, 0])
        camera_height = float(tvecs[0][2])

        # Calculate final errors
        final_errors = calibration_result.get('errors', [])
        mean_error = calibration_result.get('mean_error', 0)
        max_error = np.max(final_errors) if len(final_errors) > 0 else 0

        # Validation on non-inlier points
        all_available = set(self.court_3d_coordinates.keys()) & set(points_2d.keys())
        validation_points = list(all_available - set(inlier_points))
        validation_errors = []

        for point_name in validation_points:
            if point_name in points_2d and point_name in self.court_3d_coordinates:
                world_point = self.court_3d_coordinates[point_name].reshape(1, 1, 3)
                projected, _ = cv2.projectPoints(
                    world_point, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
                )
                projected = projected.reshape(2)
                actual = points_2d[point_name]
                error = np.linalg.norm(projected - actual)
                validation_errors.append(error)

                if self.debug:
                    print(f"   Validation {point_name}: {error:.2f}px")

        validation_mean = np.mean(validation_errors) if len(validation_errors) > 0 else 0
        validation_median = np.median(validation_errors) if len(validation_errors) > 0 else 0

        print(f"📊 Mean reprojection error: {mean_error:.2f}px")
        print(f"📊 Max reprojection error: {max_error:.2f}px")
        print(f"📏 Camera height: {abs(camera_height):.1f}m")
        print(f"🔍 Focal length: {focal_length:.0f}px")
        print(f"🎯 Inlier points: {len(inlier_points)}")
        print(f"📊 Validation error: {validation_mean:.2f}px mean / {validation_median:.2f}px median ({len(validation_errors)} points)")

        # Enhanced warnings with geometric validation
        warnings = []
        if abs(camera_height) < 1 or abs(camera_height) > 50:
            warnings.append(f"Unusual camera height {abs(camera_height):.1f}m")
        if focal_length < 300 or focal_length > 8000:
            warnings.append(f"Unusual focal length {focal_length:.0f}px")
        if mean_error > 10:
            warnings.append(f"High reprojection error {mean_error:.2f}px")
        if len(inlier_points) < 10:
            warnings.append(f"Few inlier points ({len(inlier_points)})")
        if validation_mean > 50:
            warnings.append(f"High validation error {validation_mean:.2f}px")

        if warnings:
            print("⚠️  Calibration warnings:")
            for warning in warnings:
                print(f"   - {warning}")

        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvec': rvecs[0].flatten(),
            'tvec': tvecs[0].flatten(),
            'reprojection_error': float(mean_error),
            'max_reprojection_error': float(max_error),
            'validation_error_mean': float(validation_mean),
            'validation_error_median': float(validation_median),
            'camera_height': float(camera_height),
            'focal_length': focal_length,
            'point_names': inlier_points,
            'point_errors': {name: float(error) for name, error in zip(inlier_points, final_errors[:len(inlier_points)])},
            'image_size': image_size,
            'num_validation_points': len(validation_errors),
            'calibration_strategy': calibration_result.get('strategy', 'RANSAC'),
            'num_inliers': len(inlier_points),
            'total_points_available': len(all_available)
        }

    def _compute_calibration_score(self, results: Dict) -> float:
        """Compute overall calibration quality score (0-1, higher is better)."""
        # Normalize metrics to 0-1 scale (higher is better)
        error_score = max(0, 1 - results['reprojection_error'] / 20.0)  # 20px = 0 score
        validation_score = max(0, 1 - results['validation_error_median'] / 50.0)  # 50px = 0 score
        inlier_ratio = results.get('num_inliers', 0) / max(1, results.get('total_points_available', 1))

        # Camera parameters reasonableness
        height_reasonable = 1.0 if 2.0 <= abs(results['camera_height']) <= 20.0 else 0.5
        focal_reasonable = 1.0 if 500 <= results['focal_length'] <= 5000 else 0.5

        # Weighted combination
        score = (0.3 * error_score +
                 0.3 * validation_score +
                 0.2 * inlier_ratio +
                 0.1 * height_reasonable +
                 0.1 * focal_reasonable)

        return score

    def _perform_geometric_validation(self, results: Dict, points_2d: Dict[str, np.ndarray]) -> Dict:
        """Perform geometric validation of calibration results."""
        validation_results = {'warnings': [], 'scores': {}, 'overall_score': 0.0}

        # Check if parallel lines in 3D project to approximately parallel lines in 2D
        camera_matrix = results['camera_matrix']
        dist_coeffs = results['dist_coeffs']
        rvec = results['rvec'].reshape(3, 1)
        tvec = results['tvec'].reshape(3, 1)

        # Define parallel line pairs in 3D
        parallel_line_pairs = [
            (['P1', 'P4'], ['P2', 'P3']),  # Top and bottom baselines
            (['P1', 'P2'], ['P4', 'P3']),  # Left and right sidelines
            (['P5', 'P8'], ['P6', 'P7']),  # Singles lines
        ]

        parallel_scores = []

        for line1_points, line2_points in parallel_line_pairs:
            # Check if all points are available
            if all(p in points_2d for p in line1_points + line2_points):
                try:
                    # Project 3D lines to 2D
                    line1_3d = [self.court_3d_coordinates[p] for p in line1_points]
                    line2_3d = [self.court_3d_coordinates[p] for p in line2_points]

                    line1_2d_proj = []
                    line2_2d_proj = []

                    for p3d in line1_3d:
                        p2d, _ = cv2.projectPoints(p3d.reshape(1,1,3), rvec, tvec, camera_matrix, dist_coeffs)
                        line1_2d_proj.append(p2d.reshape(2))

                    for p3d in line2_3d:
                        p2d, _ = cv2.projectPoints(p3d.reshape(1,1,3), rvec, tvec, camera_matrix, dist_coeffs)
                        line2_2d_proj.append(p2d.reshape(2))

                    # Compute slopes
                    slope1 = self._compute_line_slope(line1_2d_proj[0], line1_2d_proj[1])
                    slope2 = self._compute_line_slope(line2_2d_proj[0], line2_2d_proj[1])

                    # Check parallelism (allowing for perspective effects)
                    if abs(slope1) != float('inf') and abs(slope2) != float('inf'):
                        slope_diff = abs(slope1 - slope2)
                        parallel_score = max(0, 1 - slope_diff / 0.5)  # 0.5 slope difference = 0 score
                        parallel_scores.append(parallel_score)

                        if slope_diff > 0.3:
                            validation_results['warnings'].append(
                                f"Lines {line1_points}-{line2_points} not parallel in projection (slopes: {slope1:.2f}, {slope2:.2f})"
                            )

                except Exception as e:
                    validation_results['warnings'].append(f"Could not validate parallelism for {line1_points}-{line2_points}: {e}")

        # Compute overall geometric score
        if parallel_scores:
            validation_results['scores']['parallel_lines'] = np.mean(parallel_scores)
            validation_results['overall_score'] = np.mean(parallel_scores)

        return validation_results

    def save_calibration_results(self, results: Dict, output_path: str, is_mirrored: bool = False) -> None:
        """Save enhanced calibration results to CSV - UNCHANGED output format."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['# Enhanced Badminton Court Calibration Results with Auto-Fix'])
            writer.writerow(['# Coordinate System: P1 = top-left origin (0,0,0) - FINAL OUTPUT'])
            writer.writerow([f'# Mirrored points used: {is_mirrored}'])
            writer.writerow([f'# Calibration strategy: {results.get("calibration_strategy", "Unknown")}'])
            writer.writerow([''])

            # Camera intrinsics - UNCHANGED output format
            writer.writerow(['# Camera Matrix'])
            writer.writerow(['fx', results['camera_matrix'][0, 0]])
            writer.writerow(['fy', results['camera_matrix'][1, 1]])
            writer.writerow(['cx', results['camera_matrix'][0, 2]])
            writer.writerow(['cy', results['camera_matrix'][1, 2]])
            writer.writerow([''])

            # Distortion
            writer.writerow(['# Distortion Coefficients'])
            for i, coeff in enumerate(results['dist_coeffs'].flatten()):
                writer.writerow([f'k{i+1}' if i < 3 else f'p{i-2}' if i < 5 else f'k{i-2}', coeff])
            writer.writerow([''])

            # Extrinsics - UNCHANGED output format
            writer.writerow(['# Camera Pose'])
            writer.writerow(['rx', results['rvec'][0]])
            writer.writerow(['ry', results['rvec'][1]])
            writer.writerow(['rz', results['rvec'][2]])
            writer.writerow(['tx', results['tvec'][0]])
            writer.writerow(['ty', results['tvec'][1]])
            writer.writerow(['tz', results['tvec'][2]])
            writer.writerow([''])

            # Enhanced quality metrics
            writer.writerow(['# Quality Metrics'])
            writer.writerow(['reprojection_error_px', results['reprojection_error']])
            writer.writerow(['max_reprojection_error_px', results['max_reprojection_error']])
            writer.writerow(['validation_error_mean_px', results['validation_error_mean']])
            writer.writerow(['validation_error_median_px', results['validation_error_median']])
            writer.writerow(['camera_height_m', abs(results['camera_height'])])
            writer.writerow(['focal_length_px', results['focal_length']])
            writer.writerow(['num_inliers', results.get('num_inliers', len(results['point_names']))])
            writer.writerow(['total_points_available', results.get('total_points_available', 0)])
            writer.writerow(['num_validation_points', results['num_validation_points']])
            writer.writerow(['inlier_ratio', results.get('num_inliers', 0) / max(1, results.get('total_points_available', 1))])
            writer.writerow([''])

            # Point details
            writer.writerow(['# Inlier Point Reprojection Errors'])
            writer.writerow(['Point', 'Error_px'])
            for name in results['point_names']:
                if name in results['point_errors']:
                    writer.writerow([name, results['point_errors'][name]])

        print(f"✅ Results saved to: {output_path}")


def run_detection(video_path: str, output_path: str) -> bool:
    """Run court detection binary."""
    result = subprocess.run(
        f'./resources/detect {video_path} {output_path}',
        shell=True, capture_output=True, text=True
    )
    return result.returncode == 0


def get_video_size(video_path: str) -> Tuple[int, int]:
    """Get video frame dimensions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    print(f"📐 Video: {width}x{height}, {frame_count} frames @ {fps:.1f}fps")
    return (width, height)


def main():
    parser = argparse.ArgumentParser(description='Enhanced badminton court calibration with automatic coordinate system fixing')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--ransac-iterations', type=int, default=1000,
                        help='Number of RANSAC iterations (default: 1000)')
    parser.add_argument('--ransac-threshold', type=float, default=15.0,
                        help='RANSAC inlier threshold in pixels (default: 15.0)')
    args = parser.parse_args()

    print("🚀 Enhanced Badminton Court Calibrator with Auto-Fix")
    print("=" * 60)
    print("Features: Auto Coordinate Fix, RANSAC, More Points, Flexible Camera Models")

    # Setup paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    result_dir = os.path.join("results", video_name)
    os.makedirs(result_dir, exist_ok=True)

    court_csv = os.path.join(result_dir, "court.csv")
    output_csv = os.path.join(result_dir, f"{video_name}_fixed_calibration.csv")

    print(f"🎬 Video: {args.video_path}")
    print(f"💾 Results: {result_dir}")
    print(f"🎯 RANSAC: {args.ransac_iterations} iterations, {args.ransac_threshold}px threshold")

    # Step 1: Run detection
    print(f"\n📍 STEP 1: COURT DETECTION")
    if not run_detection(args.video_path, court_csv):
        raise RuntimeError("Court detection failed")

    if not os.path.exists(court_csv):
        raise RuntimeError(f"Detection output not found: {court_csv}")

    print("✅ Court Detection completed")

    # Step 2: Get video info
    print(f"\n📐 STEP 2: VIDEO ANALYSIS")
    image_size = get_video_size(args.video_path)

    # Step 3: Enhanced calibration with coordinate system fixing
    print(f"\n🔧 STEP 3: ENHANCED CALIBRATION WITH AUTO-FIX")
    calibrator = EnhancedBadmintonCalibratorWithFix(debug=args.debug)

    # Load detected points with automatic coordinate system fixing
    detected_points = calibrator.load_detected_points(court_csv, image_size)

    # Create mirrored version
    mirrored_points = calibrator.get_mirrored_points(detected_points)

    # Calibrate both with enhanced methods
    print("🔄 Calibrating normal points with RANSAC...")
    try:
        results_normal = calibrator.calibrate_camera(detected_points, image_size)
    except Exception as e:
        print(f"⚠️  Normal calibration failed: {e}")
        results_normal = None

    print("🔄 Calibrating mirrored points with RANSAC...")
    try:
        results_mirrored = calibrator.calibrate_camera(mirrored_points, image_size)
    except Exception as e:
        print(f"⚠️  Mirrored calibration failed: {e}")
        results_mirrored = None

    # Select best result with enhanced criteria
    if results_normal is None and results_mirrored is None:
        raise RuntimeError("Both calibration attempts failed")

    if results_normal is None:
        best_results = results_mirrored
        is_mirrored = True
        print("✅ Using mirrored points (normal failed)")
    elif results_mirrored is None:
        best_results = results_normal
        is_mirrored = False
        print("✅ Using normal points (mirrored failed)")
    else:
        # Compare results using multiple criteria
        normal_score = calibrator._compute_calibration_score(results_normal)
        mirrored_score = calibrator._compute_calibration_score(results_mirrored)

        if normal_score >= mirrored_score:
            best_results = results_normal
            is_mirrored = False
            print(f"✅ Selected normal points (score: {normal_score:.3f} vs {mirrored_score:.3f})")
        else:
            best_results = results_mirrored
            is_mirrored = True
            print(f"✅ Selected mirrored points (score: {mirrored_score:.3f} vs {normal_score:.3f})")

    # Perform additional geometric validation
    geometric_validation = calibrator._perform_geometric_validation(best_results, detected_points if not is_mirrored else mirrored_points)

    # Save results
    calibrator.save_calibration_results(best_results, output_csv, is_mirrored)

    # Enhanced summary
    print(f"\n🏆 ENHANCED CALIBRATION SUMMARY WITH AUTO-FIX")
    print("=" * 60)
    print(f"📊 Reprojection error: {best_results['reprojection_error']:.2f}px")
    print(f"📊 Validation error: {best_results['validation_error_mean']:.2f}px mean / {best_results['validation_error_median']:.2f}px median")
    print(f"📏 Camera height: {abs(best_results['camera_height']):.1f}m")
    print(f"🔍 Focal length: {best_results['focal_length']:.0f}px")
    print(f"🎯 Inliers: {best_results.get('num_inliers', 0)}/{best_results.get('total_points_available', 0)} points")
    print(f"🔄 Mirrored: {is_mirrored}")
    print(f"⚙️  Strategy: {best_results.get('calibration_strategy', 'Unknown')}")

    # Enhanced quality assessment
    inlier_ratio = best_results.get('num_inliers', 0) / max(1, best_results.get('total_points_available', 1))

    if (best_results['reprojection_error'] < 3.0 and
            best_results['validation_error_median'] < 10.0 and
            inlier_ratio > 0.7 and
            geometric_validation.get('overall_score', 0) > 0.8):
        print("🏆 EXCELLENT calibration!")
    elif (best_results['reprojection_error'] < 8.0 and
          best_results['validation_error_median'] < 25.0 and
          inlier_ratio > 0.5):
        print("✅ GOOD calibration")
    elif (best_results['reprojection_error'] < 15.0 and
          inlier_ratio > 0.4):
        print("⚠️  ACCEPTABLE calibration - some issues detected")
    else:
        print("❌ POOR calibration - review detection quality")

    # Geometric validation summary
    if geometric_validation:
        print(f"🔍 Geometric validation score: {geometric_validation.get('overall_score', 0):.2f}")
        if geometric_validation.get('warnings'):
            print("🔍 Geometric warnings:")
            for warning in geometric_validation['warnings']:
                print(f"   - {warning}")

    print(f"✅ Complete! Results saved to: {output_csv}")
    print("\n💡 Note: Coordinate system corrections were applied automatically")
    print("   Final output maintains standard coordinate system (P1 = top-left origin)")


if __name__ == "__main__":
    main()