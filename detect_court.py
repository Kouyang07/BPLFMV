import argparse
import os
import subprocess
import csv
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import random


class BadmintonCalibrator:
    """
    Simplified badminton court calibrator with basic coordinate system fixing.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.court_3d_coordinates = self._initialize_court_3d()

    def _initialize_court_3d(self) -> Dict[str, np.ndarray]:
        """Initialize 3D court coordinates in meters."""
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
            # Main court corners (on court surface)
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
        """Load court points from detection CSV with basic coordinate system fixing."""
        points = {}

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
                        points[name] = np.array([x, y])
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
                            points[name] = np.array([x, y])
                        except (ValueError, IndexError):
                            continue

        if self.debug:
            print(f"Loaded {len(points)} points from detection")

        # Simple coordinate system fix - flip Y if needed
        fixed_points = self._fix_coordinate_system(points, image_size)

        # Add missing NetCenter if needed
        if 'NetPole1' in fixed_points and 'NetPole2' in fixed_points and 'NetCenter' not in fixed_points:
            fixed_points['NetCenter'] = (fixed_points['NetPole1'] + fixed_points['NetPole2']) / 2.0

        return fixed_points

    def _fix_coordinate_system(self, points: Dict[str, np.ndarray], image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Enhanced coordinate system fix to ensure correct final orientation."""
        corners = ['P1', 'P2', 'P3', 'P4']
        if not all(p in points for p in corners):
            return points

        p1, p2, p3, p4 = points['P1'], points['P2'], points['P3'], points['P4']

        # Try different fixes to achieve correct orientation
        fixes_to_try = [
            ("original", lambda pts: pts),
            ("y_flip", lambda pts: {name: np.array([pt[0], image_size[1] - pt[1]]) for name, pt in pts.items()}),
            ("left_right_swap", self._swap_left_right_points),
            ("y_flip_and_swap", lambda pts: self._swap_left_right_points({name: np.array([pt[0], image_size[1] - pt[1]]) for name, pt in pts.items()}))
        ]

        for fix_name, fix_func in fixes_to_try:
            try:
                fixed_points = fix_func(points)
                if self._check_orientation(fixed_points):
                    if self.debug and fix_name != "original":
                        print(f"Applied coordinate fix: {fix_name}")
                    return fixed_points
            except:
                continue

        # If no fix works, return original and warn
        if self.debug:
            print("⚠️  Could not achieve correct coordinate orientation automatically")
        return points

    def _swap_left_right_points(self, points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Swap left-right point pairs to fix orientation."""
        swap_pairs = [
            ('P1', 'P4'), ('P2', 'P3'), ('P5', 'P8'), ('P6', 'P7'),
            ('P9', 'P10'), ('P11', 'P12'), ('P15', 'P16'), ('P17', 'P18'),
            ('P19', 'P20'), ('NetPole1', 'NetPole2')
        ]

        swapped = points.copy()
        for pair1, pair2 in swap_pairs:
            if pair1 in swapped and pair2 in swapped:
                swapped[pair1], swapped[pair2] = swapped[pair2], swapped[pair1]

        return swapped

    def _check_orientation(self, points: Dict[str, np.ndarray]) -> bool:
        """Check if points have correct orientation: P1=top-left, P4=top-right, P3=bottom-right, P2=bottom-left"""
        corners = ['P1', 'P2', 'P3', 'P4']
        if not all(p in points for p in corners):
            return False

        p1, p2, p3, p4 = points['P1'], points['P2'], points['P3'], points['P4']

        # Check expected relationships
        return (
                p1[1] < p2[1] and     # P1 above P2 (top-left above bottom-left)
                p4[1] < p3[1] and     # P4 above P3 (top-right above bottom-right)
                p1[0] < p4[0] and     # P1 left of P4 (top-left left of top-right)
                p2[0] < p3[0] and     # P2 left of P3 (bottom-left left of bottom-right)
                abs((p1[1] + p4[1])/2 - (p2[1] + p3[1])/2) > 10  # Reasonable vertical separation
        )

    def get_mirrored_points(self, detected_points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create mirrored version by swapping left-right pairs."""
        mirror_map = {
            'P1': 'P4', 'P4': 'P1', 'P2': 'P3', 'P3': 'P2',
            'P5': 'P8', 'P8': 'P5', 'P6': 'P7', 'P7': 'P6',
            'P9': 'P10', 'P10': 'P9', 'P11': 'P12', 'P12': 'P11',
            'P15': 'P16', 'P16': 'P15', 'P17': 'P18', 'P18': 'P17',
            'P19': 'P20', 'P20': 'P19', 'NetPole1': 'NetPole2', 'NetPole2': 'NetPole1',
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

        return mirrored

    def calibrate_camera(self, points_2d: Dict[str, np.ndarray], image_size: Tuple[int, int]) -> Dict:
        """Camera calibration with RANSAC for robustness."""
        # Get all available point pairs
        object_points = []
        image_points = []
        point_names = []

        for point_name in self.court_3d_coordinates.keys():
            if point_name in points_2d:
                object_points.append(self.court_3d_coordinates[point_name])
                image_points.append(points_2d[point_name])
                point_names.append(point_name)

        if len(object_points) < 8:
            raise ValueError(f"Need at least 8 points for calibration. Have {len(object_points)}")

        print(f"Calibrating with {len(object_points)} points")

        # Initial camera matrix estimate
        focal_estimate = max(image_size) * 0.8
        camera_matrix_init = np.array([
            [focal_estimate, 0, image_size[0]/2],
            [0, focal_estimate, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # RANSAC calibration
        best_result = self._ransac_calibration(
            object_points, image_points, point_names, image_size, camera_matrix_init
        )

        return best_result

    def _ransac_calibration(self, object_points: List[np.ndarray], image_points: List[np.ndarray],
                            point_names: List[str], image_size: Tuple[int, int],
                            camera_matrix_init: np.ndarray, max_iterations: int = 1000,
                            threshold: float = 15.0) -> Dict:
        """RANSAC calibration to handle outliers."""
        best_inliers = []
        best_error = float('inf')
        best_calibration = None

        for iteration in range(max_iterations):
            # Sample minimum points (8-12)
            sample_size = min(12, max(8, len(object_points) // 2))
            sample_indices = random.sample(range(len(object_points)), sample_size)

            try:
                sample_object = [object_points[i] for i in sample_indices]
                sample_image = [image_points[i] for i in sample_indices]

                # Try multiple calibration strategies for robustness
                results = self._try_calibration_strategies(
                    np.array([sample_object], dtype=np.float32),
                    np.array([sample_image], dtype=np.float32),
                    image_size, camera_matrix_init
                )

                # Choose best strategy based on reprojection error
                best_strategy = self._select_best_strategy(
                    results, object_points, image_points, point_names, threshold
                )

                if best_strategy is None:
                    continue

                strategy_name, K, dist, rvecs, tvecs, errors, inlier_mask = best_strategy
                current_inliers = [point_names[i] for i in range(len(errors)) if inlier_mask[i]]
                inlier_error = np.mean(errors[inlier_mask]) if np.any(inlier_mask) else float('inf')

                # Update best if this is better
                current_score = self._compute_strategy_score(
                    strategy_name, abs(tvecs[0][2]), K[0, 0], len(current_inliers), inlier_error
                )

                if current_score > best_error or len(current_inliers) > len(best_inliers):
                    best_inliers = current_inliers
                    best_error = current_score
                    best_calibration = {
                        'camera_matrix': K,
                        'dist_coeffs': dist,
                        'rvec': rvecs[0].flatten(),
                        'tvec': tvecs[0].flatten(),
                        'strategy': strategy_name,
                        'errors': errors,
                        'inlier_mask': inlier_mask
                    }

                # Early termination
                if len(current_inliers) >= min(20, int(0.8 * len(object_points))) and inlier_error < 5.0:
                    break

            except Exception as e:
                if self.debug and iteration < 5:
                    print(f"Iteration {iteration + 1} failed: {e}")
                continue

        if best_calibration is None:
            raise RuntimeError("RANSAC calibration failed")

        # Final refinement with inliers
        inlier_object = []
        inlier_image = []
        for i, name in enumerate(point_names):
            if name in best_inliers:
                inlier_object.append(object_points[i])
                inlier_image.append(image_points[i])

        if len(inlier_object) >= 8:
            try:
                object_array = np.array([inlier_object], dtype=np.float32)
                image_array = np.array([inlier_image], dtype=np.float32)

                if best_calibration['strategy'] == 'No Distortion':
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
                else:
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

                ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    object_array, image_array, image_size,
                    best_calibration['camera_matrix'].copy(),
                    best_calibration['dist_coeffs'].copy(),
                    flags=flags
                )

                if ret:
                    best_calibration.update({
                        'camera_matrix': K,
                        'dist_coeffs': dist,
                        'rvec': rvecs[0].flatten(),
                        'tvec': tvecs[0].flatten(),
                    })
            except:
                pass  # Use RANSAC result if refinement fails

        # Calculate final metrics
        camera_matrix = best_calibration['camera_matrix']
        dist_coeffs = best_calibration['dist_coeffs']
        rvec = best_calibration['rvec']
        tvec = best_calibration['tvec']

        # Compute errors for all points
        all_object = np.array([object_points], dtype=np.float32)
        projected, _ = cv2.projectPoints(
            all_object[0], rvec.reshape(3, 1), tvec.reshape(3, 1), camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        all_errors = np.sqrt(np.sum((np.array(image_points) - projected)**2, axis=1))

        inlier_errors = [all_errors[i] for i, name in enumerate(point_names) if name in best_inliers]

        result = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvec': rvec,
            'tvec': tvec,
            'reprojection_error': float(np.mean(inlier_errors)),
            'max_reprojection_error': float(np.max(inlier_errors)),
            'camera_height': float(abs(tvec[2])),
            'focal_length': float(camera_matrix[0, 0]),
            'point_names': best_inliers,
            'point_errors': {name: float(all_errors[i]) for i, name in enumerate(point_names) if name in best_inliers},
            'image_size': image_size,
            'calibration_strategy': best_calibration['strategy'],
            'num_inliers': len(best_inliers),
            'total_points_available': len(object_points),
        }

        # Validation on non-inlier points
        validation_points = [name for name in point_names if name not in best_inliers]
        validation_errors = [all_errors[i] for i, name in enumerate(point_names) if name in validation_points]

        result['validation_error_mean'] = float(np.mean(validation_errors)) if validation_errors else 0.0
        result['validation_error_median'] = float(np.median(validation_errors)) if validation_errors else 0.0
        result['num_validation_points'] = len(validation_errors)

        print(f"RANSAC result: {len(best_inliers)} inliers, {result['reprojection_error']:.2f}px error")
        return result

    def _try_calibration_strategies(self, sample_object: np.ndarray, sample_image: np.ndarray,
                                    image_size: Tuple[int, int], camera_matrix_init: np.ndarray) -> List:
        """Try multiple calibration strategies optimized for different camera types."""

        strategies = [
            # Strategy 1: No distortion (broadcast cameras, extreme angles)
            {
                'name': 'No Distortion',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
                'description': 'Best for broadcast cameras, extreme downward angles',
                'priority': 1
            },

            # Strategy 2: Minimal distortion K1 only (phone cameras)
            {
                'name': 'K1 Only',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
                'description': 'Best for phone cameras, moderate distortion',
                'priority': 2
            },

            # Strategy 3: K1+K2 radial (action cameras like GoPro)
            {
                'name': 'K1+K2 Radial',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
                'description': 'Best for action cameras (GoPro), wide-angle lenses',
                'priority': 3
            },

            # Strategy 4: Full distortion with constraints (fisheye/ultra-wide)
            {
                'name': 'Constrained Full',
                'flags': cv2.CALIB_USE_INTRINSIC_GUESS,
                'description': 'For fisheye or ultra-wide lenses',
                'priority': 4
            }
        ]

        results = []

        for strategy in strategies:
            try:
                ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    sample_object, sample_image, image_size,
                    camera_matrix_init.copy(), np.zeros(5),
                    flags=strategy['flags']
                )

                if ret:
                    # Validate distortion coefficients for sanity
                    if self._validate_distortion_sanity(dist, strategy['name']):
                        results.append((strategy, K, dist, rvecs, tvecs))
                        if self.debug:
                            print(f"   ✓ {strategy['name']}: Success")
                    elif self.debug:
                        print(f"   ✗ {strategy['name']}: Extreme distortion detected")

            except Exception as e:
                if self.debug:
                    print(f"   ✗ {strategy['name']}: {e}")
                continue

        return results

    def _validate_distortion_sanity(self, dist_coeffs: np.ndarray, strategy_name: str) -> bool:
        """Check if distortion coefficients are reasonable for the strategy."""
        # Ensure we have at least 5 coefficients
        if len(dist_coeffs.flatten()) < 5:
            return False

        k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

        # Define reasonable bounds based on camera type
        if strategy_name == 'No Distortion':
            return True  # No distortion to validate

        elif strategy_name == 'K1 Only':
            # Phone cameras: moderate barrel/pincushion distortion
            return abs(k1) < 2.0

        elif strategy_name == 'K1+K2 Radial':
            # Action cameras: higher distortion but still reasonable
            return abs(k1) < 5.0 and abs(k2) < 50.0

        elif strategy_name == 'Constrained Full':
            # Full model: more lenient but catch extreme cases
            return (abs(k1) < 10.0 and abs(k2) < 200.0 and
                    abs(p1) < 0.01 and abs(p2) < 0.01 and abs(k3) < 500.0)

        return True

    def _select_best_strategy(self, results: List, object_points: List[np.ndarray],
                              image_points: List[np.ndarray], point_names: List[str],
                              threshold: float) -> Optional[tuple]:
        """Select best calibration strategy using camera type heuristics."""

        best_strategy = None
        best_score = -1

        for strategy, K, dist, rvecs, tvecs in results:
            # Test all points to find inliers
            all_object = np.array([object_points], dtype=np.float32)
            projected, _ = cv2.projectPoints(
                all_object[0], rvecs[0], tvecs[0], K, dist
            )
            projected = projected.reshape(-1, 2)

            errors = np.sqrt(np.sum((np.array(image_points) - projected)**2, axis=1))
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            mean_error = np.mean(errors[inlier_mask]) if inlier_count > 0 else float('inf')

            # Camera type detection heuristics
            camera_height = abs(tvecs[0][2])
            focal_length = K[0, 0]

            # Score based on camera type suitability
            strategy_score = self._compute_strategy_score(
                strategy['name'], camera_height, focal_length, inlier_count, mean_error
            )

            if self.debug:
                print(f"     {strategy['name']}: {inlier_count} inliers, {mean_error:.2f}px, score: {strategy_score:.3f}")

            if strategy_score > best_score:
                best_score = strategy_score
                best_strategy = (strategy['name'], K, dist, rvecs, tvecs, errors, inlier_mask)

        return best_strategy

    def _compute_strategy_score(self, strategy_name: str, camera_height: float,
                                focal_length: float, inlier_count: int, mean_error: float) -> float:
        """Compute strategy score based on camera type detection and quality metrics."""

        # Base quality score (higher inliers, lower error = better)
        quality_score = inlier_count / max(1, mean_error)

        # Camera type bonuses based on heuristics
        camera_type_bonus = 1.0

        # Broadcast camera detection (high mount, long focal length)
        if camera_height > 20 and focal_length > 3000:
            if strategy_name == 'No Distortion':
                camera_type_bonus = 2.0  # Strong preference for no distortion
            elif strategy_name == 'K1 Only':
                camera_type_bonus = 1.5

        # Phone camera detection (moderate height, moderate focal length)
        elif 5 < camera_height < 25 and 1000 < focal_length < 4000:
            if strategy_name == 'K1 Only':
                camera_type_bonus = 2.0  # Phones usually have moderate distortion
            elif strategy_name == 'No Distortion':
                camera_type_bonus = 1.5

        # Action camera detection (low height or very wide angle)
        elif camera_height < 10 or focal_length < 1500:
            if strategy_name == 'K1+K2 Radial':
                camera_type_bonus = 2.0  # GoPros have significant barrel distortion
            elif strategy_name == 'Constrained Full':
                camera_type_bonus = 1.5

        # Ultra-wide/fisheye detection (very short focal length)
        elif focal_length < 800:
            if strategy_name == 'Constrained Full':
                camera_type_bonus = 2.0
            elif strategy_name == 'K1+K2 Radial':
                camera_type_bonus = 1.5

        # Priority bonus (simpler models preferred for equal performance)
        strategy_priorities = {
            'No Distortion': 1.2,      # Simplest, most robust
            'K1 Only': 1.1,           # Good balance
            'K1+K2 Radial': 1.0,      # More complex
            'Constrained Full': 0.9   # Most complex
        }
        priority_bonus = strategy_priorities.get(strategy_name, 1.0)

        final_score = quality_score * camera_type_bonus * priority_bonus

        return final_score

    def fix_coordinate_orientation(self, points: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Attempt to fix coordinate orientation automatically.
        Returns (fixed_points, was_fixed).
        """
        if _verify_coordinate_orientation(points):
            return points, False

        # Try different transformations to fix orientation
        fixes = [
            ("y_flip", lambda pts: {name: np.array([pt[0], max(pt[1] for pt in pts.values()) - pt[1]]) for name, pt in pts.items()}),
            ("x_flip", lambda pts: {name: np.array([max(pt[0] for pt in pts.values()) - pt[0], pt[1]]) for name, pt in pts.items()}),
            ("xy_flip", lambda pts: {name: np.array([max(pt[0] for pt in pts.values()) - pt[0], max(pt[1] for pt in pts.values()) - pt[1]]) for name, pt in pts.items()}),
            ("swap_xy", lambda pts: {name: np.array([pt[1], pt[0]]) for name, pt in pts.items()}),
        ]

        for fix_name, fix_func in fixes:
            try:
                fixed_points = fix_func(points)
                if _verify_coordinate_orientation(fixed_points):
                    if self.debug:
                        print(f"Fixed coordinate orientation using: {fix_name}")
                    return fixed_points, True
            except:
                continue

        return points, False

    def save_calibration_results(self, results: Dict, output_path: str, is_mirrored: bool = False) -> None:
        """Save calibration results to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['# Badminton Court Calibration Results'])
            writer.writerow(['# Coordinate System: P1 = top-left origin (0,0,0)'])
            writer.writerow([f'# Mirrored points used: {is_mirrored}'])
            writer.writerow([f'# Calibration strategy: {results.get("calibration_strategy", "Unknown")}'])
            writer.writerow([''])

            # Camera intrinsics
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

            # Extrinsics
            writer.writerow(['# Camera Pose'])
            writer.writerow(['rx', results['rvec'][0]])
            writer.writerow(['ry', results['rvec'][1]])
            writer.writerow(['rz', results['rvec'][2]])
            writer.writerow(['tx', results['tvec'][0]])
            writer.writerow(['ty', results['tvec'][1]])
            writer.writerow(['tz', results['tvec'][2]])
            writer.writerow([''])

            # Quality metrics
            writer.writerow(['# Quality Metrics'])
            writer.writerow(['reprojection_error_px', results['reprojection_error']])
            writer.writerow(['max_reprojection_error_px', results['max_reprojection_error']])
            writer.writerow(['validation_error_mean_px', results['validation_error_mean']])
            writer.writerow(['validation_error_median_px', results['validation_error_median']])
            writer.writerow(['camera_height_m', results['camera_height']])
            writer.writerow(['focal_length_px', results['focal_length']])
            writer.writerow(['num_inliers', results['num_inliers']])
            writer.writerow(['total_points_available', results['total_points_available']])
            writer.writerow(['num_validation_points', results['num_validation_points']])
            writer.writerow([''])

            # Point details
            writer.writerow(['# Inlier Point Reprojection Errors'])
            writer.writerow(['Point', 'Error_px'])
            for name in results['point_names']:
                if name in results['point_errors']:
                    writer.writerow([name, results['point_errors'][name]])

        print(f"Results saved to: {output_path}")

    def save_court_points(self, points: Dict[str, np.ndarray], output_path: str) -> None:
        """Save court points for pose detection."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Point', 'X', 'Y'])
            for point_name, coords in points.items():
                writer.writerow([point_name, coords[0], coords[1]])


def run_detection(video_path: str, output_path: str) -> bool:
    """Run court detection binary."""
    result = subprocess.run(
        f'./resources/detect {video_path} {output_path}',
        shell=True, capture_output=True, text=True
    )
    return result.returncode == 0


def _verify_coordinate_orientation(points: Dict[str, np.ndarray]) -> bool:
    """Verify that points follow the expected orientation:
    P1=top-left, P4=top-right, P3=bottom-right, P2=bottom-left"""
    corners = ['P1', 'P2', 'P3', 'P4']
    if not all(p in points for p in corners):
        return False

    p1, p2, p3, p4 = points['P1'], points['P2'], points['P3'], points['P4']

    # Check relationships:
    # P1 should be top-left: smallest Y, smallest X among top points
    # P4 should be top-right: smallest Y, largest X among top points
    # P2 should be bottom-left: largest Y, smallest X among bottom points
    # P3 should be bottom-right: largest Y, largest X among bottom points

    # Top points should have smaller Y than bottom points
    top_y = min(p1[1], p4[1])
    bottom_y = max(p2[1], p3[1])

    # Left points should have smaller X than right points
    left_x = min(p1[0], p2[0])
    right_x = max(p4[0], p3[0])

    # Basic checks
    orientation_correct = (
            top_y < bottom_y and  # Top above bottom
            left_x < right_x and  # Left left of right
            p1[1] < p2[1] and     # P1 above P2
            p4[1] < p3[1] and     # P4 above P3
            p1[0] < p4[0] and     # P1 left of P4
            p2[0] < p3[0]         # P2 left of P3
    )

    return orientation_correct


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
    print(f"Video: {width}x{height}, {frame_count} frames @ {fps:.1f}fps")
    return (width, height)


def main():
    parser = argparse.ArgumentParser(description='Simplified badminton court calibration')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    print("🚀 Badminton Court Calibrator")
    print("=" * 50)

    # Setup paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    result_dir = os.path.join("results", video_name)
    os.makedirs(result_dir, exist_ok=True)

    court_csv = os.path.join(result_dir, "court.csv")
    output_csv = os.path.join(result_dir, "calibration.csv")

    print(f"Video: {args.video_path}")
    print(f"Results: {result_dir}")

    # Step 1: Run detection
    print(f"\nSTEP 1: COURT DETECTION")
    if not run_detection(args.video_path, court_csv):
        raise RuntimeError("Court detection failed")

    if not os.path.exists(court_csv):
        raise RuntimeError(f"Detection output not found: {court_csv}")

    print("✅ Court Detection completed")

    # Step 2: Get video info
    print(f"\nSTEP 2: VIDEO ANALYSIS")
    image_size = get_video_size(args.video_path)

    # Step 3: Calibration
    print(f"\nSTEP 3: CALIBRATION")
    calibrator = BadmintonCalibrator(debug=args.debug)

    # Load detected points and try to fix coordinate orientation
    detected_points = calibrator.load_detected_points(court_csv, image_size)
    detected_points, was_fixed = calibrator.fix_coordinate_orientation(detected_points)
    if was_fixed:
        print("✅ Automatically fixed coordinate orientation")

    calibrator.save_court_points(detected_points, court_csv)

    # Create mirrored version
    mirrored_points = calibrator.get_mirrored_points(detected_points)
    mirrored_points, was_mirrored_fixed = calibrator.fix_coordinate_orientation(mirrored_points)
    if was_mirrored_fixed:
        print("✅ Automatically fixed mirrored coordinate orientation")

    # Calibrate both versions
    results_normal = None
    results_mirrored = None

    try:
        print("Calibrating normal points...")
        results_normal = calibrator.calibrate_camera(detected_points, image_size)
    except Exception as e:
        print(f"Normal calibration failed: {e}")

    try:
        print("Calibrating mirrored points...")
        results_mirrored = calibrator.calibrate_camera(mirrored_points, image_size)
    except Exception as e:
        print(f"Mirrored calibration failed: {e}")

    # Select best result - ENSURE CORRECT COORDINATE ORIENTATION
    if results_normal is None and results_mirrored is None:
        raise RuntimeError("Both calibration attempts failed")

    # Prioritize coordinate orientation correctness over calibration quality
    normal_orientation_correct = results_normal is not None and _verify_coordinate_orientation(detected_points)
    mirrored_orientation_correct = results_mirrored is not None and _verify_coordinate_orientation(mirrored_points)

    if normal_orientation_correct and not mirrored_orientation_correct:
        best_results = results_normal
        is_mirrored = False
        print("✅ Selected normal points (correct orientation)")
    elif mirrored_orientation_correct and not normal_orientation_correct:
        best_results = results_mirrored
        is_mirrored = True
        print("✅ Selected mirrored points (correct orientation)")
    elif normal_orientation_correct and mirrored_orientation_correct:
        # Both have correct orientation, choose based on quality
        normal_score = results_normal['num_inliers'] / max(1, results_normal['reprojection_error'])
        mirrored_score = results_mirrored['num_inliers'] / max(1, results_mirrored['reprojection_error'])

        if normal_score >= mirrored_score:
            best_results = results_normal
            is_mirrored = False
            print("✅ Selected normal points (better quality, correct orientation)")
        else:
            best_results = results_mirrored
            is_mirrored = True
            print("✅ Selected mirrored points (better quality, correct orientation)")
    else:
        # Neither has correct orientation, choose based on quality and warn
        if results_normal is None:
            best_results = results_mirrored
            is_mirrored = True
            print("⚠️  Using mirrored points (orientation incorrect)")
        elif results_mirrored is None:
            best_results = results_normal
            is_mirrored = False
            print("⚠️  Using normal points (orientation incorrect)")
        else:
            normal_score = results_normal['num_inliers'] / max(1, results_normal['reprojection_error'])
            mirrored_score = results_mirrored['num_inliers'] / max(1, results_mirrored['reprojection_error'])

            if normal_score >= mirrored_score:
                best_results = results_normal
                is_mirrored = False
                print("⚠️  Selected normal points (better quality, but orientation incorrect)")
            else:
                best_results = results_mirrored
                is_mirrored = True
                print("⚠️  Selected mirrored points (better quality, but orientation incorrect)")

    # Final verification and warning
    final_points = mirrored_points if is_mirrored else detected_points
    if not _verify_coordinate_orientation(final_points):
        print("\n⚠️  WARNING: Final coordinate orientation is incorrect!")
        print("   Expected: P1=top-left, P4=top-right, P3=bottom-right, P2=bottom-left")
        print("   This may cause issues in downstream applications that rely on court coordinates.")
        print("   Consider manually adjusting the court detection or coordinate mapping.")

    # Save results
    calibrator.save_calibration_results(best_results, output_csv, is_mirrored)

    # Summary
    print(f"\n🏆 CALIBRATION SUMMARY")
    print("=" * 50)
    print(f"Reprojection error: {best_results['reprojection_error']:.2f}px")
    print(f"Camera height: {best_results['camera_height']:.1f}m")
    print(f"Focal length: {best_results['focal_length']:.0f}px")
    print(f"Inliers: {best_results['num_inliers']}/{best_results['total_points_available']} points")
    print(f"Mirrored: {is_mirrored}")
    print(f"Strategy: {best_results['calibration_strategy']}")

    # Quality assessment
    if (best_results['reprojection_error'] < 5.0 and
            best_results['num_inliers'] >= 10):
        print("🏆 EXCELLENT calibration!")
    elif (best_results['reprojection_error'] < 10.0 and
          best_results['num_inliers'] >= 8):
        print("✅ GOOD calibration")
    else:
        print("⚠️  ACCEPTABLE calibration")

    print(f"\n✅ Complete! Results saved to: {output_csv}")


if __name__ == "__main__":
    main()