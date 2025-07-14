#!/usr/bin/env python3
"""
Enhanced Badminton Tracking Evaluator with Jump Correction Comparison

Evaluates tracking performance against ground truth and compares the improvement
from jump correction by analyzing both positions.json and corrected_positions.json.

Usage: python3 evaluator.py <video_file_path>

Requirements:
    pip install numpy matplotlib
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math


class TrackingEvaluator:
    """Enhanced evaluator that compares original and corrected tracking results."""

    def __init__(self, video_path: str):
        """Initialize the evaluator."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name

        # Data containers
        self.ground_truth = {}  # frame -> (x, y)
        self.original_positions = {}  # frame -> [(player_data), ...]
        self.corrected_positions = {}  # frame -> [(player_data), ...]

        # Evaluation results
        self.original_errors = []
        self.corrected_errors = []
        self.frame_comparisons = []  # For detailed frame-by-frame analysis

    def load_ground_truth(self) -> bool:
        """Load ground truth data."""
        gt_files = [
            "ground_truth_positions.json",
            "video_click_ground_truth.json",
            "manual_annotations.json"
        ]

        for filename in gt_files:
            gt_file = self.results_dir / filename
            if gt_file.exists():
                try:
                    with open(gt_file, 'r') as f:
                        data = json.load(f)

                    positions = data.get('player_positions', [])
                    for pos in positions:
                        frame_idx = pos['frame_index']
                        # Use hip position as ground truth
                        self.ground_truth[frame_idx] = (
                            pos['hip_world_X'],
                            pos['hip_world_Y']
                        )

                    print(f"✅ Loaded ground truth from: {filename}")
                    print(f"   Ground truth frames: {len(self.ground_truth)}")
                    return True

                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    continue

        print(f"❌ No ground truth file found in {self.results_dir}")
        return False

    def load_tracking_data(self) -> Tuple[bool, bool]:
        """Load original and corrected tracking data."""
        original_loaded = False
        corrected_loaded = False

        # Load original positions
        original_file = self.results_dir / "positions.json"
        if original_file.exists():
            try:
                with open(original_file, 'r') as f:
                    data = json.load(f)

                for position in data.get('player_positions', []):
                    frame_idx = position['frame_index']
                    if frame_idx not in self.original_positions:
                        self.original_positions[frame_idx] = []
                    self.original_positions[frame_idx].append(position)

                print(f"✅ Loaded original tracking results from: positions.json")
                print(f"   Original tracking frames: {len(self.original_positions)}")
                original_loaded = True

            except Exception as e:
                print(f"❌ Error loading positions.json: {e}")
        else:
            print(f"⚠️  Original positions file not found: positions.json")

        # Load corrected positions
        corrected_file = self.results_dir / "corrected_positions.json"
        if corrected_file.exists():
            try:
                with open(corrected_file, 'r') as f:
                    data = json.load(f)

                for position in data.get('player_positions', []):
                    frame_idx = position['frame_index']
                    if frame_idx not in self.corrected_positions:
                        self.corrected_positions[frame_idx] = []
                    self.corrected_positions[frame_idx].append(position)

                print(f"✅ Loaded corrected tracking results from: corrected_positions.json")
                print(f"   Corrected tracking frames: {len(self.corrected_positions)}")
                corrected_loaded = True

            except Exception as e:
                print(f"❌ Error loading corrected_positions.json: {e}")
        else:
            print(f"⚠️  Corrected positions file not found: corrected_positions.json")

        return original_loaded, corrected_loaded

    def calculate_midpoint_position(self, position_data):
        """Calculate midpoint of hip, left ankle, and right ankle coordinates."""
        hip_x = position_data['hip_world_X']
        hip_y = position_data['hip_world_Y']
        left_ankle_x = position_data['left_ankle_world_X']
        left_ankle_y = position_data['left_ankle_world_Y']
        right_ankle_x = position_data['right_ankle_world_X']
        right_ankle_y = position_data['right_ankle_world_Y']

        midpoint_x = (hip_x + left_ankle_x + right_ankle_x) / 3
        midpoint_y = (hip_y + left_ankle_y + right_ankle_y) / 3

        return midpoint_x, midpoint_y

    def find_closest_player(self, positions):
        """Find the closest player to camera (lowest player_id)."""
        if not positions:
            return None
        return min(positions, key=lambda p: p.get('player_id', 0))

    def calculate_error(self, predicted_pos, ground_truth_pos):
        """Calculate Euclidean distance error."""
        pred_x, pred_y = predicted_pos
        gt_x, gt_y = ground_truth_pos
        return math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

    def evaluate_tracking_data(self, positions_dict, data_type="tracking"):
        """Evaluate tracking data against ground truth."""
        errors = []
        frame_results = []

        for frame_idx, gt_pos in self.ground_truth.items():
            if frame_idx in positions_dict:
                positions = positions_dict[frame_idx]
                closest_player = self.find_closest_player(positions)

                if closest_player:
                    predicted_pos = self.calculate_midpoint_position(closest_player)
                    error = self.calculate_error(predicted_pos, gt_pos)
                    errors.append(error)

                    frame_results.append({
                        'frame_index': frame_idx,
                        'ground_truth': gt_pos,
                        'predicted': predicted_pos,
                        'error': error,
                        'data_type': data_type
                    })

        return errors, frame_results

    def calculate_metrics(self, errors):
        """Calculate comprehensive tracking metrics."""
        if not errors:
            return {}

        errors_array = np.array(errors)

        metrics = {
            'total_comparisons': len(errors),
            'mean_error': float(np.mean(errors_array)),
            'median_error': float(np.median(errors_array)),
            'std_error': float(np.std(errors_array)),
            'min_error': float(np.min(errors_array)),
            'max_error': float(np.max(errors_array)),
            'rmse': float(np.sqrt(np.mean(errors_array**2))),
            'percentiles': {
                '25th': float(np.percentile(errors_array, 25)),
                '50th': float(np.percentile(errors_array, 50)),
                '75th': float(np.percentile(errors_array, 75)),
                '90th': float(np.percentile(errors_array, 90)),
                '95th': float(np.percentile(errors_array, 95)),
                '99th': float(np.percentile(errors_array, 99))
            },
            'accuracy_thresholds': {
                '0.25m': (errors_array < 0.25).sum(),
                '0.5m': (errors_array < 0.5).sum(),
                '1.0m': (errors_array < 1.0).sum(),
                '1.5m': (errors_array < 1.5).sum(),
                '2.0m': (errors_array < 2.0).sum()
            }
        }

        return metrics

    def calculate_improvement_metrics(self):
        """Calculate improvement metrics between original and corrected."""
        if not self.original_errors or not self.corrected_errors:
            return {}

        # Find common frames
        original_frames = {comp['frame_index'] for comp in self.frame_comparisons if comp['data_type'] == 'original'}
        corrected_frames = {comp['frame_index'] for comp in self.frame_comparisons if comp['data_type'] == 'corrected'}
        common_frames = original_frames.intersection(corrected_frames)

        if not common_frames:
            return {}

        # Get errors for common frames only
        original_common = []
        corrected_common = []
        frame_improvements = []

        for comp in self.frame_comparisons:
            if comp['frame_index'] in common_frames:
                if comp['data_type'] == 'original':
                    original_common.append(comp['error'])
                elif comp['data_type'] == 'corrected':
                    corrected_common.append(comp['error'])

        # Calculate frame-by-frame improvements
        original_dict = {comp['frame_index']: comp['error'] for comp in self.frame_comparisons if comp['data_type'] == 'original'}
        corrected_dict = {comp['frame_index']: comp['error'] for comp in self.frame_comparisons if comp['data_type'] == 'corrected'}

        for frame_idx in common_frames:
            if frame_idx in original_dict and frame_idx in corrected_dict:
                orig_error = original_dict[frame_idx]
                corr_error = corrected_dict[frame_idx]
                improvement = orig_error - corr_error
                improvement_pct = (improvement / orig_error) * 100 if orig_error > 0 else 0

                frame_improvements.append({
                    'frame_index': frame_idx,
                    'original_error': orig_error,
                    'corrected_error': corr_error,
                    'absolute_improvement': improvement,
                    'percent_improvement': improvement_pct
                })

        # Calculate overall improvement statistics
        improvements = [f['absolute_improvement'] for f in frame_improvements]
        improvement_pcts = [f['percent_improvement'] for f in frame_improvements]

        original_mean = np.mean(original_common)
        corrected_mean = np.mean(corrected_common)

        improvement_metrics = {
            'common_frames': len(common_frames),
            'original_mean_error': float(original_mean),
            'corrected_mean_error': float(corrected_mean),
            'absolute_improvement': float(original_mean - corrected_mean),
            'percent_improvement': float(((original_mean - corrected_mean) / original_mean) * 100),
            'improved_frames': sum(1 for imp in improvements if imp > 0),
            'degraded_frames': sum(1 for imp in improvements if imp < 0),
            'unchanged_frames': sum(1 for imp in improvements if abs(imp) < 0.001),
            'mean_frame_improvement': float(np.mean(improvements)),
            'median_frame_improvement': float(np.median(improvements)),
            'max_improvement': float(np.max(improvements)),
            'max_degradation': float(np.min(improvements)),
            'improvement_std': float(np.std(improvements)),
            'frame_improvements': frame_improvements
        }

        return improvement_metrics

    def create_comparison_plots(self):
        """Create simple comparison plots."""
        try:
            if not self.original_errors and not self.corrected_errors:
                print("⚠️  No data available for plotting")
                return

            # Create simple comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Tracking Performance Comparison - {self.video_name}', fontsize=14, fontweight='bold')

            # Plot 1: Error distribution comparison
            if self.original_errors and self.corrected_errors:
                ax1.hist(self.original_errors, bins=30, alpha=0.7, label='Original', color='red', density=True)
                ax1.hist(self.corrected_errors, bins=30, alpha=0.7, label='Corrected', color='blue', density=True)
                ax1.set_xlabel('Error (meters)')
                ax1.set_ylabel('Density')
                ax1.set_title('Error Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Plot 2: Box plot comparison
            if self.original_errors and self.corrected_errors:
                data_to_plot = [self.original_errors, self.corrected_errors]
                labels = ['Original', 'Corrected']
                bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('red')
                bp['boxes'][1].set_facecolor('blue')
                ax2.set_ylabel('Error (meters)')
                ax2.set_title('Error Statistics')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = self.results_dir / "comparison_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plots saved to: {plot_file}")

            plt.close()

        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")

    def generate_comparison_report(self, original_metrics, corrected_metrics, improvement_metrics):
        """Generate comprehensive comparison report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = []
        report.append("BADMINTON TRACKING COMPARISON REPORT")
        report.append("=" * 50)
        report.append(f"Video: {self.video_name}")
        report.append(f"Date: {timestamp}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)

        if original_metrics and corrected_metrics:
            report.append(f"Original Mean Error: {original_metrics['mean_error']:.3f} meters")
            report.append(f"Corrected Mean Error: {corrected_metrics['mean_error']:.3f} meters")

            if improvement_metrics:
                report.append(f"Absolute Improvement: {improvement_metrics['absolute_improvement']:.3f} meters")
                report.append(f"Relative Improvement: {improvement_metrics['percent_improvement']:.1f}%")
                report.append(f"Frames Improved: {improvement_metrics['improved_frames']}/{improvement_metrics['common_frames']} ({improvement_metrics['improved_frames']/improvement_metrics['common_frames']*100:.1f}%)")

        report.append("")

        # Original Performance
        if original_metrics:
            report.append("ORIGINAL TRACKING PERFORMANCE")
            report.append("-" * 35)
            report.append(f"Total Comparisons: {original_metrics['total_comparisons']}")
            report.append(f"Mean Absolute Error (MAE): {original_metrics['mean_error']:.3f} meters")
            report.append(f"Root Mean Square Error (RMSE): {original_metrics['rmse']:.3f} meters")
            report.append(f"Median Error: {original_metrics['median_error']:.3f} meters")
            report.append(f"Standard Deviation: {original_metrics['std_error']:.3f} meters")
            report.append(f"95th Percentile Error: {original_metrics['percentiles']['95th']:.3f} meters")
            report.append(f"Precision at 1m: {original_metrics['accuracy_thresholds']['1.0m']/original_metrics['total_comparisons']*100:.1f}%")
            report.append("")

        # Corrected Performance
        if corrected_metrics:
            report.append("CORRECTED TRACKING PERFORMANCE")
            report.append("-" * 36)
            report.append(f"Total Comparisons: {corrected_metrics['total_comparisons']}")
            report.append(f"Mean Absolute Error (MAE): {corrected_metrics['mean_error']:.3f} meters")
            report.append(f"Root Mean Square Error (RMSE): {corrected_metrics['rmse']:.3f} meters")
            report.append(f"Median Error: {corrected_metrics['median_error']:.3f} meters")
            report.append(f"Standard Deviation: {corrected_metrics['std_error']:.3f} meters")
            report.append(f"95th Percentile Error: {corrected_metrics['percentiles']['95th']:.3f} meters")
            report.append(f"Precision at 1m: {corrected_metrics['accuracy_thresholds']['1.0m']/corrected_metrics['total_comparisons']*100:.1f}%")
            report.append("")

        # Improvement Analysis
        if improvement_metrics:
            report.append("JUMP CORRECTION IMPROVEMENT ANALYSIS")
            report.append("-" * 42)
            report.append(f"Common Frames Analyzed: {improvement_metrics['common_frames']}")
            report.append(f"Mean Error Reduction: {improvement_metrics['absolute_improvement']:.3f} meters ({improvement_metrics['percent_improvement']:.1f}%)")
            report.append(f"Median Frame Improvement: {improvement_metrics['median_frame_improvement']:.3f} meters")
            report.append(f"Maximum Single-Frame Improvement: {improvement_metrics['max_improvement']:.3f} meters")
            report.append(f"Maximum Single-Frame Degradation: {abs(improvement_metrics['max_degradation']):.3f} meters")
            report.append("")
            report.append("Frame-by-Frame Results:")
            report.append(f"  Improved frames: {improvement_metrics['improved_frames']} ({improvement_metrics['improved_frames']/improvement_metrics['common_frames']*100:.1f}%)")
            report.append(f"  Degraded frames: {improvement_metrics['degraded_frames']} ({improvement_metrics['degraded_frames']/improvement_metrics['common_frames']*100:.1f}%)")
            report.append(f"  Unchanged frames: {improvement_metrics['unchanged_frames']} ({improvement_metrics['unchanged_frames']/improvement_metrics['common_frames']*100:.1f}%)")
            report.append("")

        # Detailed Accuracy Comparison
        if original_metrics and corrected_metrics:
            report.append("ACCURACY THRESHOLD COMPARISON")
            report.append("-" * 34)
            thresholds = ['0.25m', '0.5m', '1.0m', '1.5m', '2.0m']
            for threshold in thresholds:
                orig_count = original_metrics['accuracy_thresholds'][threshold]
                orig_pct = orig_count / original_metrics['total_comparisons'] * 100
                corr_count = corrected_metrics['accuracy_thresholds'][threshold]
                corr_pct = corr_count / corrected_metrics['total_comparisons'] * 100
                improvement = corr_pct - orig_pct

                report.append(f"Errors < {threshold}:")
                report.append(f"  Original: {orig_count} ({orig_pct:.1f}%)")
                report.append(f"  Corrected: {corr_count} ({corr_pct:.1f}%)")
                report.append(f"  Improvement: {improvement:+.1f} percentage points")
                report.append("")

        # Performance Grade
        if corrected_metrics:
            mae = corrected_metrics['mean_error']
            if mae < 0.2:
                grade = "EXCELLENT"
            elif mae < 0.3:
                grade = "VERY GOOD"
            elif mae < 0.5:
                grade = "GOOD"
            elif mae < 1.0:
                grade = "ACCEPTABLE"
            else:
                grade = "NEEDS IMPROVEMENT"

            report.append("FINAL PERFORMANCE GRADE")
            report.append("-" * 26)
            report.append(f"Corrected Tracking Performance: {grade} (MAE: {mae:.3f}m)")

            if improvement_metrics and improvement_metrics['percent_improvement'] > 0:
                report.append(f"Jump Correction Effectiveness: {improvement_metrics['percent_improvement']:.1f}% improvement")
            report.append("")

        return "\n".join(report)

    def save_detailed_results(self, original_metrics, corrected_metrics, improvement_metrics):
        """Save detailed results to JSON."""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj

        detailed_results = {
            'evaluation_info': {
                'video_name': self.video_name,
                'video_path': str(self.video_path),
                'timestamp': datetime.now().isoformat(),
                'evaluator_version': '2.0_simplified'
            },
            'summary': {
                'original_mean_error': float(original_metrics.get('mean_error', 0)) if original_metrics else None,
                'corrected_mean_error': float(corrected_metrics.get('mean_error', 0)) if corrected_metrics else None,
                'absolute_improvement': float(improvement_metrics.get('absolute_improvement', 0)) if improvement_metrics else None,
                'percent_improvement': float(improvement_metrics.get('percent_improvement', 0)) if improvement_metrics else None,
                'frames_improved': int(improvement_metrics.get('improved_frames', 0)) if improvement_metrics else None
            }
        }

        # Convert all numpy types to native Python types
        detailed_results = convert_numpy_types(detailed_results)

        results_file = self.results_dir / "comparison_summary.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"✅ Comparison summary saved to: {results_file}")

    def run_evaluation(self):
        """Run the complete evaluation."""
        print(f"Starting comparison evaluation for: {self.video_name}")
        print("=" * 60)

        # Load all data
        if not self.load_ground_truth():
            return False

        original_loaded, corrected_loaded = self.load_tracking_data()

        if not original_loaded and not corrected_loaded:
            print("❌ No tracking data found!")
            return False

        print("\nCalculating tracking errors...")

        # Evaluate original tracking
        original_metrics = {}
        if original_loaded:
            self.original_errors, original_frames = self.evaluate_tracking_data(
                self.original_positions, "original"
            )
            self.frame_comparisons.extend(original_frames)
            original_metrics = self.calculate_metrics(self.original_errors)
            print(f"✅ Original tracking evaluation complete: {len(self.original_errors)} comparisons")

        # Evaluate corrected tracking
        corrected_metrics = {}
        if corrected_loaded:
            self.corrected_errors, corrected_frames = self.evaluate_tracking_data(
                self.corrected_positions, "corrected"
            )
            self.frame_comparisons.extend(corrected_frames)
            corrected_metrics = self.calculate_metrics(self.corrected_errors)
            print(f"✅ Corrected tracking evaluation complete: {len(self.corrected_errors)} comparisons")

        # Calculate improvement metrics
        improvement_metrics = {}
        if original_loaded and corrected_loaded:
            improvement_metrics = self.calculate_improvement_metrics()
            print(f"✅ Improvement analysis complete: {improvement_metrics.get('common_frames', 0)} frames compared")

        # Generate report
        report = self.generate_comparison_report(original_metrics, corrected_metrics, improvement_metrics)
        print("\n" + report)

        # Save report
        report_file = self.results_dir / "tracking_comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✅ Comparison report saved to: {report_file}")

        # Create plots
        self.create_comparison_plots()

        # Save detailed results
        self.save_detailed_results(original_metrics, corrected_metrics, improvement_metrics)

        print("✅ Evaluation completed successfully!")
        return True


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python3 evaluator.py <video_file_path>")
        print("\nBadminton Tracking Evaluator with Jump Correction Comparison")
        print("=" * 65)
        print("This evaluator compares original and corrected tracking performance:")
        print("  ✓ Loads ground truth from manual annotations")
        print("  ✓ Evaluates original tracking (positions.json)")
        print("  ✓ Evaluates corrected tracking (corrected_positions.json)")
        print("  ✓ Calculates improvement metrics from jump correction")
        print("  ✓ Generates comparison plots and detailed reports")
        print("\nRequired files in results/[video_name]/ directory:")
        print("  - ground_truth_positions.json (or similar ground truth file)")
        print("  - positions.json (original tracking results)")
        print("  - corrected_positions.json (jump-corrected results)")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    try:
        evaluator = TrackingEvaluator(video_path)
        success = evaluator.run_evaluation()

        if not success:
            sys.exit(1)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()