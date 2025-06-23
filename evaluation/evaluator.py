#!/usr/bin/env python3
"""
Badminton Tracking Evaluation Script

Compares automated tracking results with manual ground truth annotations.
Generates comprehensive evaluation metrics for research paper publication.

Usage: python evaluate_tracking.py <video_file_path> [options]

Requirements:
    pip install numpy matplotlib pandas
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import time


class TrackingEvaluator:
    """Evaluates tracking performance against ground truth annotations."""

    def __init__(self, video_path: str, verbose: bool = True):
        """Initialize evaluator."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.verbose = verbose

        # Results directory
        self.results_dir = Path("results") / self.video_name

        # Data containers
        self.tracking_data = None
        self.ground_truth_data = None
        self.evaluation_results = {}

    def load_tracking_results(self) -> bool:
        """Load tracking results from the tracking scripts."""
        # Try to load corrected positions first (from jump correction script), then regular positions
        files_to_try = [
            ("corrected_positions.json", "jump-corrected tracking results"),
            ("positions.json", "basic tracking results")
        ]

        for filename, description in files_to_try:
            file_path = self.results_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        self.tracking_data = json.load(f)

                    if self.verbose:
                        print(f"✅ Loaded {description} from: {filename}")

                    # Store which file was used for reporting
                    self.tracking_data['_source_file'] = filename
                    return True

                except Exception as e:
                    if self.verbose:
                        print(f"❌ Error loading {filename}: {e}")

        if self.verbose:
            print(f"❌ No tracking results found in: {self.results_dir}")
        return False

    def load_ground_truth(self) -> bool:
        """Load ground truth annotations."""
        ground_truth_file = self.results_dir / "ground_truth_positions.json"

        if ground_truth_file.exists():
            try:
                with open(ground_truth_file, 'r') as f:
                    self.ground_truth_data = json.load(f)

                if self.verbose:
                    print(f"✅ Loaded ground truth from: ground_truth_positions.json")
                return True

            except Exception as e:
                if self.verbose:
                    print(f"❌ Error loading ground truth: {e}")
        else:
            if self.verbose:
                print(f"❌ Ground truth file not found: {ground_truth_file}")

        return False

    def calculate_tracking_errors(self) -> Dict[str, Any]:
        """Calculate comprehensive tracking errors between automated tracking and ground truth."""
        if not self.tracking_data or not self.ground_truth_data:
            return {"error": "Missing data"}

        # Create lookup tables
        tracking_lookup = {}
        for pos in self.tracking_data.get('player_positions', []):
            key = (pos['frame_index'], pos.get('player_id'))
            tracking_lookup[key] = (pos.get('hip_world_X', 0), pos.get('hip_world_Y', 0))

        gt_lookup = {}
        for pos in self.ground_truth_data.get('player_positions', []):
            key = (pos['frame_index'], pos.get('player_id'))
            gt_lookup[key] = (pos.get('hip_world_X', 0), pos.get('hip_world_Y', 0))

        # Calculate errors for common frames/players
        all_errors = []
        frame_errors = {}
        player_errors = {0: [], 1: []}
        detailed_errors = []

        for key in gt_lookup:
            if key in tracking_lookup:
                gt_x, gt_y = gt_lookup[key]
                track_x, track_y = tracking_lookup[key]

                # Euclidean distance error
                error = np.sqrt((gt_x - track_x)**2 + (gt_y - track_y)**2)
                all_errors.append(error)

                frame_idx, player_id = key
                frame_errors[frame_idx] = frame_errors.get(frame_idx, []) + [error]
                if player_id in player_errors:
                    player_errors[player_id].append(error)

                # Store detailed error info
                detailed_errors.append({
                    'frame_index': frame_idx,
                    'player_id': player_id,
                    'ground_truth': (gt_x, gt_y),
                    'tracking': (track_x, track_y),
                    'error_distance': error,
                    'error_x': abs(gt_x - track_x),
                    'error_y': abs(gt_y - track_y)
                })

        if not all_errors:
            return {"error": "No common frames found between tracking and ground truth"}

        # Calculate comprehensive statistics
        errors = np.array(all_errors)

        # Basic statistics
        result = {
            'total_comparisons': len(errors),
            'mean_error_meters': float(np.mean(errors)),
            'median_error_meters': float(np.median(errors)),
            'std_error_meters': float(np.std(errors)),
            'max_error_meters': float(np.max(errors)),
            'min_error_meters': float(np.min(errors)),
            'rmse_meters': float(np.sqrt(np.mean(errors**2))),

            # Error percentiles
            'error_percentiles': {
                '25th': float(np.percentile(errors, 25)),
                '50th': float(np.percentile(errors, 50)),
                '75th': float(np.percentile(errors, 75)),
                '90th': float(np.percentile(errors, 90)),
                '95th': float(np.percentile(errors, 95)),
                '99th': float(np.percentile(errors, 99))
            },

            # Player-specific performance
            'player_errors': {
                'player_0': {
                    'mean': float(np.mean(player_errors[0])) if player_errors[0] else 0,
                    'median': float(np.median(player_errors[0])) if player_errors[0] else 0,
                    'std': float(np.std(player_errors[0])) if player_errors[0] else 0,
                    'count': len(player_errors[0])
                },
                'player_1': {
                    'mean': float(np.mean(player_errors[1])) if player_errors[1] else 0,
                    'median': float(np.median(player_errors[1])) if player_errors[1] else 0,
                    'std': float(np.std(player_errors[1])) if player_errors[1] else 0,
                    'count': len(player_errors[1])
                }
            },

            # Accuracy at different thresholds
            'accuracy_thresholds': {
                '0.25m': {
                    'count': int(np.sum(errors < 0.25)),
                    'percentage': float(np.sum(errors < 0.25) / len(errors) * 100)
                },
                '0.5m': {
                    'count': int(np.sum(errors < 0.5)),
                    'percentage': float(np.sum(errors < 0.5) / len(errors) * 100)
                },
                '1.0m': {
                    'count': int(np.sum(errors < 1.0)),
                    'percentage': float(np.sum(errors < 1.0) / len(errors) * 100)
                },
                '1.5m': {
                    'count': int(np.sum(errors < 1.5)),
                    'percentage': float(np.sum(errors < 1.5) / len(errors) * 100)
                },
                '2.0m': {
                    'count': int(np.sum(errors < 2.0)),
                    'percentage': float(np.sum(errors < 2.0) / len(errors) * 100)
                }
            },

            # Coverage information
            'coverage': {
                'ground_truth_frames': len(set(pos['frame_index'] for pos in self.ground_truth_data.get('player_positions', []))),
                'tracking_frames': len(set(pos['frame_index'] for pos in self.tracking_data.get('player_positions', []))),
                'common_frames': len(set(pos['frame_index'] for pos in self.ground_truth_data.get('player_positions', [])) &
                                     set(pos['frame_index'] for pos in self.tracking_data.get('player_positions', []))),
                'coverage_percentage': float(len(errors) / len(set(pos['frame_index'] for pos in self.ground_truth_data.get('player_positions', []))) * 100) if self.ground_truth_data.get('player_positions') else 0
            },

            # Detailed error data for further analysis
            'detailed_errors': detailed_errors
        }

        return result

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results or 'error' in self.evaluation_results:
            error_msg = self.evaluation_results.get('error', 'Unknown error')
            return f"Error: {error_msg}"

        results = self.evaluation_results

        # Determine tracking method
        tracking_method = "Unknown"
        source_file = self.tracking_data.get('_source_file', 'unknown')
        jump_corrected = source_file == 'corrected_positions.json'

        if 'processing_info' in self.tracking_data:
            tracking_method = self.tracking_data['processing_info'].get('tracking_method', tracking_method)

        # Generate comprehensive report
        report = f"""
BADMINTON TRACKING EVALUATION REPORT
====================================

Video: {self.video_name}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Tracking Method: {tracking_method}
Source File: {source_file}
Jump Correction Applied: {"Yes" if jump_corrected else "No"}

EXECUTIVE SUMMARY
----------------
Total Comparisons: {results['total_comparisons']}
Mean Absolute Error (MAE): {results['mean_error_meters']:.3f} meters
Root Mean Square Error (RMSE): {results['rmse_meters']:.3f} meters
Median Error: {results['median_error_meters']:.3f} meters
95th Percentile Error: {results['error_percentiles']['95th']:.3f} meters
Precision at 1m: {results['accuracy_thresholds']['1.0m']['percentage']:.1f}%

DETAILED PERFORMANCE METRICS
----------------------------
Mean error: {results['mean_error_meters']:.3f} meters
Median error: {results['median_error_meters']:.3f} meters
Standard deviation: {results['std_error_meters']:.3f} meters
Min error: {results['min_error_meters']:.3f} meters
Max error: {results['max_error_meters']:.3f} meters
RMSE: {results['rmse_meters']:.3f} meters

ERROR DISTRIBUTION (Percentiles)
-------------------------------
25th percentile: {results['error_percentiles']['25th']:.3f} meters
50th percentile: {results['error_percentiles']['50th']:.3f} meters
75th percentile: {results['error_percentiles']['75th']:.3f} meters
90th percentile: {results['error_percentiles']['90th']:.3f} meters
95th percentile: {results['error_percentiles']['95th']:.3f} meters
99th percentile: {results['error_percentiles']['99th']:.3f} meters

ACCURACY AT DIFFERENT THRESHOLDS
--------------------------------
Errors < 0.25m: {results['accuracy_thresholds']['0.25m']['count']} ({results['accuracy_thresholds']['0.25m']['percentage']:.1f}%)
Errors < 0.5m:  {results['accuracy_thresholds']['0.5m']['count']} ({results['accuracy_thresholds']['0.5m']['percentage']:.1f}%)
Errors < 1.0m:  {results['accuracy_thresholds']['1.0m']['count']} ({results['accuracy_thresholds']['1.0m']['percentage']:.1f}%)
Errors < 1.5m:  {results['accuracy_thresholds']['1.5m']['count']} ({results['accuracy_thresholds']['1.5m']['percentage']:.1f}%)
Errors < 2.0m:  {results['accuracy_thresholds']['2.0m']['count']} ({results['accuracy_thresholds']['2.0m']['percentage']:.1f}%)

PLAYER-SPECIFIC PERFORMANCE
---------------------------
Player 0: {results['player_errors']['player_0']['mean']:.3f}m ± {results['player_errors']['player_0']['std']:.3f}m (n={results['player_errors']['player_0']['count']})
Player 1: {results['player_errors']['player_1']['mean']:.3f}m ± {results['player_errors']['player_1']['std']:.3f}m (n={results['player_errors']['player_1']['count']})

DATASET COVERAGE
---------------
Ground truth frames: {results['coverage']['ground_truth_frames']}
Tracking frames: {results['coverage']['tracking_frames']}
Common frames: {results['coverage']['common_frames']}
Coverage: {results['coverage']['coverage_percentage']:.1f}%

RESEARCH PAPER METRICS (Key Values for Publication)
--------------------------------------------------
Mean Absolute Error (MAE): {results['mean_error_meters']:.3f} ± {results['std_error_meters']:.3f} meters
Root Mean Square Error (RMSE): {results['rmse_meters']:.3f} meters
95th Percentile Error: {results['error_percentiles']['95th']:.3f} meters
Precision at 1m threshold: {results['accuracy_thresholds']['1.0m']['percentage']:.1f}%
Median Error: {results['median_error_meters']:.3f} meters

PERFORMANCE GRADE
----------------"""

        # Add performance grade based on mean error
        mean_error = results['mean_error_meters']
        if mean_error < 0.3:
            grade = "EXCELLENT"
        elif mean_error < 0.5:
            grade = "VERY GOOD"
        elif mean_error < 0.8:
            grade = "GOOD"
        elif mean_error < 1.2:
            grade = "FAIR"
        else:
            grade = "NEEDS IMPROVEMENT"

        report += f"\nOverall Performance: {grade} (MAE: {mean_error:.3f}m)\n"

        return report

    def create_visualization_plots(self, save_dir: Optional[Path] = None) -> None:
        """Create visualization plots for the evaluation."""
        if not self.evaluation_results or 'detailed_errors' not in self.evaluation_results:
            print("No detailed error data available for visualization")
            return

        if save_dir is None:
            save_dir = self.results_dir

        detailed_errors = self.evaluation_results['detailed_errors']
        errors = [e['error_distance'] for e in detailed_errors]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Tracking Evaluation Results - {self.video_name}', fontsize=16)

        # 1. Error distribution histogram
        ax1.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Error Distance (meters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}m')
        ax1.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.3f}m')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Error over time (frame index)
        frames = [e['frame_index'] for e in detailed_errors]
        ax2.scatter(frames, errors, alpha=0.6, s=20)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Error Distance (meters)')
        ax2.set_title('Error Over Time')
        ax2.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(frames, errors, 1)
        p = np.poly1d(z)
        ax2.plot(frames, p(frames), "r--", alpha=0.8, label=f'Trend: {z[0]:.6f}x + {z[1]:.3f}')
        ax2.legend()

        # 3. Player comparison
        player_0_errors = [e['error_distance'] for e in detailed_errors if e['player_id'] == 0]
        player_1_errors = [e['error_distance'] for e in detailed_errors if e['player_id'] == 1]

        player_data = []
        labels = []
        if player_0_errors:
            player_data.append(player_0_errors)
            labels.append(f'Player 0 (n={len(player_0_errors)})')
        if player_1_errors:
            player_data.append(player_1_errors)
            labels.append(f'Player 1 (n={len(player_1_errors)})')

        if player_data:
            ax3.boxplot(player_data, labels=labels)
            ax3.set_ylabel('Error Distance (meters)')
            ax3.set_title('Player-Specific Error Distribution')
            ax3.grid(True, alpha=0.3)

        # 4. Cumulative error distribution
        sorted_errors = np.sort(errors)
        percentiles = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        ax4.plot(sorted_errors, percentiles, linewidth=2)
        ax4.set_xlabel('Error Distance (meters)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('Cumulative Error Distribution')
        ax4.grid(True, alpha=0.3)

        # Add threshold lines
        thresholds = [0.5, 1.0, 1.5, 2.0]
        for threshold in thresholds:
            if threshold <= max(sorted_errors):
                pct = np.sum(errors < threshold) / len(errors) * 100
                ax4.axvline(threshold, color='red', linestyle=':', alpha=0.7)
                ax4.text(threshold, pct + 5, f'{threshold}m\n({pct:.1f}%)',
                         ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save plot
        plot_file = save_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')

        if self.verbose:
            print(f"✅ Visualization plots saved to: {plot_file}")

        plt.close()

    def save_detailed_results(self, save_dir: Optional[Path] = None) -> None:
        """Save detailed evaluation results to JSON file."""
        if save_dir is None:
            save_dir = self.results_dir

        # Prepare comprehensive results
        detailed_results = {
            'video_info': {
                'video_path': str(self.video_path),
                'video_name': self.video_name,
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'tracking_info': {
                'source_file': self.tracking_data.get('_source_file', 'unknown'),
                'jump_correction_applied': self.tracking_data.get('_source_file') == 'corrected_positions.json',
                'tracking_method': self.tracking_data.get('processing_info', {}).get('tracking_method', 'unknown')
            },
            'evaluation_results': self.evaluation_results
        }

        # Save to file
        results_file = save_dir / "detailed_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        if self.verbose:
            print(f"✅ Detailed results saved to: {results_file}")

    def run_evaluation(self, create_plots: bool = True, save_results: bool = True) -> bool:
        """Run complete evaluation pipeline."""
        if self.verbose:
            print(f"Starting evaluation for: {self.video_name}")
            print("=" * 50)

        # Load data
        if not self.load_tracking_results():
            return False

        if not self.load_ground_truth():
            return False

        # Calculate errors
        if self.verbose:
            print("Calculating tracking errors...")

        self.evaluation_results = self.calculate_tracking_errors()

        if 'error' in self.evaluation_results:
            if self.verbose:
                print(f"❌ Evaluation failed: {self.evaluation_results['error']}")
            return False

        # Generate and display report
        report = self.generate_evaluation_report()
        print(report)

        # Save report to file
        report_file = self.results_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        if self.verbose:
            print(f"\n✅ Report saved to: {report_file}")

        # Create visualizations
        if create_plots:
            try:
                self.create_visualization_plots()
            except ImportError:
                if self.verbose:
                    print("⚠️  Matplotlib not available - skipping plots")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error creating plots: {e}")

        # Save detailed results
        if save_results:
            self.save_detailed_results()

        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate badminton player tracking performance')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating visualization plots')
    parser.add_argument('--no-save', action='store_true', help='Skip saving detailed results')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Create evaluator
    evaluator = TrackingEvaluator(args.video_path, verbose=not args.quiet)

    # Run evaluation
    success = evaluator.run_evaluation(
        create_plots=not args.no_plots,
        save_results=not args.no_save
    )

    if not success:
        print("\n❌ Evaluation failed. Please check that you have:")
        print("1. Run the tracking scripts to generate position data")
        print("2. Created ground truth annotations using the annotation tool")
        sys.exit(1)

    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()