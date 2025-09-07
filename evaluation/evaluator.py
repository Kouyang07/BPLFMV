#!/usr/bin/env python3
"""
Enhanced Badminton Tracking Evaluator with Two-Player Support and Jump Correction Comparison

Evaluates tracking performance against ground truth for both players and compares the improvement
from jump correction by analyzing both positions.json and corrected_positions.json.

Usage: python3 evaluator.py <video_file_path> [--player <1|2|both>] [--closest-only]

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
import argparse


class TrackingEvaluator:
    """Enhanced evaluator that compares original and corrected tracking results for both players."""

    def __init__(self, video_path: str, target_player: str = "closest", closest_only: bool = False):
        """Initialize the evaluator.

        Args:
            video_path: Path to the video file
            target_player: Which player to evaluate ("1", "2", "both", or "closest")
            closest_only: If True, only evaluate the closest player to camera
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path(self.video_name)  # Changed from Path("results") / self.video_name

        self.target_player = target_player
        self.closest_only = closest_only

        # Data containers - now support multiple players
        self.ground_truth = {}  # frame -> {player_id: (x, y)} or frame -> (x, y) for single player
        self.original_positions = {}  # frame -> [(player_data), ...]
        self.corrected_positions = {}  # frame -> [(player_data), ...]

        # Evaluation results - now per player
        self.evaluation_results = {
            'player_1': {'original_errors': [], 'corrected_errors': [], 'frame_comparisons': []},
            'player_2': {'original_errors': [], 'corrected_errors': [], 'frame_comparisons': []},
            'closest': {'original_errors': [], 'corrected_errors': [], 'frame_comparisons': []}
        }

    def load_ground_truth(self) -> bool:
        """Load ground truth data with support for both single and multi-player formats."""
        gt_files = [
            "video_click_ground_truth.json",  # New two-player format
            "ground_truth_positions.json",   # Legacy format
            "manual_annotations.json"        # Alternative format
        ]

        for filename in gt_files:
            gt_file = self.results_dir / filename
            if gt_file.exists():
                try:
                    with open(gt_file, 'r') as f:
                        data = json.load(f)

                    positions = data.get('player_positions', [])

                    # Check if this is the new multi-player format
                    has_player_ids = any('player_id' in pos for pos in positions)

                    if has_player_ids:
                        # New multi-player format
                        for pos in positions:
                            frame_idx = pos['frame_index']
                            player_id = pos['player_id']

                            if frame_idx not in self.ground_truth:
                                self.ground_truth[frame_idx] = {}

                            # Use hip position as ground truth
                            self.ground_truth[frame_idx][player_id] = (
                                pos['hip_world_X'],
                                pos['hip_world_Y']
                            )

                        print(f"✅ Loaded multi-player ground truth from: {filename}")
                        player_counts = {}
                        for frame_data in self.ground_truth.values():
                            for player_id in frame_data.keys():
                                player_counts[player_id] = player_counts.get(player_id, 0) + 1

                        for player_id, count in player_counts.items():
                            print(f"   Player {player_id}: {count} annotated frames")

                    else:
                        # Legacy single-player format
                        for pos in positions:
                            frame_idx = pos['frame_index']
                            # Store as single position (backward compatibility)
                            self.ground_truth[frame_idx] = (
                                pos['hip_world_X'],
                                pos['hip_world_Y']
                            )

                        print(f"✅ Loaded single-player ground truth from: {filename}")
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

    def find_player_by_id(self, positions, player_id):
        """Find player by specific ID."""
        for pos in positions:
            if pos.get('player_id') == player_id:
                return pos
        return None

    def calculate_error(self, predicted_pos, ground_truth_pos):
        """Calculate Euclidean distance error."""
        pred_x, pred_y = predicted_pos
        gt_x, gt_y = ground_truth_pos
        return math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

    def evaluate_tracking_data(self, positions_dict, data_type="tracking"):
        """Evaluate tracking data against ground truth for all target players."""
        results = {
            'player_1': {'errors': [], 'frame_results': []},
            'player_2': {'errors': [], 'frame_results': []},
            'closest': {'errors': [], 'frame_results': []}
        }

        for frame_idx, gt_data in self.ground_truth.items():
            if frame_idx not in positions_dict:
                continue

            positions = positions_dict[frame_idx]

            # Handle both legacy (single position) and new (multi-player) ground truth formats
            if isinstance(gt_data, tuple):
                # Legacy format - single position
                gt_pos = gt_data

                if self.target_player in ["closest", "both"]:
                    closest_player = self.find_closest_player(positions)
                    if closest_player:
                        predicted_pos = self.calculate_midpoint_position(closest_player)
                        error = self.calculate_error(predicted_pos, gt_pos)
                        results['closest']['errors'].append(error)
                        results['closest']['frame_results'].append({
                            'frame_index': frame_idx,
                            'ground_truth': gt_pos,
                            'predicted': predicted_pos,
                            'error': error,
                            'data_type': data_type,
                            'player_id': 'closest'
                        })

            else:
                # New multi-player format
                gt_positions = gt_data

                # Evaluate each player separately
                for target_player_id in [1, 2]:
                    if (self.target_player in [str(target_player_id), "both"] and
                            target_player_id in gt_positions):

                        gt_pos = gt_positions[target_player_id]
                        predicted_player = self.find_player_by_id(positions, target_player_id)

                        if predicted_player:
                            predicted_pos = self.calculate_midpoint_position(predicted_player)
                            error = self.calculate_error(predicted_pos, gt_pos)

                            player_key = f'player_{target_player_id}'
                            results[player_key]['errors'].append(error)
                            results[player_key]['frame_results'].append({
                                'frame_index': frame_idx,
                                'ground_truth': gt_pos,
                                'predicted': predicted_pos,
                                'error': error,
                                'data_type': data_type,
                                'player_id': target_player_id
                            })

                # Also evaluate closest player if requested
                if self.target_player in ["closest", "both"]:
                    closest_player = self.find_closest_player(positions)
                    if closest_player:
                        closest_id = closest_player.get('player_id', 1)
                        if closest_id in gt_positions:
                            gt_pos = gt_positions[closest_id]
                            predicted_pos = self.calculate_midpoint_position(closest_player)
                            error = self.calculate_error(predicted_pos, gt_pos)

                            results['closest']['errors'].append(error)
                            results['closest']['frame_results'].append({
                                'frame_index': frame_idx,
                                'ground_truth': gt_pos,
                                'predicted': predicted_pos,
                                'error': error,
                                'data_type': data_type,
                                'player_id': f'closest(P{closest_id})'
                            })

        return results

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

    def calculate_improvement_metrics(self, player_key):
        """Calculate improvement metrics for a specific player."""
        original_errors = self.evaluation_results[player_key]['original_errors']
        corrected_errors = self.evaluation_results[player_key]['corrected_errors']

        if not original_errors or not corrected_errors:
            return {}

        # Find common frames
        original_frames = {comp['frame_index'] for comp in self.evaluation_results[player_key]['frame_comparisons']
                           if comp['data_type'] == 'original'}
        corrected_frames = {comp['frame_index'] for comp in self.evaluation_results[player_key]['frame_comparisons']
                            if comp['data_type'] == 'corrected'}
        common_frames = original_frames.intersection(corrected_frames)

        if not common_frames:
            return {}

        # Get errors for common frames only
        original_dict = {comp['frame_index']: comp['error']
                         for comp in self.evaluation_results[player_key]['frame_comparisons']
                         if comp['data_type'] == 'original'}
        corrected_dict = {comp['frame_index']: comp['error']
                          for comp in self.evaluation_results[player_key]['frame_comparisons']
                          if comp['data_type'] == 'corrected'}

        # Calculate frame-by-frame improvements
        frame_improvements = []
        original_common = []
        corrected_common = []

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

                original_common.append(orig_error)
                corrected_common.append(corr_error)

        # Calculate overall improvement statistics
        improvements = [f['absolute_improvement'] for f in frame_improvements]

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
        """Create comparison plots for all evaluated players."""
        try:
            # Count how many players have data
            players_with_data = []
            for player_key in ['player_1', 'player_2', 'closest']:
                if (self.evaluation_results[player_key]['original_errors'] or
                        self.evaluation_results[player_key]['corrected_errors']):
                    players_with_data.append(player_key)

            if not players_with_data:
                print("⚠️  No data available for plotting")
                return

            # Create plots for each player with data
            fig_height = 5 * len(players_with_data)
            fig, axes = plt.subplots(len(players_with_data), 2, figsize=(12, fig_height))

            if len(players_with_data) == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f'Tracking Performance Comparison - {self.video_name}', fontsize=14, fontweight='bold')

            for idx, player_key in enumerate(players_with_data):
                original_errors = self.evaluation_results[player_key]['original_errors']
                corrected_errors = self.evaluation_results[player_key]['corrected_errors']

                player_title = player_key.replace('_', ' ').title()

                # Plot 1: Error distribution comparison
                if original_errors and corrected_errors:
                    axes[idx, 0].hist(original_errors, bins=30, alpha=0.7, label='Original', color='red', density=True)
                    axes[idx, 0].hist(corrected_errors, bins=30, alpha=0.7, label='Corrected', color='blue', density=True)
                    axes[idx, 0].set_xlabel('Error (meters)')
                    axes[idx, 0].set_ylabel('Density')
                    axes[idx, 0].set_title(f'{player_title} - Error Distribution')
                    axes[idx, 0].legend()
                    axes[idx, 0].grid(True, alpha=0.3)

                # Plot 2: Box plot comparison
                if original_errors and corrected_errors:
                    data_to_plot = [original_errors, corrected_errors]
                    labels = ['Original', 'Corrected']
                    bp = axes[idx, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
                    bp['boxes'][0].set_facecolor('red')
                    bp['boxes'][1].set_facecolor('blue')
                    axes[idx, 1].set_ylabel('Error (meters)')
                    axes[idx, 1].set_title(f'{player_title} - Error Statistics')
                    axes[idx, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = self.results_dir / "comparison_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plots saved to: {plot_file}")

            plt.close()

        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")

    def generate_comparison_report(self, all_metrics, all_improvements):
        """Generate comprehensive comparison report for all players."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = []
        report.append("BADMINTON TRACKING COMPARISON REPORT - TWO PLAYER SUPPORT")
        report.append("=" * 65)
        report.append(f"Video: {self.video_name}")
        report.append(f"Date: {timestamp}")
        report.append(f"Evaluation Target: {self.target_player}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)

        for player_key in ['player_1', 'player_2', 'closest']:
            if player_key in all_metrics['original'] and player_key in all_metrics['corrected']:
                original_metrics = all_metrics['original'][player_key]
                corrected_metrics = all_metrics['corrected'][player_key]
                improvement_metrics = all_improvements.get(player_key, {})

                if original_metrics and corrected_metrics:
                    player_title = player_key.replace('_', ' ').title()
                    report.append(f"\n{player_title}:")
                    report.append(f"  Original Mean Error: {original_metrics['mean_error']:.3f} meters")
                    report.append(f"  Corrected Mean Error: {corrected_metrics['mean_error']:.3f} meters")

                    if improvement_metrics:
                        report.append(f"  Absolute Improvement: {improvement_metrics['absolute_improvement']:.3f} meters")
                        report.append(f"  Relative Improvement: {improvement_metrics['percent_improvement']:.1f}%")
                        report.append(f"  Frames Improved: {improvement_metrics['improved_frames']}/{improvement_metrics['common_frames']} ({improvement_metrics['improved_frames']/improvement_metrics['common_frames']*100:.1f}%)")

        report.append("")

        # Detailed results for each player
        for player_key in ['player_1', 'player_2', 'closest']:
            if player_key in all_metrics['original'] or player_key in all_metrics['corrected']:
                player_title = player_key.replace('_', ' ').title()
                report.append(f"{player_title.upper()} DETAILED RESULTS")
                report.append("=" * (len(player_title) + 17))

                # Original Performance
                if player_key in all_metrics['original'] and all_metrics['original'][player_key]:
                    original_metrics = all_metrics['original'][player_key]
                    report.append("Original Tracking Performance:")
                    report.append(f"  Total Comparisons: {original_metrics['total_comparisons']}")
                    report.append(f"  Mean Absolute Error (MAE): {original_metrics['mean_error']:.3f} meters")
                    report.append(f"  Root Mean Square Error (RMSE): {original_metrics['rmse']:.3f} meters")
                    report.append(f"  Median Error: {original_metrics['median_error']:.3f} meters")
                    report.append(f"  95th Percentile Error: {original_metrics['percentiles']['95th']:.3f} meters")
                    report.append(f"  Precision at 1m: {original_metrics['accuracy_thresholds']['1.0m']/original_metrics['total_comparisons']*100:.1f}%")
                    report.append("")

                # Corrected Performance
                if player_key in all_metrics['corrected'] and all_metrics['corrected'][player_key]:
                    corrected_metrics = all_metrics['corrected'][player_key]
                    report.append("Corrected Tracking Performance:")
                    report.append(f"  Total Comparisons: {corrected_metrics['total_comparisons']}")
                    report.append(f"  Mean Absolute Error (MAE): {corrected_metrics['mean_error']:.3f} meters")
                    report.append(f"  Root Mean Square Error (RMSE): {corrected_metrics['rmse']:.3f} meters")
                    report.append(f"  Median Error: {corrected_metrics['median_error']:.3f} meters")
                    report.append(f"  95th Percentile Error: {corrected_metrics['percentiles']['95th']:.3f} meters")
                    report.append(f"  Precision at 1m: {corrected_metrics['accuracy_thresholds']['1.0m']/corrected_metrics['total_comparisons']*100:.1f}%")
                    report.append("")

                # Improvement Analysis
                if player_key in all_improvements and all_improvements[player_key]:
                    improvement_metrics = all_improvements[player_key]
                    report.append("Jump Correction Improvement Analysis:")
                    report.append(f"  Common Frames Analyzed: {improvement_metrics['common_frames']}")
                    report.append(f"  Mean Error Reduction: {improvement_metrics['absolute_improvement']:.3f} meters ({improvement_metrics['percent_improvement']:.1f}%)")
                    report.append(f"  Improved frames: {improvement_metrics['improved_frames']} ({improvement_metrics['improved_frames']/improvement_metrics['common_frames']*100:.1f}%)")
                    report.append(f"  Degraded frames: {improvement_metrics['degraded_frames']} ({improvement_metrics['degraded_frames']/improvement_metrics['common_frames']*100:.1f}%)")
                    report.append("")

        return "\n".join(report)

    def save_detailed_results(self, all_metrics, all_improvements):
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
                'evaluator_version': '3.0_two_player',
                'target_player': self.target_player,
                'closest_only': self.closest_only
            },
            'summary': {},
            'detailed_metrics': all_metrics,
            'improvement_analysis': all_improvements
        }

        # Create summary for each player
        for player_key in ['player_1', 'player_2', 'closest']:
            if (player_key in all_metrics.get('original', {}) and
                    player_key in all_metrics.get('corrected', {})):

                original_metrics = all_metrics['original'][player_key]
                corrected_metrics = all_metrics['corrected'][player_key]
                improvement_metrics = all_improvements.get(player_key, {})

                detailed_results['summary'][player_key] = {
                    'original_mean_error': float(original_metrics.get('mean_error', 0)) if original_metrics else None,
                    'corrected_mean_error': float(corrected_metrics.get('mean_error', 0)) if corrected_metrics else None,
                    'absolute_improvement': float(improvement_metrics.get('absolute_improvement', 0)) if improvement_metrics else None,
                    'percent_improvement': float(improvement_metrics.get('percent_improvement', 0)) if improvement_metrics else None,
                    'frames_improved': int(improvement_metrics.get('improved_frames', 0)) if improvement_metrics else None
                }

        # Convert all numpy types to native Python types
        detailed_results = convert_numpy_types(detailed_results)

        results_file = self.results_dir / "comparison_summary.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"✅ Comparison summary saved to: {results_file}")

    def run_evaluation(self):
        """Run the complete evaluation."""
        print(f"Starting two-player comparison evaluation for: {self.video_name}")
        print(f"Target player(s): {self.target_player}")
        print("=" * 70)

        # Load all data
        if not self.load_ground_truth():
            return False

        original_loaded, corrected_loaded = self.load_tracking_data()

        if not original_loaded and not corrected_loaded:
            print("❌ No tracking data found!")
            return False

        print("\nCalculating tracking errors...")

        # Evaluate original tracking
        all_metrics = {'original': {}, 'corrected': {}}
        all_improvements = {}

        if original_loaded:
            original_results = self.evaluate_tracking_data(self.original_positions, "original")

            for player_key, player_data in original_results.items():
                if player_data['errors']:
                    self.evaluation_results[player_key]['original_errors'] = player_data['errors']
                    self.evaluation_results[player_key]['frame_comparisons'].extend(player_data['frame_results'])
                    all_metrics['original'][player_key] = self.calculate_metrics(player_data['errors'])
                    print(f"✅ Original tracking evaluation complete for {player_key}: {len(player_data['errors'])} comparisons")

        # Evaluate corrected tracking
        if corrected_loaded:
            corrected_results = self.evaluate_tracking_data(self.corrected_positions, "corrected")

            for player_key, player_data in corrected_results.items():
                if player_data['errors']:
                    self.evaluation_results[player_key]['corrected_errors'] = player_data['errors']
                    self.evaluation_results[player_key]['frame_comparisons'].extend(player_data['frame_results'])
                    all_metrics['corrected'][player_key] = self.calculate_metrics(player_data['errors'])
                    print(f"✅ Corrected tracking evaluation complete for {player_key}: {len(player_data['errors'])} comparisons")

        # Calculate improvement metrics for each player
        if original_loaded and corrected_loaded:
            for player_key in ['player_1', 'player_2', 'closest']:
                if (self.evaluation_results[player_key]['original_errors'] and
                        self.evaluation_results[player_key]['corrected_errors']):
                    all_improvements[player_key] = self.calculate_improvement_metrics(player_key)
                    print(f"✅ Improvement analysis complete for {player_key}: {all_improvements[player_key].get('common_frames', 0)} frames compared")

        # Generate report
        report = self.generate_comparison_report(all_metrics, all_improvements)
        print("\n" + report)

        # Save report
        report_file = self.results_dir / "tracking_comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✅ Comparison report saved to: {report_file}")

        # Create plots
        self.create_comparison_plots()

        # Save detailed results
        self.save_detailed_results(all_metrics, all_improvements)

        print("✅ Evaluation completed successfully!")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Badminton Tracking Evaluator with Two-Player Support')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--player', choices=['1', '2', 'both', 'closest'], default='closest',
                        help='Which player to evaluate (default: closest)')
    parser.add_argument('--closest-only', action='store_true',
                        help='Only evaluate the closest player to camera')

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nBadminton Tracking Evaluator with Two-Player Support and Jump Correction Comparison")
        print("=" * 85)
        print("This evaluator compares original and corrected tracking performance for both players:")
        print("  ✓ Loads ground truth from two-player manual annotations")
        print("  ✓ Evaluates original tracking (positions.json)")
        print("  ✓ Evaluates corrected tracking (corrected_positions.json)")
        print("  ✓ Supports evaluation of Player 1, Player 2, or both")
        print("  ✓ Calculates improvement metrics from jump correction")
        print("  ✓ Generates comparison plots and detailed reports")
        print("  ✓ Works with new directory structure: ./[video_name]/")
        print("\nRequired files in [video_name]/ directory:")
        print("  - video_click_ground_truth.json (two-player ground truth file)")
        print("  - positions.json (original tracking results)")
        print("  - corrected_positions.json (jump-corrected results)")
        print("\nUsage examples:")
        print("  python3 evaluator.py video.mp4                    # Evaluate closest player")
        print("  python3 evaluator.py video.mp4 --player 1         # Evaluate Player 1 only")
        print("  python3 evaluator.py video.mp4 --player both      # Evaluate both players")
        print("  python3 evaluator.py video.mp4 --closest-only     # Force closest player mode")
        sys.exit(0)

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        evaluator = TrackingEvaluator(args.video_path, args.player, args.closest_only)
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