#!/usr/bin/env python3
"""
CVAT-Based Badminton Tracking Evaluator with Automatic Player Mapping

Evaluates tracking performance against CVAT ground truth annotations converted to world coordinates.
Generates research paper style error analysis with comprehensive statistical evaluation.
Automatically resolves player ID mapping issues by testing both configurations.

Usage: python3 evaluation/evaluator.py <video_path>

Requirements:
    pip install numpy matplotlib scipy seaborn
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import math
import argparse
import warnings

# Handle optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings('ignore')


class CVATTrackingEvaluator:
    """Evaluator for badminton tracking against CVAT ground truth with research-grade analysis."""

    def __init__(self, video_path: str):
        """Initialize the evaluator.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem

        # Updated directory structure
        self.results_dir = Path("results") / self.video_name

        # Data containers
        self.ground_truth = {}  # frame -> {player_id: (x, y)}
        self.original_positions = {}  # frame -> [player_data, ...]
        self.corrected_positions = {}  # frame -> [player_data, ...]

        # Evaluation results per player
        self.evaluation_results = {
            'player_0': {'original_errors': [], 'corrected_errors': [], 'frame_comparisons': []},
            'player_1': {'original_errors': [], 'corrected_errors': [], 'frame_comparisons': []}
        }

    def load_cvat_ground_truth(self) -> bool:
        """Load CVAT ground truth data from cvat_ground_truth.json."""
        gt_files = [
            "cvat_ground_truth.json",
            "cvat_annotations.json"
        ]

        for filename in gt_files:
            gt_file = self.results_dir / filename
            if gt_file.exists():
                try:
                    with open(gt_file, 'r') as f:
                        data = json.load(f)

                    # Extract player positions from CVAT format
                    positions = data.get('player_positions', [])

                    for pos in positions:
                        frame_idx = pos['frame_index']
                        player_id = pos['player_id']
                        world_x = pos['world_x']
                        world_y = pos['world_y']

                        if frame_idx not in self.ground_truth:
                            self.ground_truth[frame_idx] = {}

                        self.ground_truth[frame_idx][player_id] = (world_x, world_y)

                    print(f"Loaded CVAT ground truth from: {filename}")

                    # Print summary statistics
                    player_counts = {}
                    for frame_data in self.ground_truth.values():
                        for player_id in frame_data.keys():
                            player_counts[player_id] = player_counts.get(player_id, 0) + 1

                    total_frames = len(self.ground_truth)
                    print(f"   Total annotated frames: {total_frames}")
                    for player_id, count in player_counts.items():
                        print(f"   Player {player_id}: {count} annotations ({count/total_frames*100:.1f}% coverage)")

                    return True

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

        print(f"No CVAT ground truth file found in {self.results_dir}")
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

                # Handle the nested frame_data structure
                frame_data = data.get('frame_data', {})
                for frame_str, players_data in frame_data.items():
                    frame_idx = int(frame_str)

                    if frame_idx not in self.original_positions:
                        self.original_positions[frame_idx] = []

                    # Convert the nested player data to the expected format
                    for player_key, player_data in players_data.items():
                        # Extract player_id from key (player_0 -> 0, player_1 -> 1)
                        player_id = int(player_key.split('_')[1])

                        # Create position entry using center_position
                        position_entry = {
                            'frame_index': frame_idx,
                            'player_id': player_id,
                            'hip_world_X': player_data['center_position']['x'],
                            'hip_world_Y': player_data['center_position']['y'],
                            'method': player_data.get('ankles', [{}])[0].get('method', 'unknown')
                        }

                        # Add ankle data if available
                        ankles = player_data.get('ankles', [])
                        for ankle in ankles:
                            if ankle.get('ankle_side') == 'left':
                                position_entry['left_ankle_world_X'] = ankle['world_x']
                                position_entry['left_ankle_world_Y'] = ankle['world_y']
                            elif ankle.get('ankle_side') == 'right':
                                position_entry['right_ankle_world_X'] = ankle['world_x']
                                position_entry['right_ankle_world_Y'] = ankle['world_y']

                        self.original_positions[frame_idx].append(position_entry)

                print(f"Loaded original tracking: {len(self.original_positions)} frames")
                original_loaded = True

            except Exception as e:
                print(f"Error loading positions.json: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Original positions file not found: positions.json")

        # Load corrected positions
        corrected_file = self.results_dir / "corrected_positions.json"
        if corrected_file.exists():
            try:
                with open(corrected_file, 'r') as f:
                    data = json.load(f)

                # Handle the nested frame_data structure for corrected data
                frame_data = data.get('frame_data', {})
                for frame_str, players_data in frame_data.items():
                    frame_idx = int(frame_str)

                    if frame_idx not in self.corrected_positions:
                        self.corrected_positions[frame_idx] = []

                    # Convert the nested player data to the expected format
                    for player_key, player_data in players_data.items():
                        # Extract player_id from key (player_0 -> 0, player_1 -> 1)
                        player_id = int(player_key.split('_')[1])

                        # Create position entry using center_position
                        position_entry = {
                            'frame_index': frame_idx,
                            'player_id': player_id,
                            'hip_world_X': player_data['center_position']['x'],
                            'hip_world_Y': player_data['center_position']['y'],
                            'method': player_data.get('ankles', [{}])[0].get('method', 'unknown')
                        }

                        # Add ankle data if available
                        ankles = player_data.get('ankles', [])
                        for ankle in ankles:
                            if ankle.get('ankle_side') == 'left':
                                position_entry['left_ankle_world_X'] = ankle['world_x']
                                position_entry['left_ankle_world_Y'] = ankle['world_y']
                            elif ankle.get('ankle_side') == 'right':
                                position_entry['right_ankle_world_X'] = ankle['world_x']
                                position_entry['right_ankle_world_Y'] = ankle['world_y']

                        self.corrected_positions[frame_idx].append(position_entry)

                print(f"Loaded corrected tracking: {len(self.corrected_positions)} frames")
                corrected_loaded = True

            except Exception as e:
                print(f"Error loading corrected_positions.json: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Corrected positions file not found: corrected_positions.json")

        return original_loaded, corrected_loaded

    def calculate_midpoint_position(self, position_data: Dict) -> Tuple[float, float]:
        """Calculate representative position from tracking data."""
        # Use hip position as primary, fall back to midpoint if needed
        if 'hip_world_X' in position_data and 'hip_world_Y' in position_data:
            return position_data['hip_world_X'], position_data['hip_world_Y']

        # Fallback: calculate midpoint of available joints
        x_coords = []
        y_coords = []

        joint_pairs = [
            ('hip_world_X', 'hip_world_Y'),
            ('left_ankle_world_X', 'left_ankle_world_Y'),
            ('right_ankle_world_X', 'right_ankle_world_Y')
        ]

        for x_key, y_key in joint_pairs:
            if x_key in position_data and y_key in position_data:
                x_coords.append(position_data[x_key])
                y_coords.append(position_data[y_key])

        if x_coords and y_coords:
            return np.mean(x_coords), np.mean(y_coords)

        raise ValueError("No valid position data found")

    def find_player_by_id(self, positions: List[Dict], player_id: int) -> Optional[Dict]:
        """Find player by specific ID."""
        for pos in positions:
            if pos.get('player_id') == player_id:
                return pos
        return None

    def calculate_euclidean_error(self, predicted_pos: Tuple[float, float],
                                  ground_truth_pos: Tuple[float, float]) -> float:
        """Calculate Euclidean distance error."""
        pred_x, pred_y = predicted_pos
        gt_x, gt_y = ground_truth_pos
        return math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

    def evaluate_with_player_mapping(self, positions_dict: Dict, data_type: str,
                                     mapping: Dict[int, int]) -> Tuple[Dict, float]:
        """Evaluate tracking data with a specific player ID mapping."""
        results = {
            'player_0': {'errors': [], 'frame_results': []},
            'player_1': {'errors': [], 'frame_results': []}
        }

        total_error = 0.0
        comparison_count = 0

        for frame_idx, gt_data in self.ground_truth.items():
            if frame_idx not in positions_dict:
                continue

            positions = positions_dict[frame_idx]

            # Evaluate each player separately using the mapping
            for gt_player_id in [0, 1]:
                if gt_player_id in gt_data:
                    gt_pos = gt_data[gt_player_id]

                    # Use mapping to find corresponding tracking player
                    tracking_player_id = mapping[gt_player_id]
                    predicted_player = self.find_player_by_id(positions, tracking_player_id)

                    if predicted_player:
                        try:
                            predicted_pos = self.calculate_midpoint_position(predicted_player)
                            error = self.calculate_euclidean_error(predicted_pos, gt_pos)

                            player_key = f'player_{gt_player_id}'
                            results[player_key]['errors'].append(error)
                            results[player_key]['frame_results'].append({
                                'frame_index': frame_idx,
                                'ground_truth': gt_pos,
                                'predicted': predicted_pos,
                                'error': error,
                                'data_type': data_type,
                                'player_id': gt_player_id,
                                'tracking_player_id': tracking_player_id,
                                'mapping_used': str(mapping)
                            })

                            total_error += error
                            comparison_count += 1

                        except Exception:
                            # Skip frames with invalid position data
                            continue

        avg_error = total_error / comparison_count if comparison_count > 0 else float('inf')
        return results, avg_error

    def evaluate_tracking_dataset(self, positions_dict: Dict, data_type: str = "tracking") -> Dict:
        """Evaluate tracking data against ground truth, trying both player mappings."""
        # Try both possible player ID mappings
        mapping_1 = {0: 0, 1: 1}  # Direct mapping
        mapping_2 = {0: 1, 1: 0}  # Swapped mapping

        print(f"   Trying direct player mapping (GT 0→Track 0, GT 1→Track 1)...")
        results_1, avg_error_1 = self.evaluate_with_player_mapping(positions_dict, data_type, mapping_1)

        print(f"   Trying swapped player mapping (GT 0→Track 1, GT 1→Track 0)...")
        results_2, avg_error_2 = self.evaluate_with_player_mapping(positions_dict, data_type, mapping_2)

        # Choose the mapping with lower average error
        if avg_error_1 <= avg_error_2:
            best_results = results_1
            best_mapping = mapping_1
            best_error = avg_error_1
            print(f"   Selected direct mapping (average error: {avg_error_1:.3f}m)")
        else:
            best_results = results_2
            best_mapping = mapping_2
            best_error = avg_error_2
            print(f"   Selected swapped mapping (average error: {avg_error_2:.3f}m)")

        # Store the mapping information for reporting
        if not hasattr(self, 'player_mappings'):
            self.player_mappings = {}
        self.player_mappings[data_type] = {
            'mapping': best_mapping,
            'average_error': best_error,
            'direct_mapping_error': avg_error_1,
            'swapped_mapping_error': avg_error_2
        }

        print(f"   Comparison: Direct={avg_error_1:.3f}m vs Swapped={avg_error_2:.3f}m")

        return best_results

    def calculate_comprehensive_metrics(self, errors: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive statistical metrics."""
        if not errors:
            return {}

        errors_array = np.array(errors)

        # Basic statistics
        metrics = {
            'n_samples': len(errors),
            'mean': float(np.mean(errors_array)),
            'median': float(np.median(errors_array)),
            'std': float(np.std(errors_array)),
            'variance': float(np.var(errors_array)),
            'min': float(np.min(errors_array)),
            'max': float(np.max(errors_array)),
            'range': float(np.max(errors_array) - np.min(errors_array)),
            'rmse': float(np.sqrt(np.mean(errors_array**2))),
            'mae': float(np.mean(errors_array)),
        }

        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        metrics['percentiles'] = {}
        for p in percentiles:
            metrics['percentiles'][f'p{p}'] = float(np.percentile(errors_array, p))

        # Accuracy at different thresholds
        thresholds = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        metrics['accuracy_at_threshold'] = {}
        for threshold in thresholds:
            count = (errors_array <= threshold).sum()
            percentage = (count / len(errors_array)) * 100
            metrics['accuracy_at_threshold'][f'{threshold}m'] = {
                'count': int(count),
                'percentage': float(percentage)
            }

        # Distribution properties (if scipy available)
        if HAS_SCIPY:
            try:
                metrics['skewness'] = float(stats.skew(errors_array))
                metrics['kurtosis'] = float(stats.kurtosis(errors_array))

                # Confidence intervals (95%)
                ci_95 = stats.t.interval(0.95, len(errors_array)-1,
                                         loc=np.mean(errors_array),
                                         scale=stats.sem(errors_array))
                metrics['confidence_interval_95'] = {
                    'lower': float(ci_95[0]),
                    'upper': float(ci_95[1])
                }
            except:
                metrics['skewness'] = None
                metrics['kurtosis'] = None
                metrics['confidence_interval_95'] = None

        return metrics

    def perform_statistical_tests(self, original_errors: List[float],
                                  corrected_errors: List[float]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if not original_errors or not corrected_errors or not HAS_SCIPY:
            return {}

        results = {}

        # Independent t-test
        try:
            t_stat, p_value = stats.ttest_ind(original_errors, corrected_errors)
            results['independent_t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except Exception as e:
            results['independent_t_test_error'] = str(e)

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_value = stats.mannwhitneyu(original_errors, corrected_errors,
                                                 alternative='two-sided')
            results['mann_whitney_u'] = {
                'statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except Exception as e:
            results['mann_whitney_u_error'] = str(e)

        # Effect size (Cohen's d)
        try:
            orig_mean = np.mean(original_errors)
            corr_mean = np.mean(corrected_errors)
            pooled_std = np.sqrt(((len(original_errors) - 1) * np.var(original_errors) +
                                  (len(corrected_errors) - 1) * np.var(corrected_errors)) /
                                 (len(original_errors) + len(corrected_errors) - 2))

            if pooled_std > 0:
                cohens_d = (orig_mean - corr_mean) / pooled_std

                # Effect size interpretation
                if abs(cohens_d) < 0.2:
                    effect_size = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_size = "small"
                elif abs(cohens_d) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"

                results['effect_size'] = {
                    'cohens_d': float(cohens_d),
                    'interpretation': effect_size
                }
        except Exception as e:
            results['effect_size_error'] = str(e)

        return results

    def create_research_plots(self):
        """Create publication-quality plots for research analysis."""
        # Set basic style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")

        # Collect data for all players
        all_original_errors = []
        all_corrected_errors = []

        for player_key in ['player_0', 'player_1']:
            all_original_errors.extend(self.evaluation_results[player_key]['original_errors'])
            all_corrected_errors.extend(self.evaluation_results[player_key]['corrected_errors'])

        if not all_original_errors and not all_corrected_errors:
            print("No data available for plotting")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Error distributions comparison
        ax1 = plt.subplot(2, 3, 1)
        if all_original_errors and all_corrected_errors:
            max_error = max(max(all_original_errors), max(all_corrected_errors))
            bins = np.linspace(0, max_error, 30)
            ax1.hist(all_original_errors, bins=bins, alpha=0.6, label='Original',
                     color='red', density=True)
            ax1.hist(all_corrected_errors, bins=bins, alpha=0.6, label='Corrected',
                     color='blue', density=True)
            ax1.set_xlabel('Error (meters)')
            ax1.set_ylabel('Density')
            ax1.set_title('Error Distribution Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Box plots
        ax2 = plt.subplot(2, 3, 2)
        if all_original_errors and all_corrected_errors:
            data_to_plot = [all_original_errors, all_corrected_errors]
            labels = ['Original', 'Corrected']
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][1].set_facecolor('blue')
            ax2.set_ylabel('Error (meters)')
            ax2.set_title('Error Statistics Comparison')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Per-player comparison
        ax3 = plt.subplot(2, 3, 3)
        player_means_orig = []
        player_means_corr = []
        player_names = []

        for player_key in ['player_0', 'player_1']:
            original_errors = self.evaluation_results[player_key]['original_errors']
            corrected_errors = self.evaluation_results[player_key]['corrected_errors']

            if original_errors and corrected_errors:
                player_means_orig.append(np.mean(original_errors))
                player_means_corr.append(np.mean(corrected_errors))
                player_names.append(f'Player {player_key[-1]}')

        if player_means_orig and player_means_corr:
            x_pos = np.arange(len(player_names))
            width = 0.35
            ax3.bar(x_pos - width/2, player_means_orig, width, label='Original',
                    color='red', alpha=0.7)
            ax3.bar(x_pos + width/2, player_means_corr, width, label='Corrected',
                    color='blue', alpha=0.7)
            ax3.set_xlabel('Player')
            ax3.set_ylabel('Mean Error (meters)')
            ax3.set_title('Mean Error by Player')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(player_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Accuracy at thresholds
        ax4 = plt.subplot(2, 3, 4)
        thresholds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

        if all_original_errors and all_corrected_errors:
            orig_acc = [(np.array(all_original_errors) <= t).mean() * 100 for t in thresholds]
            corr_acc = [(np.array(all_corrected_errors) <= t).mean() * 100 for t in thresholds]

            ax4.plot(thresholds, orig_acc, 'o-', color='red', label='Original', linewidth=2)
            ax4.plot(thresholds, corr_acc, 's-', color='blue', label='Corrected', linewidth=2)
            ax4.set_xlabel('Error Threshold (meters)')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Accuracy at Different Thresholds')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Error over time
        ax5 = plt.subplot(2, 3, 5)
        all_frame_data = []
        for player_key in ['player_0', 'player_1']:
            all_frame_data.extend(self.evaluation_results[player_key]['frame_comparisons'])

        if all_frame_data:
            all_frame_data.sort(key=lambda x: x['frame_index'])

            orig_data = [x for x in all_frame_data if x['data_type'] == 'original']
            corr_data = [x for x in all_frame_data if x['data_type'] == 'corrected']

            if orig_data:
                orig_frames = [x['frame_index'] for x in orig_data]
                orig_errors = [x['error'] for x in orig_data]
                ax5.scatter(orig_frames, orig_errors, alpha=0.5, color='red',
                            label='Original', s=20)

            if corr_data:
                corr_frames = [x['frame_index'] for x in corr_data]
                corr_errors = [x['error'] for x in corr_data]
                ax5.scatter(corr_frames, corr_errors, alpha=0.5, color='blue',
                            label='Corrected', s=20)

            ax5.set_xlabel('Frame Index')
            ax5.set_ylabel('Error (meters)')
            ax5.set_title('Tracking Error Over Time')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Plot 6: Improvement distribution
        ax6 = plt.subplot(2, 3, 6)
        improvements = []

        for player_key in ['player_0', 'player_1']:
            frame_comparisons = self.evaluation_results[player_key]['frame_comparisons']

            orig_dict = {x['frame_index']: x['error'] for x in frame_comparisons if x['data_type'] == 'original'}
            corr_dict = {x['frame_index']: x['error'] for x in frame_comparisons if x['data_type'] == 'corrected'}

            for frame_idx in orig_dict:
                if frame_idx in corr_dict:
                    improvement = orig_dict[frame_idx] - corr_dict[frame_idx]
                    improvements.append(improvement)

        if improvements:
            ax6.hist(improvements, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
            ax6.axvline(x=np.mean(improvements), color='orange', linestyle='-', linewidth=2,
                        label=f'Mean: {np.mean(improvements):.3f}m')
            ax6.set_xlabel('Error Reduction (meters)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of Error Improvements')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.suptitle(f'Tracking Performance Analysis - {self.video_name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_file = self.results_dir / "research_analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Research plots saved to: {plot_file}")
        plt.close()

    def generate_research_paper_report(self) -> str:
        """Generate a research paper style error analysis report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Collect overall statistics
        all_original_errors = []
        all_corrected_errors = []
        player_stats = {}

        for player_key in ['player_0', 'player_1']:
            original_errors = self.evaluation_results[player_key]['original_errors']
            corrected_errors = self.evaluation_results[player_key]['corrected_errors']

            if original_errors:
                all_original_errors.extend(original_errors)
            if corrected_errors:
                all_corrected_errors.extend(corrected_errors)

            player_stats[player_key] = {
                'original_metrics': self.calculate_comprehensive_metrics(original_errors) if original_errors else {},
                'corrected_metrics': self.calculate_comprehensive_metrics(corrected_errors) if corrected_errors else {}
            }

        # Calculate overall metrics
        overall_original_metrics = self.calculate_comprehensive_metrics(all_original_errors)
        overall_corrected_metrics = self.calculate_comprehensive_metrics(all_corrected_errors)

        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests(all_original_errors, all_corrected_errors)

        # Generate report
        report = []

        # Title and abstract
        report.append("AUTOMATED BADMINTON PLAYER TRACKING SYSTEM EVALUATION:")
        report.append("A COMPREHENSIVE ERROR ANALYSIS WITH JUMP CORRECTION")
        report.append("=" * 80)
        report.append("")
        report.append("ABSTRACT")
        report.append("-" * 8)

        if overall_original_metrics and overall_corrected_metrics:
            mean_improvement = overall_original_metrics['mean'] - overall_corrected_metrics['mean']
            percent_improvement = (mean_improvement / overall_original_metrics['mean']) * 100

            report.append(f"This study evaluates an automated badminton player tracking system using")
            report.append(f"computer vision and pose estimation techniques. The system was tested on")
            report.append(f"video '{self.video_name}' with {overall_original_metrics['n_samples']} tracking")
            report.append(f"measurements validated against manual CVAT annotations. The original")
            report.append(f"tracking achieved a mean absolute error of {overall_original_metrics['mean']:.3f}m")

            if overall_original_metrics.get('confidence_interval_95'):
                ci = overall_original_metrics['confidence_interval_95']
                report.append(f"(95% CI: {ci['lower']:.3f}-{ci['upper']:.3f}m).")
            else:
                report.append(".")

            report.append(f"Jump correction algorithms reduced the mean error to {overall_corrected_metrics['mean']:.3f}m,")
            report.append(f"representing a {percent_improvement:.1f}% improvement. Statistical analysis")

            if statistical_tests.get('independent_t_test', {}).get('significant', False):
                report.append(f"confirmed the improvement as statistically significant (p < 0.05).")
            else:
                report.append(f"showed mixed statistical significance of the improvements.")

        report.append("")

        # Methodology
        report.append("1. METHODOLOGY")
        report.append("-" * 15)
        report.append("1.1 Data Collection and Ground Truth")
        report.append(f"Ground truth annotations were collected using CVAT (Computer Vision")
        report.append(f"Annotation Tool) and converted to world coordinates using homography")
        report.append(f"transformation. Manual annotations provided precise player positions")
        report.append(f"for accuracy assessment.")
        report.append("")
        report.append("1.2 Tracking System")
        report.append(f"The automated tracking system uses pose estimation to detect player")
        report.append(f"joint positions, followed by coordinate transformation to world space.")
        report.append(f"A jump correction algorithm was applied to reduce tracking errors")
        report.append(f"during rapid player movements.")
        report.append("")
        report.append("1.3 Player ID Mapping Resolution")
        if hasattr(self, 'player_mappings'):
            orig_mapping = self.player_mappings.get('original', {})
            corr_mapping = self.player_mappings.get('corrected', {})

            if orig_mapping:
                report.append(f"Automatic player ID mapping resolution was employed to handle")
                report.append(f"potential mismatches between ground truth and tracking player IDs.")
                report.append(f"Original tracking: {orig_mapping.get('mapping', 'N/A')} mapping selected")
                report.append(f"  Direct mapping error: {orig_mapping.get('direct_mapping_error', 0):.3f}m")
                report.append(f"  Swapped mapping error: {orig_mapping.get('swapped_mapping_error', 0):.3f}m")

                if corr_mapping:
                    report.append(f"Corrected tracking: {corr_mapping.get('mapping', 'N/A')} mapping selected")
                    report.append(f"  Direct mapping error: {corr_mapping.get('direct_mapping_error', 0):.3f}m")
                    report.append(f"  Swapped mapping error: {corr_mapping.get('swapped_mapping_error', 0):.3f}m")
                report.append("")

        report.append("1.4 Evaluation Metrics")
        report.append(f"Primary metric: Mean Absolute Error (MAE) in meters")
        report.append(f"Secondary metrics: RMSE, median error, 95th percentile error")
        if HAS_SCIPY:
            report.append(f"Statistical tests: Independent t-test, Mann-Whitney U test")
            report.append(f"Effect size: Cohen's d")
        report.append("")

        # Results
        report.append("2. RESULTS")
        report.append("-" * 10)

        # Overall performance
        report.append("2.1 Overall Tracking Performance")
        if overall_original_metrics and overall_corrected_metrics:
            report.append(f"Original Tracking:")
            report.append(f"  Mean Absolute Error: {overall_original_metrics['mean']:.3f} ± {overall_original_metrics['std']:.3f} m")
            report.append(f"  Root Mean Square Error: {overall_original_metrics['rmse']:.3f} m")
            report.append(f"  Median Error: {overall_original_metrics['median']:.3f} m")
            report.append(f"  95th Percentile Error: {overall_original_metrics['percentiles']['p95']:.3f} m")
            report.append(f"  Sample Size: n = {overall_original_metrics['n_samples']}")
            report.append("")
            report.append(f"Corrected Tracking (with Jump Correction):")
            report.append(f"  Mean Absolute Error: {overall_corrected_metrics['mean']:.3f} ± {overall_corrected_metrics['std']:.3f} m")
            report.append(f"  Root Mean Square Error: {overall_corrected_metrics['rmse']:.3f} m")
            report.append(f"  Median Error: {overall_corrected_metrics['median']:.3f} m")
            report.append(f"  95th Percentile Error: {overall_corrected_metrics['percentiles']['p95']:.3f} m")
            report.append(f"  Sample Size: n = {overall_corrected_metrics['n_samples']}")

            mean_improvement = overall_original_metrics['mean'] - overall_corrected_metrics['mean']
            percent_improvement = (mean_improvement / overall_original_metrics['mean']) * 100
            report.append("")
            report.append(f"Improvement Summary:")
            report.append(f"  Absolute Error Reduction: {mean_improvement:.3f} m ({percent_improvement:.1f}%)")
            report.append(f"  RMSE Reduction: {overall_original_metrics['rmse'] - overall_corrected_metrics['rmse']:.3f} m")
            report.append("")

        # Per-player analysis
        report.append("2.2 Per-Player Performance Analysis")
        for player_key in ['player_0', 'player_1']:
            orig_metrics = player_stats[player_key]['original_metrics']
            corr_metrics = player_stats[player_key]['corrected_metrics']

            if orig_metrics and corr_metrics:
                player_num = player_key[-1]
                report.append(f"Player {player_num}:")
                report.append(f"  Original MAE: {orig_metrics['mean']:.3f} m (n={orig_metrics['n_samples']})")
                report.append(f"  Corrected MAE: {corr_metrics['mean']:.3f} m (n={corr_metrics['n_samples']})")

                if orig_metrics['mean'] > 0:
                    player_improvement = ((orig_metrics['mean'] - corr_metrics['mean']) / orig_metrics['mean']) * 100
                    report.append(f"  Improvement: {player_improvement:.1f}%")
                report.append("")

        # Accuracy analysis
        report.append("2.3 Accuracy at Clinical Thresholds")
        if overall_original_metrics and overall_corrected_metrics:
            thresholds = [0.25, 0.5, 1.0, 1.5, 2.0]
            report.append("Percentage of measurements within threshold:")
            report.append("Threshold  | Original | Corrected | Improvement")
            report.append("-----------|----------|-----------|------------")

            for threshold in thresholds:
                orig_acc = overall_original_metrics['accuracy_at_threshold'][f'{threshold}m']['percentage']
                corr_acc = overall_corrected_metrics['accuracy_at_threshold'][f'{threshold}m']['percentage']
                improvement = corr_acc - orig_acc
                report.append(f"  {threshold:4.2f}m   |  {orig_acc:6.1f}%  |   {corr_acc:6.1f}%  |   {improvement:+5.1f}%")
            report.append("")

        # Statistical significance
        report.append("2.4 Statistical Significance Testing")
        if statistical_tests and HAS_SCIPY:
            if 'independent_t_test' in statistical_tests:
                it = statistical_tests['independent_t_test']
                report.append(f"Independent t-test:")
                report.append(f"  t-statistic: {it['statistic']:.3f}")
                report.append(f"  p-value: {it['p_value']:.6f}")
                report.append(f"  Statistically significant: {'Yes' if it['significant'] else 'No'} (α = 0.05)")
                report.append("")

            if 'mann_whitney_u' in statistical_tests:
                mw = statistical_tests['mann_whitney_u']
                report.append(f"Mann-Whitney U test (non-parametric):")
                report.append(f"  U-statistic: {mw['statistic']:.1f}")
                report.append(f"  p-value: {mw['p_value']:.6f}")
                report.append(f"  Statistically significant: {'Yes' if mw['significant'] else 'No'} (α = 0.05)")
                report.append("")

            if 'effect_size' in statistical_tests:
                es = statistical_tests['effect_size']
                report.append(f"Effect Size Analysis:")
                report.append(f"  Cohen's d: {es['cohens_d']:.3f}")
                report.append(f"  Interpretation: {es['interpretation']} effect")
                report.append("")
        elif not HAS_SCIPY:
            report.append("Statistical testing requires scipy package (not installed)")
            report.append("")

        # Discussion
        report.append("3. DISCUSSION")
        report.append("-" * 13)
        report.append("3.1 Performance Assessment")

        if overall_original_metrics and overall_corrected_metrics:
            if overall_corrected_metrics['mean'] < 0.5:
                performance_level = "excellent"
            elif overall_corrected_metrics['mean'] < 1.0:
                performance_level = "good"
            elif overall_corrected_metrics['mean'] < 1.5:
                performance_level = "acceptable"
            else:
                performance_level = "requires improvement"

            report.append(f"The tracking system demonstrates {performance_level} performance with a")
            report.append(f"mean absolute error of {overall_corrected_metrics['mean']:.3f}m after jump correction.")

            if overall_corrected_metrics['percentiles']['p95'] < 2.0:
                report.append(f"95% of measurements are within {overall_corrected_metrics['percentiles']['p95']:.3f}m,")
                report.append(f"indicating consistent tracking accuracy across different game scenarios.")
            else:
                report.append(f"However, 95th percentile errors of {overall_corrected_metrics['percentiles']['p95']:.3f}m")
                report.append(f"suggest occasional large tracking errors that may require attention.")

        report.append("")
        report.append("3.2 Jump Correction Effectiveness")

        if statistical_tests.get('independent_t_test', {}).get('significant', False):
            report.append(f"The jump correction algorithm shows statistically significant improvement")
            report.append(f"in tracking accuracy. This suggests the algorithm effectively addresses")
            report.append(f"tracking instabilities during rapid player movements characteristic")
            report.append(f"of badminton gameplay.")
        else:
            report.append(f"While numerical improvements were observed, statistical significance")
            report.append(f"was not consistently achieved across all tests. This may indicate")
            report.append(f"the need for algorithm refinement or larger sample sizes.")

        report.append("")
        report.append("3.3 Player ID Mapping Analysis")
        if hasattr(self, 'player_mappings'):
            orig_mapping = self.player_mappings.get('original', {})
            if orig_mapping:
                direct_error = orig_mapping.get('direct_mapping_error', 0)
                swapped_error = orig_mapping.get('swapped_mapping_error', 0)
                selected_mapping = orig_mapping.get('mapping', {})

                if abs(direct_error - swapped_error) > 1.0:
                    report.append(f"Significant player ID mapping issues were detected and automatically")
                    report.append(f"resolved. The direct mapping yielded {direct_error:.3f}m average error")
                    report.append(f"while the swapped mapping yielded {swapped_error:.3f}m average error.")
                    report.append(f"This indicates inconsistent player labeling between ground truth")
                    report.append(f"and tracking data, which was corrected by selecting the optimal mapping.")
                else:
                    report.append(f"Player ID mapping was consistent between ground truth and tracking")
                    report.append(f"data, with minimal difference between mapping strategies.")

        report.append("")
        report.append("3.4 Limitations and Future Work")
        report.append(f"This evaluation was conducted on a single video sequence and may not")
        report.append(f"generalize to all court conditions, lighting scenarios, or player")
        report.append(f"movement patterns. Future work should include:")
        report.append(f"  • Multi-video validation across different courts and conditions")
        report.append(f"  • Analysis of tracking performance during specific game events")
        report.append(f"  • Comparison with other state-of-the-art tracking methods")
        report.append(f"  • Real-time performance evaluation")
        report.append("")

        # Conclusion
        report.append("4. CONCLUSION")
        report.append("-" * 13)

        if overall_original_metrics and overall_corrected_metrics:
            mean_improvement = overall_original_metrics['mean'] - overall_corrected_metrics['mean']
            percent_improvement = (mean_improvement / overall_original_metrics['mean']) * 100

            report.append(f"This study demonstrates the effectiveness of an automated badminton")
            report.append(f"player tracking system with jump correction capabilities. The system")
            report.append(f"achieved a mean tracking error of {overall_corrected_metrics['mean']:.3f}m, representing a")
            report.append(f"{percent_improvement:.1f}% improvement over baseline tracking.")

            if overall_corrected_metrics['accuracy_at_threshold']['1.0m']['percentage'] > 80:
                report.append(f"With {overall_corrected_metrics['accuracy_at_threshold']['1.0m']['percentage']:.1f}% of measurements within 1.0m accuracy,")
                report.append(f"the system shows promise for automated sports analysis applications.")
            else:
                report.append(f"However, with only {overall_corrected_metrics['accuracy_at_threshold']['1.0m']['percentage']:.1f}% of measurements within")
                report.append(f"1.0m accuracy, further algorithmic improvements are recommended.")

        report.append("")

        # Technical specifications
        report.append("5. TECHNICAL SPECIFICATIONS")
        report.append("-" * 28)
        report.append(f"Video File: {self.video_name}")
        report.append(f"Ground Truth Method: CVAT manual annotation")
        report.append(f"Coordinate System: World coordinates via homography")
        report.append(f"Evaluation Date: {timestamp}")
        report.append(f"Tracking Method: Pose estimation + coordinate transformation")
        report.append(f"Jump Correction: Temporal smoothing algorithm")
        report.append(f"Dependencies: numpy, matplotlib" + (", scipy, seaborn" if HAS_SCIPY and HAS_SEABORN else ""))
        report.append("")

        # Data summary
        report.append("6. DATA SUMMARY")
        report.append("-" * 16)

        total_gt_frames = len(self.ground_truth)
        total_orig_frames = len(self.original_positions)
        total_corr_frames = len(self.corrected_positions)

        report.append(f"Ground Truth Annotations: {total_gt_frames} frames")
        report.append(f"Original Tracking Results: {total_orig_frames} frames")
        report.append(f"Corrected Tracking Results: {total_corr_frames} frames")

        if overall_original_metrics:
            report.append(f"Evaluation Coverage: {overall_original_metrics['n_samples']} measurements")

        # Player-specific coverage
        for player_key in ['player_0', 'player_1']:
            orig_metrics = player_stats[player_key]['original_metrics']
            if orig_metrics:
                player_num = player_key[-1]
                report.append(f"Player {player_num} Coverage: {orig_metrics['n_samples']} measurements")

        return "\n".join(report)

    def save_comprehensive_results(self, all_metrics: Dict, statistical_tests: Dict):
        """Save comprehensive evaluation results to JSON."""
        def convert_numpy_types(obj):
            """Convert numpy types for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj

        # Collect all frame comparisons
        all_frame_comparisons = []
        for player_key in ['player_0', 'player_1']:
            all_frame_comparisons.extend(self.evaluation_results[player_key]['frame_comparisons'])

        results = {
            'evaluation_metadata': {
                'video_name': self.video_name,
                'video_path': str(self.video_path),
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluator_version': 'CVAT_Research_Evaluator_v3.0_AutoOptimization',
                'ground_truth_method': 'CVAT_manual_annotation',
                'coordinate_system': 'world_coordinates_homography',
                'dependencies': {
                    'scipy_available': HAS_SCIPY,
                    'seaborn_available': HAS_SEABORN
                }
            },
            'optimization_analysis': {
                'evaluation_configs': getattr(self, 'evaluation_configs', {}),
                'all_combinations': getattr(self, 'all_combinations', {})
            },
            'data_summary': {
                'ground_truth_frames': len(self.ground_truth),
                'original_tracking_frames': len(self.original_positions),
                'corrected_tracking_frames': len(self.corrected_positions),
                'total_comparisons': len(all_frame_comparisons)
            },
            'overall_metrics': all_metrics,
            'statistical_analysis': statistical_tests,
            'per_player_results': {},
            'frame_level_comparisons': all_frame_comparisons
        }

        # Add per-player detailed results
        for player_key in ['player_0', 'player_1']:
            original_errors = self.evaluation_results[player_key]['original_errors']
            corrected_errors = self.evaluation_results[player_key]['corrected_errors']

            results['per_player_results'][player_key] = {
                'original_metrics': self.calculate_comprehensive_metrics(original_errors) if original_errors else None,
                'corrected_metrics': self.calculate_comprehensive_metrics(corrected_errors) if corrected_errors else None,
                'sample_sizes': {
                    'original': len(original_errors),
                    'corrected': len(corrected_errors)
                }
            }

        # Convert numpy types
        results = convert_numpy_types(results)

        # Save to file
        results_file = self.results_dir / "comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Comprehensive results saved to: {results_file}")

    def run_evaluation(self) -> bool:
        """Run the complete CVAT-based evaluation."""
        print(f"CVAT-Based Badminton Tracking Evaluation")
        print(f"Video: {self.video_name}")
        print("=" * 60)

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Check dependencies
        if not HAS_SCIPY:
            print("Warning: scipy not available - statistical tests will be skipped")
        if not HAS_SEABORN:
            print("Warning: seaborn not available - using matplotlib default styling")

        # Load all data
        if not self.load_cvat_ground_truth():
            print("Failed to load ground truth data")
            return False

        original_loaded, corrected_loaded = self.load_tracking_data()

        if not original_loaded and not corrected_loaded:
            print("No tracking data found!")
            return False

        print("\nEvaluating tracking performance...")

        # Evaluate tracking data
        all_metrics = {}

        if original_loaded:
            print("Analyzing original tracking results...")
            original_results = self.evaluate_tracking_dataset(self.original_positions, "original")

            for player_key, player_data in original_results.items():
                if player_data['errors']:
                    self.evaluation_results[player_key]['original_errors'] = player_data['errors']
                    self.evaluation_results[player_key]['frame_comparisons'].extend(player_data['frame_results'])

            # Calculate overall original metrics
            all_original_errors = []
            for player_key in ['player_0', 'player_1']:
                all_original_errors.extend(self.evaluation_results[player_key]['original_errors'])

            all_metrics['original'] = self.calculate_comprehensive_metrics(all_original_errors)
            print(f"   Original evaluation complete: {len(all_original_errors)} measurements")

        if corrected_loaded:
            print("Analyzing corrected tracking results...")
            corrected_results = self.evaluate_tracking_dataset(self.corrected_positions, "corrected")

            for player_key, player_data in corrected_results.items():
                if player_data['errors']:
                    self.evaluation_results[player_key]['corrected_errors'] = player_data['errors']
                    self.evaluation_results[player_key]['frame_comparisons'].extend(player_data['frame_results'])

            # Calculate overall corrected metrics
            all_corrected_errors = []
            for player_key in ['player_0', 'player_1']:
                all_corrected_errors.extend(self.evaluation_results[player_key]['corrected_errors'])

            all_metrics['corrected'] = self.calculate_comprehensive_metrics(all_corrected_errors)
            print(f"   Corrected evaluation complete: {len(all_corrected_errors)} measurements")

        # Perform statistical analysis
        statistical_tests = {}
        if original_loaded and corrected_loaded:
            print("Performing statistical significance testing...")
            all_original_errors = []
            all_corrected_errors = []

            for player_key in ['player_0', 'player_1']:
                all_original_errors.extend(self.evaluation_results[player_key]['original_errors'])
                all_corrected_errors.extend(self.evaluation_results[player_key]['corrected_errors'])

            statistical_tests = self.perform_statistical_tests(all_original_errors, all_corrected_errors)
            print("   Statistical analysis complete")

        # Generate research report
        print("Generating research paper style report...")
        research_report = self.generate_research_paper_report()

        # Save report
        report_file = self.results_dir / "research_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(research_report)
        print(f"Research report saved to: {report_file}")

        # Create research plots
        print("Creating research analysis plots...")
        self.create_research_plots()

        # Save comprehensive results with mapping information
        print("Saving comprehensive evaluation results...")
        self.save_comprehensive_results(all_metrics, statistical_tests)

        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Print mapping information
        if hasattr(self, 'player_mappings'):
            print("PLAYER ID MAPPING RESOLUTION:")
            for data_type, mapping_info in self.player_mappings.items():
                mapping = mapping_info['mapping']
                print(f"  {data_type.title()}: GT Player 0→Track Player {mapping[0]}, GT Player 1→Track Player {mapping[1]}")
                print(f"    Direct mapping would give: {mapping_info['direct_mapping_error']:.3f}m average error")
                print(f"    Swapped mapping would give: {mapping_info['swapped_mapping_error']:.3f}m average error")
                print(f"    Selected mapping gives: {mapping_info['average_error']:.3f}m average error")
            print("")

        # Print key findings
        if all_metrics.get('original') and all_metrics.get('corrected'):
            orig_mae = all_metrics['original']['mean']
            corr_mae = all_metrics['corrected']['mean']
            improvement = ((orig_mae - corr_mae) / orig_mae) * 100

            print(f"PERFORMANCE METRICS:")
            print(f"  Original MAE: {orig_mae:.3f}m")
            print(f"  Corrected MAE: {corr_mae:.3f}m")
            print(f"  Improvement: {improvement:.1f}%")
            print("")

            if statistical_tests.get('independent_t_test', {}).get('significant', False):
                print("  Improvement is statistically significant (p < 0.05)")
            else:
                print("  Improvement not statistically significant")

            # Performance assessment
            if corr_mae < 0.5:
                assessment = "EXCELLENT - suitable for detailed analysis"
            elif corr_mae < 1.0:
                assessment = "GOOD - suitable for general tracking"
            elif corr_mae < 2.0:
                assessment = "ACCEPTABLE - may need refinement"
            else:
                assessment = "POOR - requires significant improvement"

            print(f"  Overall Assessment: {assessment}")

        print(f"\nAll results saved to: {self.results_dir}")
        print("CVAT-based evaluation completed successfully!")

        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='CVAT-Based Badminton Tracking Evaluator with Research Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CVAT-Based Badminton Tracking Evaluator with Automatic Player Mapping
===================================================================

This evaluator performs comprehensive research-grade analysis of badminton 
tracking systems against CVAT ground truth annotations with automatic
resolution of player ID mapping issues.

Features:
• Evaluates against CVAT ground truth (cvat_ground_truth.json)
• Compares original vs corrected tracking performance
• Automatic player ID mapping resolution
• Generates research paper style error analysis
• Performs statistical significance testing (requires scipy)
• Creates publication-quality plots
• Calculates comprehensive accuracy metrics

Required Directory Structure:
results/[video_name]/
├── cvat_ground_truth.json    (CVAT converted ground truth)
├── positions.json            (Original tracking results)
├── corrected_positions.json  (Jump-corrected results)

Generated Outputs:
├── research_analysis_report.txt
├── research_analysis_plots.png
├── comprehensive_evaluation_results.json

Dependencies:
Required: numpy, matplotlib
Optional: scipy (for statistical tests), seaborn (for better plots)

Usage Examples:
python3 evaluation/evaluator.py video.mp4
python3 evaluation/evaluator.py /path/to/video.mp4
        """
    )

    parser.add_argument('video_path', help='Path to the video file')

    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Validate video path
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        # Initialize and run evaluator
        evaluator = CVATTrackingEvaluator(args.video_path)
        success = evaluator.run_evaluation()

        if not success:
            print("Evaluation failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()