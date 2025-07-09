#!/usr/bin/env python3
"""
Tracking Evaluation Results Combiner

This script automatically finds all detailed_evaluation_results.json files
and combines them into a comprehensive research report with statistical analysis.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import argparse

class TrackingEvaluationCombiner:
    def __init__(self, search_directory: str = "."):
        """
        Initialize the combiner with a search directory.

        Args:
            search_directory: Root directory to search for JSON files
        """
        self.search_directory = Path(search_directory)
        self.evaluation_data = []
        self.combined_stats = {}

    def find_evaluation_files(self) -> List[Path]:
        """Find all detailed_evaluation_results.json files recursively."""
        pattern = "**/detailed_evaluation_results.json"
        files = list(self.search_directory.glob(pattern))

        if not files:
            # Also try case variations
            patterns = [
                "**/detailed_evaluation_results.json",
                "**/Detailed_Evaluation_Results.json",
                "**/detailed_evaluation_results.JSON"
            ]
            for pattern in patterns:
                files.extend(self.search_directory.glob(pattern))

        print(f"Found {len(files)} evaluation files:")
        for file in files:
            print(f"  - {file}")

        return files

    def load_evaluation_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and validate a single evaluation file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Add file metadata
            data['file_info'] = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'parent_directory': file_path.parent.name,
                'file_size_bytes': file_path.stat().st_size,
                'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }

            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from evaluation data."""
        if not data or 'evaluation_results' not in data:
            return {}

        results = data['evaluation_results']
        video_info = data.get('video_info', {})
        tracking_info = data.get('tracking_info', {})

        metrics = {
            'video_name': video_info.get('video_name', 'unknown'),
            'video_path': video_info.get('video_path', 'unknown'),
            'evaluation_date': video_info.get('evaluation_date', 'unknown'),
            'tracking_method': tracking_info.get('tracking_method', 'unknown'),
            'jump_correction_applied': tracking_info.get('jump_correction_applied', False),
            'total_comparisons': results.get('total_comparisons', 0),
            'mean_error_meters': results.get('mean_error_meters', 0),
            'median_error_meters': results.get('median_error_meters', 0),
            'std_error_meters': results.get('std_error_meters', 0),
            'rmse_meters': results.get('rmse_meters', 0),
            'max_error_meters': results.get('max_error_meters', 0),
            'min_error_meters': results.get('min_error_meters', 0),
            'coverage_percentage': results.get('coverage', {}).get('coverage_percentage', 0),
            'file_path': data.get('file_info', {}).get('file_path', 'unknown')
        }

        # Extract accuracy thresholds
        accuracy_thresholds = results.get('accuracy_thresholds', {})
        for threshold, data_point in accuracy_thresholds.items():
            metrics[f'accuracy_{threshold}'] = data_point.get('percentage', 0)

        # Extract percentiles
        percentiles = results.get('error_percentiles', {})
        for percentile, value in percentiles.items():
            metrics[f'error_{percentile}_percentile'] = value

        return metrics

    def load_all_evaluations(self):
        """Load all evaluation files and extract metrics."""
        files = self.find_evaluation_files()

        if not files:
            raise FileNotFoundError("No detailed_evaluation_results.json files found!")

        self.evaluation_data = []

        for file_path in files:
            data = self.load_evaluation_file(file_path)
            if data:
                metrics = self.extract_key_metrics(data)
                if metrics:
                    self.evaluation_data.append({
                        'raw_data': data,
                        'metrics': metrics
                    })

        print(f"Successfully loaded {len(self.evaluation_data)} evaluation files")

    def calculate_combined_statistics(self) -> Dict[str, Any]:
        """Calculate combined statistics across all evaluations."""
        if not self.evaluation_data:
            return {}

        # Extract metrics into DataFrame for easier analysis
        metrics_list = [item['metrics'] for item in self.evaluation_data]
        df = pd.DataFrame(metrics_list)

        # Calculate aggregate statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        stats = {
            'summary': {
                'total_evaluations': len(self.evaluation_data),
                'total_comparisons': df['total_comparisons'].sum(),
                'unique_videos': df['video_name'].nunique(),
                'unique_tracking_methods': df['tracking_method'].nunique(),
                'date_range': {
                    'earliest': df['evaluation_date'].min(),
                    'latest': df['evaluation_date'].max()
                }
            },
            'error_statistics': {},
            'accuracy_statistics': {},
            'coverage_statistics': {
                'mean_coverage': float(df['coverage_percentage'].mean()) if 'coverage_percentage' in df.columns else 0,
                'std_coverage': float(df['coverage_percentage'].std()) if 'coverage_percentage' in df.columns else 0,
                'min_coverage': float(df['coverage_percentage'].min()) if 'coverage_percentage' in df.columns else 0,
                'max_coverage': float(df['coverage_percentage'].max()) if 'coverage_percentage' in df.columns else 0
            }
        }

        # Error statistics
        error_metrics = ['mean_error_meters', 'median_error_meters', 'rmse_meters',
                         'std_error_meters', 'max_error_meters', 'min_error_meters']

        for metric in error_metrics:
            if metric in df.columns:
                values = df[metric].dropna()  # Remove NaN values
                if len(values) > 0:
                    stats['error_statistics'][metric] = {
                        'overall_mean': float(values.mean()),
                        'overall_std': float(values.std()),
                        'overall_min': float(values.min()),
                        'overall_max': float(values.max()),
                        'overall_median': float(values.median())
                    }

        # Accuracy statistics
        accuracy_columns = [col for col in df.columns if col.startswith('accuracy_')]
        for col in accuracy_columns:
            threshold = col.replace('accuracy_', '')
            values = df[col].dropna()  # Remove NaN values
            if len(values) > 0:
                stats['accuracy_statistics'][threshold] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median())
                }

        # Per-video statistics
        per_video_grouped = df.groupby('video_name')[error_metrics].agg(['mean', 'std'])
        stats['per_video_stats'] = {}
        for video in per_video_grouped.index:
            stats['per_video_stats'][video] = {}
            for metric in error_metrics:
                if metric in per_video_grouped.columns.get_level_values(0):
                    stats['per_video_stats'][video][f'{metric}_mean'] = per_video_grouped.loc[video, (metric, 'mean')]
                    stats['per_video_stats'][video][f'{metric}_std'] = per_video_grouped.loc[video, (metric, 'std')]

        # Per-method statistics
        if 'tracking_method' in df.columns and df['tracking_method'].nunique() > 1:
            per_method_grouped = df.groupby('tracking_method')[error_metrics].agg(['mean', 'std'])
            stats['per_method_stats'] = {}
            for method in per_method_grouped.index:
                stats['per_method_stats'][method] = {}
                for metric in error_metrics:
                    if metric in per_method_grouped.columns.get_level_values(0):
                        mean_val = per_method_grouped.loc[method, (metric, 'mean')]
                        std_val = per_method_grouped.loc[method, (metric, 'std')]
                        stats['per_method_stats'][method][f'{metric}_mean'] = float(mean_val) if pd.notna(mean_val) else 0.0
                        stats['per_method_stats'][method][f'{metric}_std'] = float(std_val) if pd.notna(std_val) else 0.0

        return stats

    def generate_visualizations(self, output_dir: Path):
        """Generate visualization plots."""
        if not self.evaluation_data:
            return

        # Create visualizations directory
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # Extract metrics for plotting
        metrics_list = [item['metrics'] for item in self.evaluation_data]
        df = pd.DataFrame(metrics_list)

        plt.style.use('default')

        # 1. Error distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Mean error distribution
        axes[0, 0].hist(df['mean_error_meters'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Mean Errors')
        axes[0, 0].set_xlabel('Mean Error (meters)')
        axes[0, 0].set_ylabel('Frequency')

        # RMSE distribution
        axes[0, 1].hist(df['rmse_meters'], bins=20, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Distribution of RMSE')
        axes[0, 1].set_xlabel('RMSE (meters)')
        axes[0, 1].set_ylabel('Frequency')

        # Max error distribution
        axes[1, 0].hist(df['max_error_meters'], bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribution of Max Errors')
        axes[1, 0].set_xlabel('Max Error (meters)')
        axes[1, 0].set_ylabel('Frequency')

        # Coverage distribution
        axes[1, 1].hist(df['coverage_percentage'], bins=20, alpha=0.7, color='gold')
        axes[1, 1].set_title('Distribution of Coverage Percentage')
        axes[1, 1].set_xlabel('Coverage (%)')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(viz_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy vs error correlation
        fig, ax = plt.subplots(figsize=(10, 8))
        accuracy_cols = [col for col in df.columns if col.startswith('accuracy_')]

        if accuracy_cols:
            # Use the first accuracy threshold for correlation
            acc_col = accuracy_cols[0]
            ax.scatter(df['mean_error_meters'], df[acc_col], alpha=0.6)
            ax.set_xlabel('Mean Error (meters)')
            ax.set_ylabel(f'{acc_col.replace("accuracy_", "Accuracy at ")} (%)')
            ax.set_title('Error vs Accuracy Correlation')

            # Add trend line
            z = np.polyfit(df['mean_error_meters'], df[acc_col], 1)
            p = np.poly1d(z)
            ax.plot(df['mean_error_meters'], p(df['mean_error_meters']), "r--", alpha=0.8)

        plt.savefig(viz_dir / 'error_accuracy_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Per-video comparison (if multiple videos)
        if df['video_name'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            video_stats = df.groupby('video_name')['mean_error_meters'].agg(['mean', 'std'])

            x_pos = range(len(video_stats))
            ax.bar(x_pos, video_stats['mean'], yerr=video_stats['std'],
                   capsize=5, alpha=0.7, color='steelblue')
            ax.set_xlabel('Video')
            ax.set_ylabel('Mean Error (meters)')
            ax.set_title('Mean Error by Video')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(video_stats.index, rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(viz_dir / 'per_video_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to {viz_dir}")

    def _json_serializer(self, obj):
        """Custom JSON serializer to handle numpy types and other non-serializable objects."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)

    def generate_report(self, output_file: str = "combined_tracking_evaluation_report.json"):
        """Generate the combined report."""
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True)

        # Calculate combined statistics
        self.combined_stats = self.calculate_combined_statistics()

        # Prepare report structure
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'script_version': '1.0.0',
                'search_directory': str(self.search_directory),
                'total_files_processed': len(self.evaluation_data)
            },
            'combined_statistics': self.combined_stats,
            'individual_evaluations': []
        }

        # Add individual evaluation summaries
        for item in self.evaluation_data:
            eval_summary = {
                'video_info': item['raw_data'].get('video_info', {}),
                'tracking_info': item['raw_data'].get('tracking_info', {}),
                'key_metrics': item['metrics'],
                'file_info': item['raw_data'].get('file_info', {})
            }
            report['individual_evaluations'].append(eval_summary)

        # Save main report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)

        # Generate CSV summary for easy analysis
        csv_path = output_path.with_suffix('.csv')
        metrics_df = pd.DataFrame([item['metrics'] for item in self.evaluation_data])
        metrics_df.to_csv(csv_path, index=False)

        # Generate markdown summary
        md_path = output_path.with_suffix('.md')
        self.generate_markdown_summary(md_path)

        # Generate visualizations
        self.generate_visualizations(output_dir)

        print(f"\nReport generation complete!")
        print(f"Main report: {output_path}")
        print(f"CSV summary: {csv_path}")
        print(f"Markdown summary: {md_path}")

        return output_path

    def generate_markdown_summary(self, output_path: Path):
        """Generate a markdown summary report."""
        with open(output_path, 'w') as f:
            f.write("# Tracking Evaluation Combined Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.combined_stats:
                f.write("No data available.\n")
                return

            # Summary statistics
            summary = self.combined_stats.get('summary', {})
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Evaluations**: {summary.get('total_evaluations', 0)}\n")
            f.write(f"- **Total Comparisons**: {summary.get('total_comparisons', 0):,}\n")
            f.write(f"- **Unique Videos**: {summary.get('unique_videos', 0)}\n")
            f.write(f"- **Unique Tracking Methods**: {summary.get('unique_tracking_methods', 0)}\n\n")

            # Error statistics
            error_stats = self.combined_stats.get('error_statistics', {})
            if error_stats:
                f.write("## Error Statistics (Overall)\n\n")
                f.write("| Metric | Mean | Std | Min | Max | Median |\n")
                f.write("|--------|------|-----|-----|-----|--------|\n")

                for metric, stats in error_stats.items():
                    f.write(f"| {metric.replace('_', ' ').title()} | "
                            f"{stats.get('overall_mean', 0):.4f} | "
                            f"{stats.get('overall_std', 0):.4f} | "
                            f"{stats.get('overall_min', 0):.4f} | "
                            f"{stats.get('overall_max', 0):.4f} | "
                            f"{stats.get('overall_median', 0):.4f} |\n")
                f.write("\n")

            # Accuracy statistics
            acc_stats = self.combined_stats.get('accuracy_statistics', {})
            if acc_stats:
                f.write("## Accuracy Statistics\n\n")
                f.write("| Threshold | Mean (%) | Std (%) | Min (%) | Max (%) | Median (%) |\n")
                f.write("|-----------|----------|---------|---------|---------|------------|\n")

                for threshold, stats in acc_stats.items():
                    f.write(f"| {threshold} | "
                            f"{stats.get('mean', 0):.2f} | "
                            f"{stats.get('std', 0):.2f} | "
                            f"{stats.get('min', 0):.2f} | "
                            f"{stats.get('max', 0):.2f} | "
                            f"{stats.get('median', 0):.2f} |\n")
                f.write("\n")

            # Individual evaluations summary
            f.write("## Individual Evaluations\n\n")
            for i, item in enumerate(self.evaluation_data, 1):
                metrics = item['metrics']
                f.write(f"### {i}. {metrics.get('video_name', 'Unknown')}\n\n")
                f.write(f"- **File**: `{metrics.get('file_path', 'Unknown')}`\n")
                f.write(f"- **Tracking Method**: {metrics.get('tracking_method', 'Unknown')}\n")
                f.write(f"- **Total Comparisons**: {metrics.get('total_comparisons', 0)}\n")
                f.write(f"- **Mean Error**: {metrics.get('mean_error_meters', 0):.4f} m\n")
                f.write(f"- **RMSE**: {metrics.get('rmse_meters', 0):.4f} m\n")
                f.write(f"- **Coverage**: {metrics.get('coverage_percentage', 0):.2f}%\n\n")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Combine tracking evaluation results into a comprehensive report"
    )
    parser.add_argument(
        "--search-dir",
        type=str,
        default=".",
        help="Directory to search for detailed_evaluation_results.json files (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_tracking_evaluation_report.json",
        help="Output file name for the combined report (default: combined_tracking_evaluation_report.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Initialize combiner
        combiner = TrackingEvaluationCombiner(args.search_dir)

        # Load all evaluation files
        combiner.load_all_evaluations()

        if not combiner.evaluation_data:
            print("No evaluation files found! Please check the search directory.")
            return

        # Generate combined report
        output_path = combiner.generate_report(args.output)

        # Print summary
        stats = combiner.combined_stats
        if stats and 'summary' in stats:
            print(f"\n=== SUMMARY ===")
            print(f"Total evaluations processed: {stats['summary']['total_evaluations']}")
            print(f"Total comparisons: {stats['summary']['total_comparisons']:,}")
            print(f"Unique videos: {stats['summary']['unique_videos']}")

            if 'error_statistics' in stats and 'mean_error_meters' in stats['error_statistics']:
                mean_error = stats['error_statistics']['mean_error_meters']['overall_mean']
                print(f"Overall mean error: {mean_error:.4f} meters")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()