#!/usr/bin/env python3
"""
Badminton Position and Leap Frame Movement Visualization (BPLFMV) Pipeline

Complete automated pipeline for badminton video analysis:
1. Court detection
2. Pose estimation
3. Position calculation
4. Jump artifact removal
5. Visualization (Stage 4 - corrected positions)

Usage: python3 pipeline.py samples/ds1.mp4
"""

import sys
import os
import subprocess
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

class BPLFMVPipeline:
    """Complete BPLFMV analysis pipeline"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name

        # Stage file paths
        self.court_csv = self.results_dir / "court.csv"
        self.pose_json = self.results_dir / "pose.json"
        self.positions_json = self.results_dir / "positions.json"
        self.corrected_positions_json = self.results_dir / "corrected_positions.json"
        self.final_visualization = self.results_dir / f"{self.video_name}_corrected_viz.mp4"

        # Validate input
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Initialized pipeline for: {self.video_path}")
        logging.info(f"Results directory: {self.results_dir}")

    def run_subprocess(self, cmd: list, stage_name: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
        """Run subprocess with proper error handling and logging"""
        logging.info(f"Starting {stage_name}...")
        logging.info(f"Command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                logging.info(f"✅ {stage_name} completed successfully in {duration:.2f}s")
                if result.stdout.strip():
                    logging.info(f"{stage_name} output:\n{result.stdout}")
                return True, result.stdout
            else:
                logging.error(f"❌ {stage_name} failed with return code {result.returncode}")
                logging.error(f"Error output:\n{result.stderr}")
                if result.stdout.strip():
                    logging.error(f"Standard output:\n{result.stdout}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logging.error(f"❌ {stage_name} timed out after {timeout}s")
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            logging.error(f"❌ {stage_name} failed with exception: {e}")
            return False, str(e)

    def stage1_detect_court(self) -> bool:
        """Stage 1: Court detection"""
        if self.court_csv.exists():
            logging.info(f"Court CSV already exists: {self.court_csv}")
            return True

        cmd = ["python3", "detect_court.py", str(self.video_path)]
        success, output = self.run_subprocess(cmd, "Court Detection", timeout=300)

        if success and self.court_csv.exists():
            logging.info(f"✅ Court detection output saved to: {self.court_csv}")
            return True
        else:
            logging.error("❌ Court detection failed or output not found")
            return False

    def stage2_detect_pose(self) -> bool:
        """Stage 2: Pose estimation"""
        if not self.court_csv.exists():
            logging.error("❌ Court CSV not found - cannot proceed with pose detection")
            return False

        if self.pose_json.exists():
            logging.info(f"Pose JSON already exists: {self.pose_json}")
            return True

        cmd = ["python3", "detect_pose.py", str(self.video_path)]
        success, output = self.run_subprocess(cmd, "Pose Detection", timeout=600)

        if success and self.pose_json.exists():
            logging.info(f"✅ Pose detection output saved to: {self.pose_json}")
            return True
        else:
            logging.error("❌ Pose detection failed or output not found")
            return False

    def stage3_calculate_location(self) -> bool:
        """Stage 3: Position calculation"""
        if not self.pose_json.exists():
            logging.error("❌ Pose JSON not found - cannot proceed with position calculation")
            return False

        if self.positions_json.exists():
            logging.info(f"Positions JSON already exists: {self.positions_json}")
            return True

        cmd = ["python3", "calculate_location.py", str(self.video_path)]
        success, output = self.run_subprocess(cmd, "Position Calculation", timeout=900)

        if success and self.positions_json.exists():
            logging.info(f"✅ Position calculation output saved to: {self.positions_json}")
            return True
        else:
            logging.error("❌ Position calculation failed or output not found")
            return False

    def stage4_remove_artifact(self) -> bool:
        """Stage 4: Jump artifact removal"""
        if not self.positions_json.exists():
            logging.error("❌ Positions JSON not found - cannot proceed with artifact removal")
            return False

        if self.corrected_positions_json.exists():
            logging.info(f"Corrected positions JSON already exists: {self.corrected_positions_json}")
            return True

        # Check if YOLO model exists
        model_path = Path("resources/BLPFMV.pt")
        if not model_path.exists():
            logging.warning(f"⚠️  YOLO model not found at {model_path}")
            logging.warning("⚠️  Jump artifact removal may not work optimally")

        cmd = ["python3", "remove_artifact.py", str(self.video_path)]
        success, output = self.run_subprocess(cmd, "Jump Artifact Removal", timeout=1200)

        if success and self.corrected_positions_json.exists():
            logging.info(f"✅ Artifact removal output saved to: {self.corrected_positions_json}")
            return True
        else:
            logging.error("❌ Artifact removal failed or output not found")
            return False

    def stage5_visualize(self) -> bool:
        """Stage 5: Visualization (Stage 4 - corrected positions)"""
        if not self.corrected_positions_json.exists():
            logging.error("❌ Corrected positions JSON not found - cannot proceed with visualization")
            return False

        if self.final_visualization.exists():
            logging.info(f"Final visualization already exists: {self.final_visualization}")
            return True

        cmd = [
            "python3", "visualize.py",
            str(self.video_path),
            "--stage", "4",
            "--data_path", str(self.corrected_positions_json),
            "--output", str(self.final_visualization)
        ]

        success, output = self.run_subprocess(cmd, "Final Visualization", timeout=1800)

        if success and self.final_visualization.exists():
            logging.info(f"✅ Final visualization saved to: {self.final_visualization}")
            return True
        else:
            logging.error("❌ Visualization failed or output not found")
            return False

    def check_dependencies(self) -> bool:
        """Check if all required scripts and dependencies exist"""
        required_scripts = [
            "detect_court.py",
            "detect_pose.py",
            "calculate_location.py",
            "remove_artifact.py",
            "visualize.py"
        ]

        missing_scripts = []
        for script in required_scripts:
            if not Path(script).exists():
                missing_scripts.append(script)

        if missing_scripts:
            logging.error(f"❌ Missing required scripts: {', '.join(missing_scripts)}")
            return False

        # Check for detect binary
        detect_binary = Path("resources/detect")
        if not detect_binary.exists():
            logging.warning(f"⚠️  Court detection binary not found at {detect_binary}")
            logging.warning("⚠️  Court detection may fail")

        logging.info("✅ All required scripts found")
        return True

    def print_summary(self) -> None:
        """Print pipeline execution summary"""
        logging.info("\n" + "="*60)
        logging.info("BPLFMV PIPELINE EXECUTION SUMMARY")
        logging.info("="*60)
        logging.info(f"Video: {self.video_path}")
        logging.info(f"Results directory: {self.results_dir}")
        logging.info("")

        stages = [
            ("Stage 1: Court Detection", self.court_csv),
            ("Stage 2: Pose Estimation", self.pose_json),
            ("Stage 3: Position Calculation", self.positions_json),
            ("Stage 4: Artifact Removal", self.corrected_positions_json),
            ("Stage 5: Final Visualization", self.final_visualization)
        ]

        for stage_name, output_file in stages:
            status = "✅ COMPLETED" if output_file.exists() else "❌ FAILED"
            file_size = ""
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                file_size = f" ({size_mb:.2f} MB)"

            logging.info(f"{stage_name:<35} {status}{file_size}")
            if output_file.exists():
                logging.info(f"{'':>37} → {output_file}")

        logging.info("="*60)

    def run_pipeline(self) -> bool:
        """Execute the complete BPLFMV pipeline"""
        pipeline_start_time = time.time()

        logging.info("🚀 Starting BPLFMV Pipeline...")
        logging.info(f"Target video: {self.video_path}")

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Execute stages sequentially
        stages = [
            ("Stage 1: Court Detection", self.stage1_detect_court),
            ("Stage 2: Pose Estimation", self.stage2_detect_pose),
            ("Stage 3: Position Calculation", self.stage3_calculate_location),
            ("Stage 4: Artifact Removal", self.stage4_remove_artifact),
            ("Stage 5: Final Visualization", self.stage5_visualize)
        ]

        completed_stages = 0

        for stage_name, stage_func in stages:
            logging.info(f"\n{'='*20} {stage_name} {'='*20}")

            stage_start_time = time.time()
            success = stage_func()
            stage_duration = time.time() - stage_start_time

            if success:
                completed_stages += 1
                logging.info(f"✅ {stage_name} completed in {stage_duration:.2f}s")
            else:
                logging.error(f"❌ {stage_name} failed after {stage_duration:.2f}s")
                logging.error("❌ Pipeline execution stopped due to failure")
                break

        # Calculate total execution time
        total_duration = time.time() - pipeline_start_time

        # Print summary
        self.print_summary()

        # Final status
        if completed_stages == len(stages):
            logging.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info(f"🎉 Total execution time: {total_duration:.2f}s")
            logging.info(f"🎉 Final output: {self.final_visualization}")
            return True
        else:
            logging.error(f"💥 PIPELINE FAILED after {completed_stages}/{len(stages)} stages")
            logging.error(f"💥 Total execution time: {total_duration:.2f}s")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="BPLFMV: Complete pipeline for badminton video analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py samples/ds1.mp4
  python3 pipeline.py videos/badminton_match.mp4
  
This pipeline will execute all stages sequentially:
1. Court detection
2. Pose estimation  
3. Position calculation
4. Jump artifact removal
5. Final visualization (corrected positions)

Output files will be saved in results/<video_name>/ directory.
        """
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input badminton video file"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all stages (ignore existing outputs)"
    )

    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize pipeline
        pipeline = BPLFMVPipeline(args.video_path)

        # Handle force flag by removing existing outputs
        if args.force:
            logging.info("🔄 Force flag enabled - removing existing outputs")
            for output_file in [pipeline.court_csv, pipeline.pose_json,
                                pipeline.positions_json, pipeline.corrected_positions_json,
                                pipeline.final_visualization]:
                if output_file.exists():
                    output_file.unlink()
                    logging.info(f"Removed: {output_file}")

        # Execute pipeline
        success = pipeline.run_pipeline()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        logging.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()