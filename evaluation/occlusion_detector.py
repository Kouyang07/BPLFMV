#!/usr/bin/env python3
"""
Badminton Court Occlusion Detector

This script analyzes badminton videos to detect frames where the court is occluded
by checking if 75% of the court is present using color sampling at court keypoints.

Usage: python3 occlusion_detector.py samples/occluded.mp4
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm


class FastCourtOcclusionDetector:
    def __init__(self, video_path, threshold=0.75):
        """
        Initialize the fast court occlusion detector.

        Args:
            video_path (str): Path to the video file
            threshold (float): Percentage of court that must be visible (0.75 = 75%)
        """
        self.video_path = video_path
        self.threshold = threshold
        self.court_points = {}
        self.reference_colors = {}
        self.reference_frame_idx = None

        # Color tolerance for matching
        self.color_tolerance = 30  # RGB tolerance

    def run_court_detection(self):
        """Run the court detection script to get court.csv and reference frame."""
        base_name = Path(self.video_path).stem
        result_dir = f"results/{base_name}"
        os.makedirs(result_dir, exist_ok=True)

        court_csv_path = os.path.join(result_dir, "court.csv")

        print("Running court detection...")
        while True:
            print("Attempting to detect court")
            result = subprocess.run(
                f'./resources/detect {self.video_path} {court_csv_path}',
                shell=True, capture_output=True, text=True
            )
            print(result.stdout)

            # Extract reference frame number from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "reference frame" in line.lower() or "frame" in line:
                    # Try to extract frame number
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            self.reference_frame_idx = int(word)
                            print(f"Reference frame detected: {self.reference_frame_idx}")
                            break

            if "Processing error: Not enough line candidates were found." not in result.stdout:
                break

        return court_csv_path

    def load_court_points(self, court_csv_path):
        """Load court keypoints from CSV file."""
        try:
            df = pd.read_csv(court_csv_path)
            for _, row in df.iterrows():
                self.court_points[row['Point']] = (int(row['X']), int(row['Y']))
            print(f"Loaded {len(self.court_points)} court points")
            return True
        except Exception as e:
            print(f"Error loading court points: {e}")
            return False

    def extract_reference_colors(self, reference_frame):
        """
        Extract reference colors at court keypoints from the reference frame.
        """
        print("Extracting reference colors from court keypoints...")

        for point_name, (x, y) in self.court_points.items():
            # Sample a small area around each keypoint for more robust color matching
            sample_size = 5  # 5x5 pixel area
            x1 = max(0, x - sample_size//2)
            y1 = max(0, y - sample_size//2)
            x2 = min(reference_frame.shape[1], x + sample_size//2 + 1)
            y2 = min(reference_frame.shape[0], y + sample_size//2 + 1)

            if x2 > x1 and y2 > y1:
                # Get average color in the sample area
                region = reference_frame[y1:y2, x1:x2]
                avg_color = np.mean(region.reshape(-1, 3), axis=0)
                self.reference_colors[point_name] = avg_color

        print(f"Extracted reference colors for {len(self.reference_colors)} points")

    def check_court_visibility(self, frame):
        """
        Check what percentage of court keypoints are visible based on color matching.

        Args:
            frame: Current video frame

        Returns:
            Tuple: (visibility_percentage, visible_points, total_points)
        """
        visible_points = 0
        total_points = len(self.reference_colors)

        for point_name, (x, y) in self.court_points.items():
            if point_name not in self.reference_colors:
                continue

            # Check if point is within frame bounds
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Sample color at keypoint
                sample_size = 5
                x1 = max(0, x - sample_size//2)
                y1 = max(0, y - sample_size//2)
                x2 = min(frame.shape[1], x + sample_size//2 + 1)
                y2 = min(frame.shape[0], y + sample_size//2 + 1)

                if x2 > x1 and y2 > y1:
                    # Get current color
                    region = frame[y1:y2, x1:x2]
                    current_color = np.mean(region.reshape(-1, 3), axis=0)

                    # Compare with reference color
                    reference_color = self.reference_colors[point_name]
                    color_diff = np.abs(current_color - reference_color)

                    # Check if colors are similar within tolerance
                    if np.all(color_diff <= self.color_tolerance):
                        visible_points += 1

        visibility_percentage = visible_points / total_points if total_points > 0 else 0
        return visibility_percentage, visible_points, total_points

    def process_video(self):
        """
        Process the entire video to detect occluded frames.
        """
        # First, run court detection
        court_csv_path = self.run_court_detection()

        if not self.load_court_points(court_csv_path):
            return None

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        # If no reference frame was detected, use frame 0
        if self.reference_frame_idx is None:
            self.reference_frame_idx = 0
            print("No reference frame detected from court detection, using frame 0")

        # Extract reference frame and colors
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.reference_frame_idx)
        ret, reference_frame = cap.read()
        if not ret:
            print("Error: Could not read reference frame")
            return None

        self.extract_reference_colors(reference_frame)

        # Reset to beginning of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        results = []
        frame_idx = 0

        print("Analyzing frames for court visibility...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check court visibility
            visibility_percentage, visible_points, total_points = self.check_court_visibility(frame)

            # Determine if frame is occluded
            is_occluded = visibility_percentage < self.threshold

            timestamp = frame_idx / fps if fps > 0 else frame_idx

            result = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'is_occluded': is_occluded,
                'visibility_percentage': visibility_percentage,
                'visible_points': visible_points,
                'total_points': total_points
            }

            results.append(result)

            # Print progress every 500 frames for speed
            if frame_idx % 500 == 0:
                print(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) - "
                      f"Visibility: {visibility_percentage:.2%}")

            frame_idx += 1

        cap.release()
        print(f"Processing complete. Analyzed {len(results)} frames.")
        return results

    def create_clean_video(self, results):
        """
        Create a new video containing only non-occluded frames at default location.

        Args:
            results: List of detection results
        """
        if not results:
            print("No results to process.")
            return None

        video_base = Path(self.video_path).stem
        output_video_path = f"results/{video_base}/{video_base}_clean.mp4"

        # Create output directory
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Get non-occluded frame indices
        non_occluded_frames = [r['frame_idx'] for r in results if not r['is_occluded']]

        if not non_occluded_frames:
            print("No non-occluded frames found. Cannot create clean video.")
            return None

        print(f"Creating clean video with {len(non_occluded_frames)} non-occluded frames...")

        # Open input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer with better compatibility
        # Try different codecs in order of preference
        codecs_to_try = [
            ('XVID', '.avi'),  # Most compatible
            ('mp4v', '.mp4'),  # Good compatibility
            ('H264', '.mp4'),  # High quality but may show warnings
        ]

        out = None
        for codec, ext in codecs_to_try:
            try:
                if ext != Path(output_video_path).suffix:
                    # Change extension to match codec
                    output_video_path = str(Path(output_video_path).with_suffix(ext))

                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                if out.isOpened():
                    print(f"Using {codec} codec for output video")
                    break
                else:
                    out.release()
                    out = None
            except:
                if out:
                    out.release()
                    out = None
                continue

        if out is None:
            # Fallback to default
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not create output video file {output_video_path}")
            cap.release()
            return None

        # Process frames with progress bar
        frames_written = 0
        with tqdm(total=len(non_occluded_frames), desc="Writing frames", unit="frame") as pbar:
            for frame_idx in non_occluded_frames:
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    out.write(frame)
                    frames_written += 1
                else:
                    print(f"Warning: Could not read frame {frame_idx}")

                pbar.update(1)

        # Clean up
        cap.release()
        out.release()

        print(f"Clean video created: {output_video_path}")
        print(f"Frames written: {frames_written}")

        # Calculate duration
        original_duration = len(results) / fps if fps > 0 else 0
        clean_duration = frames_written / fps if fps > 0 else 0

        print(f"Original duration: {original_duration:.2f}s")
        print(f"Clean duration: {clean_duration:.2f}s")
        print(f"Time reduction: {((original_duration - clean_duration) / original_duration * 100):.1f}%")

        return output_video_path

    def save_results(self, results, output_path=None):
        """Save detection results to a CSV file."""
        if not results:
            print("No results to save.")
            return

        if output_path is None:
            video_base = Path(self.video_path).stem
            output_path = f"results/{video_base}/occlusion_results.csv"

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Prepare data for CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        # Print summary
        occluded_frames = df[df['is_occluded'] == True]
        print(f"\nSummary:")
        print(f"Total frames analyzed: {len(df)}")
        print(f"Occluded frames detected: {len(occluded_frames)}")
        print(f"Occlusion percentage: {len(occluded_frames)/len(df)*100:.2f}%")
        print(f"Average court visibility: {df['visibility_percentage'].mean():.2%}")

        if len(occluded_frames) > 0:
            print(f"\nOccluded time periods (visibility < {self.threshold:.0%}):")

            # Group consecutive occluded frames
            occluded_indices = occluded_frames['frame_idx'].values
            if len(occluded_indices) > 0:
                groups = []
                current_group = [occluded_indices[0]]

                for i in range(1, len(occluded_indices)):
                    if occluded_indices[i] - occluded_indices[i-1] <= 5:  # Allow small gaps
                        current_group.append(occluded_indices[i])
                    else:
                        groups.append(current_group)
                        current_group = [occluded_indices[i]]
                groups.append(current_group)

                # Print grouped periods
                for group in groups:
                    start_frame = group[0]
                    end_frame = group[-1]
                    start_time = df[df['frame_idx'] == start_frame]['timestamp'].iloc[0]
                    end_time = df[df['frame_idx'] == end_frame]['timestamp'].iloc[0]
                    avg_visibility = df[df['frame_idx'].isin(group)]['visibility_percentage'].mean()

                    if len(group) == 1:
                        print(f"  Frame {start_frame}: {start_time:.2f}s (visibility: {avg_visibility:.1%})")
                    else:
                        print(f"  Frames {start_frame}-{end_frame}: {start_time:.2f}s-{end_time:.2f}s "
                              f"(avg visibility: {avg_visibility:.1%})")


def main():
    parser = argparse.ArgumentParser(description='Detect court occlusion in badminton videos using color sampling')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Court visibility threshold (0-1, default: 0.75 = 75%%)')
    parser.add_argument('--color-tolerance', type=int, default=30,
                        help='Color tolerance for matching (0-255, default: 30)')
    parser.add_argument('--output', help='Output CSV file path (auto-generated if not provided)')
    parser.add_argument('--create-clean-video', action='store_true',
                        help='Create a new video with only non-occluded frames')

    args = parser.parse_args()

    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Check if detect script exists
    if not os.path.exists('./resources/detect'):
        print("Error: ./resources/detect script not found")
        sys.exit(1)

    # Initialize detector
    detector = FastCourtOcclusionDetector(
        video_path=args.video_path,
        threshold=args.threshold
    )

    # Set color tolerance if specified
    if hasattr(args, 'color_tolerance'):
        detector.color_tolerance = args.color_tolerance

    # Process video
    results = detector.process_video()

    if results is None:
        print("Failed to process video.")
        sys.exit(1)

    # Save results
    detector.save_results(results, args.output)

    # Create clean video if requested
    if args.create_clean_video:
        print("\nCreating clean video with non-occluded frames...")
        clean_video_path = detector.create_clean_video(results)
        if clean_video_path:
            print(f"Clean video successfully created: {clean_video_path}")
        else:
            print("Failed to create clean video.")
    else:
        print("\nTo create a clean video with only non-occluded frames, use --create-clean-video flag")


if __name__ == "__main__":
    main()