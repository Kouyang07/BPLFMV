#!/usr/bin/env python3
"""
Badminton Tracking Comparison Visualizer

Creates a side-by-side visualization showing:
- Left: Original video frame
- Right: Court diagram with ground truth vs tracking positions

Usage: python3 visualizer.py <video_path>

Controls:
- Space: Play/Pause
- Left/Right arrows: Step frame by frame
- Q: Quit
- S: Save current frame comparison as image

Requirements:
    pip install opencv-python matplotlib numpy
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple, Optional


class TrackingVisualizer:
    """Interactive visualizer for comparing tracking results with ground truth."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.results_dir = Path("results") / self.video_name

        # Court dimensions (meters)
        self.court_width = 6.1
        self.court_length = 13.4

        # Data containers
        self.ground_truth = {}
        self.original_positions = {}
        self.corrected_positions = {}

        # Video properties
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        # Visualization state
        self.playing = False
        self.show_corrected = True
        self.show_trails = False
        self.trail_length = 10

        # Initialize
        self.load_data()
        self.setup_video()
        self.setup_visualization()

    def load_data(self):
        """Load all tracking and ground truth data."""
        # Load CVAT ground truth
        gt_file = self.results_dir / "cvat_ground_truth.json"
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                data = json.load(f)

            for pos in data.get('player_positions', []):
                frame_idx = pos['frame_index']
                player_id = pos['player_id']
                if frame_idx not in self.ground_truth:
                    self.ground_truth[frame_idx] = {}
                self.ground_truth[frame_idx][player_id] = (pos['world_x'], pos['world_y'])

            print(f"Loaded ground truth: {len(self.ground_truth)} frames")
        else:
            print("Warning: No ground truth file found")

        # Load original tracking
        orig_file = self.results_dir / "positions.json"
        if orig_file.exists():
            with open(orig_file, 'r') as f:
                data = json.load(f)

            frame_data = data.get('frame_data', {})
            for frame_str, players_data in frame_data.items():
                frame_idx = int(frame_str)
                if frame_idx not in self.original_positions:
                    self.original_positions[frame_idx] = {}

                for player_key, player_data in players_data.items():
                    player_id = int(player_key.split('_')[1])
                    self.original_positions[frame_idx][player_id] = (
                        player_data['center_position']['x'],
                        player_data['center_position']['y']
                    )

            print(f"Loaded original tracking: {len(self.original_positions)} frames")

        # Load corrected tracking
        corr_file = self.results_dir / "corrected_positions.json"
        if corr_file.exists():
            with open(corr_file, 'r') as f:
                data = json.load(f)

            frame_data = data.get('frame_data', {})
            for frame_str, players_data in frame_data.items():
                frame_idx = int(frame_str)
                if frame_idx not in self.corrected_positions:
                    self.corrected_positions[frame_idx] = {}

                for player_key, player_data in players_data.items():
                    player_id = int(player_key.split('_')[1])
                    self.corrected_positions[frame_idx][player_id] = (
                        player_data['center_position']['x'],
                        player_data['center_position']['y']
                    )

            print(f"Loaded corrected tracking: {len(self.corrected_positions)} frames")

    def setup_video(self):
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {self.total_frames} frames at {self.fps} fps")

    def setup_visualization(self):
        """Setup matplotlib visualization."""
        plt.style.use('default')
        self.fig, (self.ax_video, self.ax_court) = plt.subplots(1, 2, figsize=(16, 8))

        # Video subplot
        self.ax_video.set_title("Video Frame")
        self.ax_video.axis('off')

        # Court subplot
        self.ax_court.set_title("Court View - Ground Truth vs Tracking")
        self.ax_court.set_xlim(0, self.court_width)
        self.ax_court.set_ylim(0, self.court_length)
        self.ax_court.set_xlabel("Court Width (m)")
        self.ax_court.set_ylabel("Court Length (m)")
        self.ax_court.set_aspect('equal')
        self.ax_court.grid(True, alpha=0.3)

        # Draw court boundaries
        court_rect = patches.Rectangle((0, 0), self.court_width, self.court_length,
                                       linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3)
        self.ax_court.add_patch(court_rect)

        # Draw net
        net_y = self.court_length / 2
        self.ax_court.plot([0, self.court_width], [net_y, net_y], 'k-', linewidth=3, label='Net')

        # Initialize empty plots for positions
        self.gt_plot_p0, = self.ax_court.plot([], [], 'ro', markersize=12, label='GT Player 0', alpha=0.8)
        self.gt_plot_p1, = self.ax_court.plot([], [], 'rs', markersize=12, label='GT Player 1', alpha=0.8)
        self.orig_plot_p0, = self.ax_court.plot([], [], 'bo', markersize=10, label='Track Player 0', alpha=0.7)
        self.orig_plot_p1, = self.ax_court.plot([], [], 'bs', markersize=10, label='Track Player 1', alpha=0.7)
        self.corr_plot_p0, = self.ax_court.plot([], [], 'go', markersize=8, label='Corrected Player 0', alpha=0.6)
        self.corr_plot_p1, = self.ax_court.plot([], [], 'gs', markersize=8, label='Corrected Player 1', alpha=0.6)

        # Error lines
        self.error_lines = []

        # Legend
        self.ax_court.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Frame info text
        self.frame_text = self.ax_video.text(0.02, 0.98, '', transform=self.ax_video.transAxes,
                                             fontsize=12, verticalalignment='top',
                                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Error info text
        self.error_text = self.ax_court.text(0.02, 0.98, '', transform=self.ax_court.transAxes,
                                             fontsize=10, verticalalignment='top',
                                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get specific video frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def calculate_errors(self, frame_idx: int) -> Dict:
        """Calculate tracking errors for current frame."""
        errors = {}

        gt_data = self.ground_truth.get(frame_idx, {})
        orig_data = self.original_positions.get(frame_idx, {})
        corr_data = self.corrected_positions.get(frame_idx, {})

        for player_id in [0, 1]:
            if player_id in gt_data:
                gt_pos = gt_data[player_id]

                # Apply player mapping (from your evaluation results)
                # GT Player 0 -> Track Player 1, GT Player 1 -> Track Player 0
                track_player_id = 1 - player_id  # Swap mapping

                if track_player_id in orig_data:
                    orig_pos = orig_data[track_player_id]
                    orig_error = np.sqrt((gt_pos[0] - orig_pos[0])**2 + (gt_pos[1] - orig_pos[1])**2)
                    errors[f'original_p{player_id}'] = orig_error

                if track_player_id in corr_data:
                    corr_pos = corr_data[track_player_id]
                    corr_error = np.sqrt((gt_pos[0] - corr_pos[0])**2 + (gt_pos[1] - corr_pos[1])**2)
                    errors[f'corrected_p{player_id}'] = corr_error

        return errors

    def update_visualization(self, frame_idx: int):
        """Update the visualization for given frame."""
        # Clear previous error lines
        for line in self.error_lines:
            line.remove()
        self.error_lines.clear()

        # Get and display video frame
        frame = self.get_frame(frame_idx)
        if frame is not None:
            self.ax_video.clear()
            self.ax_video.imshow(frame)
            self.ax_video.axis('off')

            # Update frame info
            frame_info = f"Frame: {frame_idx}/{self.total_frames}\n"
            frame_info += f"Time: {frame_idx/self.fps:.2f}s"
            self.frame_text = self.ax_video.text(0.02, 0.98, frame_info, transform=self.ax_video.transAxes,
                                                 fontsize=12, verticalalignment='top',
                                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Get position data
        gt_data = self.ground_truth.get(frame_idx, {})
        orig_data = self.original_positions.get(frame_idx, {})
        corr_data = self.corrected_positions.get(frame_idx, {})

        # Apply player mapping (GT 0->Track 1, GT 1->Track 0)
        mapped_orig_data = {}
        mapped_corr_data = {}
        for gt_player_id in [0, 1]:
            track_player_id = 1 - gt_player_id  # Swap mapping
            if track_player_id in orig_data:
                mapped_orig_data[gt_player_id] = orig_data[track_player_id]
            if track_player_id in corr_data:
                mapped_corr_data[gt_player_id] = corr_data[track_player_id]

        # Update position plots
        for player_id in [0, 1]:
            # Ground truth
            if player_id in gt_data:
                gt_pos = gt_data[player_id]
                if player_id == 0:
                    self.gt_plot_p0.set_data([gt_pos[0]], [gt_pos[1]])
                else:
                    self.gt_plot_p1.set_data([gt_pos[0]], [gt_pos[1]])

                # Original tracking
                if player_id in mapped_orig_data:
                    orig_pos = mapped_orig_data[player_id]
                    if player_id == 0:
                        self.orig_plot_p0.set_data([orig_pos[0]], [orig_pos[1]])
                    else:
                        self.orig_plot_p1.set_data([orig_pos[0]], [orig_pos[1]])

                    # Draw error line
                    error_line = self.ax_court.plot([gt_pos[0], orig_pos[0]], [gt_pos[1], orig_pos[1]],
                                                    'r--', alpha=0.7, linewidth=2)[0]
                    self.error_lines.append(error_line)

                # Corrected tracking
                if self.show_corrected and player_id in mapped_corr_data:
                    corr_pos = mapped_corr_data[player_id]
                    if player_id == 0:
                        self.corr_plot_p0.set_data([corr_pos[0]], [corr_pos[1]])
                    else:
                        self.corr_plot_p1.set_data([corr_pos[0]], [corr_pos[1]])

                    # Draw corrected error line
                    error_line = self.ax_court.plot([gt_pos[0], corr_pos[0]], [gt_pos[1], corr_pos[1]],
                                                    'g:', alpha=0.7, linewidth=2)[0]
                    self.error_lines.append(error_line)
            else:
                # Clear plots if no data
                if player_id == 0:
                    self.gt_plot_p0.set_data([], [])
                    self.orig_plot_p0.set_data([], [])
                    self.corr_plot_p0.set_data([], [])
                else:
                    self.gt_plot_p1.set_data([], [])
                    self.orig_plot_p1.set_data([], [])
                    self.corr_plot_p1.set_data([], [])

        # Update error information
        errors = self.calculate_errors(frame_idx)
        error_info = f"Frame {frame_idx} Errors:\n"
        for player_id in [0, 1]:
            if f'original_p{player_id}' in errors:
                error_info += f"P{player_id} Orig: {errors[f'original_p{player_id}']:.3f}m\n"
            if f'corrected_p{player_id}' in errors:
                error_info += f"P{player_id} Corr: {errors[f'corrected_p{player_id}']:.3f}m\n"

        self.error_text.set_text(error_info)

        plt.draw()

    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == ' ':  # Space - play/pause
            self.playing = not self.playing
        elif event.key == 'right':  # Right arrow - next frame
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            self.update_visualization(self.current_frame)
        elif event.key == 'left':  # Left arrow - previous frame
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_visualization(self.current_frame)
        elif event.key == 'q':  # Q - quit
            plt.close('all')
            sys.exit(0)
        elif event.key == 's':  # S - save current frame
            self.save_current_frame()
        elif event.key == 'c':  # C - toggle corrected tracking
            self.show_corrected = not self.show_corrected
            self.update_visualization(self.current_frame)
        elif event.key == 't':  # T - toggle trails
            self.show_trails = not self.show_trails

    def save_current_frame(self):
        """Save current frame comparison as image."""
        filename = f"frame_comparison_{self.current_frame:06d}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    def animate(self, frame):
        """Animation function for automatic playback."""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.total_frames
            self.update_visualization(self.current_frame)
        return []

    def run(self):
        """Start the interactive visualization."""
        print("\nControls:")
        print("  Space: Play/Pause")
        print("  Left/Right arrows: Step frame by frame")
        print("  C: Toggle corrected tracking display")
        print("  S: Save current frame as image")
        print("  Q: Quit")
        print("\nStarting visualization...")

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial update
        self.update_visualization(self.current_frame)

        # Start animation (for auto-play when space is pressed)
        self.anim = FuncAnimation(self.fig, self.animate, interval=33, blit=False)  # ~30fps

        # Show the plot
        plt.show()

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Badminton Tracking Visualization Tool')
    parser.add_argument('video_path', help='Path to the video file')

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nBadminton Tracking Comparison Visualizer")
        print("=" * 45)
        print("Creates side-by-side visualization showing:")
        print("  • Left: Original video frame")
        print("  • Right: Court diagram with ground truth vs tracking positions")
        print("\nFeatures:")
        print("  • Interactive frame-by-frame stepping")
        print("  • Automatic player ID mapping")
        print("  • Error visualization with connecting lines")
        print("  • Real-time error calculations")
        print("  • Save frame comparisons as images")
        sys.exit(0)

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        visualizer = TrackingVisualizer(args.video_path)
        visualizer.run()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()