#!/usr/bin/env python3
"""
Badminton Player Position Manual Annotation Tool

GUI tool for manually annotating player positions in a top-down court view.
Uses the same coordinate system as the tracking scripts for evaluation purposes.
Supports saving/loading progress and exporting annotations for comparison.

Usage: python manual_annotation_tool.py <video_file_path>

Requirements:
    pip install opencv-python numpy tkinter pillow
"""

import sys
import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import time


class BadmintonCourtCanvas:
    """Draws a badminton court with proper dimensions and coordinate system."""

    # Court dimensions in meters (same as tracking scripts)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    # Court line specifications
    COURT_LINES = {
        'outer_boundary': [(0, 0), (COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH), (0, 0)],
        'center_line': [(COURT_WIDTH/2, 0), (COURT_WIDTH/2, COURT_LENGTH)],
        'service_lines': [
            # Short service lines
            [(0, 1.98), (COURT_WIDTH, 1.98)],
            [(0, COURT_LENGTH - 1.98), (COURT_WIDTH, COURT_LENGTH - 1.98)],
            # Long service lines (doubles)
            [(0, 2.59), (COURT_WIDTH, 2.59)],
            [(0, COURT_LENGTH - 2.59), (COURT_WIDTH, COURT_LENGTH - 2.59)]
        ],
        'side_lines': [
            # Singles sidelines
            [(0.46, 0), (0.46, COURT_LENGTH)],
            [(COURT_WIDTH - 0.46, 0), (COURT_WIDTH - 0.46, COURT_LENGTH)]
        ]
    }

    def __init__(self, canvas_width: int = 400, canvas_height: int = 600):
        """Initialize court canvas with given dimensions."""
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Calculate scaling and offsets to center court
        self.scale_x = (canvas_width - 40) / self.COURT_WIDTH
        self.scale_y = (canvas_height - 40) / self.COURT_LENGTH
        self.scale = min(self.scale_x, self.scale_y)

        self.offset_x = (canvas_width - self.COURT_WIDTH * self.scale) / 2
        self.offset_y = (canvas_height - self.COURT_LENGTH * self.scale) / 2

    def world_to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to canvas pixel coordinates."""
        canvas_x = int(x * self.scale + self.offset_x)
        canvas_y = int(y * self.scale + self.offset_y)
        return canvas_x, canvas_y

    def canvas_to_world(self, canvas_x: int, canvas_y: int) -> Tuple[float, float]:
        """Convert canvas pixel coordinates to world coordinates (meters)."""
        world_x = (canvas_x - self.offset_x) / self.scale
        world_y = (canvas_y - self.offset_y) / self.scale
        return world_x, world_y

    def create_court_image(self) -> Image.Image:
        """Create a PIL Image of the badminton court with labels."""
        # Create white background
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(img)

        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 10)
            small_font = ImageFont.truetype("arial.ttf", 8)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Draw court lines
        # Outer boundary
        boundary_points = [self.world_to_canvas(x, y) for x, y in self.COURT_LINES['outer_boundary']]
        draw.polygon(boundary_points, outline='black', width=3, fill=None)

        # Center line
        center_points = [self.world_to_canvas(x, y) for x, y in self.COURT_LINES['center_line']]
        draw.line(center_points, fill='black', width=2)

        # Service lines
        for line in self.COURT_LINES['service_lines']:
            line_points = [self.world_to_canvas(x, y) for x, y in line]
            draw.line(line_points, fill='black', width=1)

        # Side lines (singles)
        for line in self.COURT_LINES['side_lines']:
            line_points = [self.world_to_canvas(x, y) for x, y in line]
            draw.line(line_points, fill='gray', width=1)

        # Add labels
        labels = [
            ("Net", self.COURT_WIDTH/2, self.COURT_LENGTH/2, font),
            ("0,0", 0, 0, small_font),
            (f"{self.COURT_WIDTH:.1f},0", self.COURT_WIDTH, 0, small_font),
            (f"0,{self.COURT_LENGTH:.1f}", 0, self.COURT_LENGTH, small_font),
            (f"{self.COURT_WIDTH:.1f},{self.COURT_LENGTH:.1f}", self.COURT_WIDTH, self.COURT_LENGTH, small_font),
            ("Short Service", self.COURT_WIDTH/2, 1.98, small_font),
            ("Long Service", self.COURT_WIDTH/2, 2.59, small_font),
            ("Left Court", self.COURT_WIDTH/4, self.COURT_LENGTH/4, font),
            ("Right Court", 3*self.COURT_WIDTH/4, self.COURT_LENGTH/4, font),
        ]

        for text, x, y, text_font in labels:
            canvas_x, canvas_y = self.world_to_canvas(x, y)
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=text_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text
            text_x = canvas_x - text_width // 2
            text_y = canvas_y - text_height // 2

            draw.text((text_x, text_y), text, fill='blue', font=text_font)

        # Add coordinate grid
        for x in range(0, int(self.COURT_WIDTH) + 1):
            line_points = [self.world_to_canvas(x, 0), self.world_to_canvas(x, self.COURT_LENGTH)]
            draw.line(line_points, fill='lightgray', width=1)

        for y in range(0, int(self.COURT_LENGTH) + 1):
            line_points = [self.world_to_canvas(0, y), self.world_to_canvas(self.COURT_WIDTH, y)]
            draw.line(line_points, fill='lightgray', width=1)

        return img


class VideoPlayer:
    """Simple video player for frame navigation."""

    def __init__(self, video_path: str):
        """Initialize video player."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        print(f"Video loaded: {self.total_frames} frames at {self.fps} FPS")

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame from video."""
        if frame_number < 0 or frame_number >= self.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame_number
            return frame
        return None

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame."""
        return self.get_frame(self.current_frame)

    def close(self):
        """Close video capture."""
        self.cap.release()


class AnnotationTool:
    """Main annotation tool GUI."""

    def __init__(self, video_path: str):
        """Initialize annotation tool."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem

        # Initialize video player
        try:
            self.video_player = VideoPlayer(str(video_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not load video: {e}")
            sys.exit(1)

        # Annotation data
        self.annotations = {}  # frame_number -> {'player_0': (x, y), 'player_1': (x, y)}
        self.current_frame = 0
        self.current_player = 0  # 0 or 1

        # Results directory
        self.results_dir = Path("results") / self.video_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_file = self.results_dir / "manual_annotations.json"

        # GUI setup
        self.setup_gui()
        self.load_existing_annotations()
        self.update_display()

        # Auto-save timer
        self.auto_save_timer = None
        self.start_auto_save()

    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"Badminton Position Annotation - {self.video_name}")
        self.root.geometry("1200x800")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Video and controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video frame
        self.video_frame = ttk.Label(left_panel, text="Video will appear here")
        self.video_frame.pack(pady=5)

        # Video controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="<<", command=self.prev_frame_10).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="<", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text=">", command=self.next_frame).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text=">>", command=self.next_frame_10).pack(side=tk.LEFT)

        # Frame navigation
        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(fill=tk.X, pady=5)

        ttk.Label(nav_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_var = tk.StringVar(value="0")
        frame_entry = ttk.Entry(nav_frame, textvariable=self.frame_var, width=10)
        frame_entry.pack(side=tk.LEFT, padx=5)
        frame_entry.bind('<Return>', self.goto_frame)

        ttk.Button(nav_frame, text="Go", command=self.goto_frame).pack(side=tk.LEFT)

        self.frame_info_label = ttk.Label(nav_frame, text="")
        self.frame_info_label.pack(side=tk.LEFT, padx=10)

        # Right panel - Court and annotation controls
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Court canvas
        court_label = ttk.Label(right_panel, text="Badminton Court (Top View)")
        court_label.pack()

        self.court_canvas = tk.Canvas(right_panel, width=400, height=600, bg='white')
        self.court_canvas.pack(pady=5)
        self.court_canvas.bind('<Button-1>', self.on_court_click)

        # Player selection
        player_frame = ttk.LabelFrame(right_panel, text="Current Player")
        player_frame.pack(fill=tk.X, pady=5)

        self.player_var = tk.IntVar(value=0)
        ttk.Radiobutton(player_frame, text="Player 0 (Red)", variable=self.player_var,
                        value=0, command=self.update_court_display).pack(anchor=tk.W)
        ttk.Radiobutton(player_frame, text="Player 1 (Blue)", variable=self.player_var,
                        value=1, command=self.update_court_display).pack(anchor=tk.W)

        # Position info
        info_frame = ttk.LabelFrame(right_panel, text="Position Info")
        info_frame.pack(fill=tk.X, pady=5)

        self.position_info_label = ttk.Label(info_frame, text="No position selected", wraplength=200)
        self.position_info_label.pack(pady=5)

        # Annotation controls
        control_frame = ttk.LabelFrame(right_panel, text="Controls")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Clear Current Frame",
                   command=self.clear_current_frame).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Current Player",
                   command=self.clear_current_player).pack(fill=tk.X, pady=2)

        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT)
        self.stats_label.pack(pady=5)

        # File operations
        file_frame = ttk.LabelFrame(right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="Save Annotations",
                   command=self.save_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Annotations",
                   command=self.load_annotations_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export for Evaluation",
                   command=self.export_for_evaluation).pack(fill=tk.X, pady=2)

        # Initialize court
        self.court = BadmintonCourtCanvas(400, 600)
        self.update_court_display()

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<space>', lambda e: self.toggle_player())
        self.root.bind('<Control-s>', lambda e: self.save_annotations())
        self.root.focus_set()

    def update_display(self):
        """Update video frame and information display."""
        # Update video frame
        frame = self.video_player.get_frame(self.current_frame)
        if frame is not None:
            # Resize frame for display
            height, width = frame.shape[:2]
            display_width = 640
            display_height = int(height * display_width / width)

            frame_resized = cv2.resize(frame, (display_width, display_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to PIL and then to tkinter
            pil_image = Image.fromarray(frame_rgb)
            self.video_image = ImageTk.PhotoImage(pil_image)
            self.video_frame.configure(image=self.video_image)

        # Update frame info
        timestamp = self.current_frame / self.video_player.fps
        self.frame_var.set(str(self.current_frame))
        self.frame_info_label.config(
            text=f"/ {self.video_player.total_frames - 1} | Time: {timestamp:.2f}s"
        )

        # Update position info
        self.update_position_info()

        # Update court display
        self.update_court_display()

        # Update statistics
        self.update_statistics()

    def update_court_display(self):
        """Update the court canvas with current annotations."""
        # Clear canvas
        self.court_canvas.delete("all")

        # Draw court
        court_image = self.court.create_court_image()
        self.court_photo = ImageTk.PhotoImage(court_image)
        self.court_canvas.create_image(0, 0, anchor=tk.NW, image=self.court_photo)

        # Draw current frame annotations
        if self.current_frame in self.annotations:
            frame_annotations = self.annotations[self.current_frame]

            for player_id, (x, y) in frame_annotations.items():
                canvas_x, canvas_y = self.court.world_to_canvas(x, y)

                # Choose color based on player
                color = 'red' if player_id == 'player_0' else 'blue'

                # Draw player position
                radius = 8
                self.court_canvas.create_oval(
                    canvas_x - radius, canvas_y - radius,
                    canvas_x + radius, canvas_y + radius,
                    fill=color, outline='black', width=2
                )

                # Add player label
                self.court_canvas.create_text(
                    canvas_x, canvas_y - 15, text=player_id.split('_')[1],
                    fill='black', font=('Arial', 10, 'bold')
                )

        # Highlight current player selection
        current_player_text = f"Current: Player {self.player_var.get()}"
        color = 'red' if self.player_var.get() == 0 else 'blue'
        self.court_canvas.create_text(
            200, 20, text=current_player_text, fill=color,
            font=('Arial', 12, 'bold')
        )

    def update_position_info(self):
        """Update position information display."""
        if self.current_frame in self.annotations:
            frame_annotations = self.annotations[self.current_frame]

            info_text = f"Frame {self.current_frame} annotations:\n"
            for player_id, (x, y) in frame_annotations.items():
                info_text += f"{player_id}: ({x:.2f}, {y:.2f})\n"
        else:
            info_text = f"Frame {self.current_frame}: No annotations"

        self.position_info_label.config(text=info_text)

    def update_statistics(self):
        """Update annotation statistics."""
        total_frames = len(self.annotations)
        player_0_frames = len([f for f, ann in self.annotations.items() if 'player_0' in ann])
        player_1_frames = len([f for f, ann in self.annotations.items() if 'player_1' in ann])

        stats_text = f"Annotated frames: {total_frames}\n"
        stats_text += f"Player 0 positions: {player_0_frames}\n"
        stats_text += f"Player 1 positions: {player_1_frames}\n"

        # Coverage percentage
        coverage = (total_frames / self.video_player.total_frames) * 100 if self.video_player.total_frames > 0 else 0
        stats_text += f"Coverage: {coverage:.1f}%"

        self.stats_label.config(text=stats_text)

    def on_court_click(self, event):
        """Handle click on court canvas."""
        # Convert canvas coordinates to world coordinates
        world_x, world_y = self.court.canvas_to_world(event.x, event.y)

        # Check if click is within court bounds
        if 0 <= world_x <= self.court.COURT_WIDTH and 0 <= world_y <= self.court.COURT_LENGTH:
            # Add annotation for current player
            current_player = self.player_var.get()
            player_key = f"player_{current_player}"

            if self.current_frame not in self.annotations:
                self.annotations[self.current_frame] = {}

            self.annotations[self.current_frame][player_key] = (world_x, world_y)

            print(f"Annotated Frame {self.current_frame}, {player_key}: ({world_x:.2f}, {world_y:.2f})")

            # Update display
            self.update_display()

    def clear_current_frame(self):
        """Clear all annotations for current frame."""
        if self.current_frame in self.annotations:
            del self.annotations[self.current_frame]
            self.update_display()
            print(f"Cleared annotations for frame {self.current_frame}")

    def clear_current_player(self):
        """Clear annotation for current player in current frame."""
        current_player = self.player_var.get()
        player_key = f"player_{current_player}"

        if (self.current_frame in self.annotations and
                player_key in self.annotations[self.current_frame]):
            del self.annotations[self.current_frame][player_key]

            # Remove frame entry if no players left
            if not self.annotations[self.current_frame]:
                del self.annotations[self.current_frame]

            self.update_display()
            print(f"Cleared {player_key} annotation for frame {self.current_frame}")

    def toggle_player(self):
        """Toggle between player 0 and player 1."""
        self.player_var.set(1 - self.player_var.get())
        self.update_court_display()

    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.video_player.total_frames - 1:
            self.current_frame += 1
            self.update_display()

    def prev_frame_10(self):
        """Go back 10 frames."""
        self.current_frame = max(0, self.current_frame - 10)
        self.update_display()

    def next_frame_10(self):
        """Go forward 10 frames."""
        self.current_frame = min(self.video_player.total_frames - 1, self.current_frame + 10)
        self.update_display()

    def goto_frame(self, event=None):
        """Go to specific frame."""
        try:
            frame_num = int(self.frame_var.get())
            if 0 <= frame_num < self.video_player.total_frames:
                self.current_frame = frame_num
                self.update_display()
            else:
                messagebox.showerror("Error", f"Frame must be between 0 and {self.video_player.total_frames - 1}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number")

    def save_annotations(self):
        """Save annotations to file."""
        try:
            # Convert to serializable format
            save_data = {
                'video_path': str(self.video_path),
                'video_name': self.video_name,
                'total_frames': self.video_player.total_frames,
                'fps': self.video_player.fps,
                'court_dimensions': {
                    'width': self.court.COURT_WIDTH,
                    'length': self.court.COURT_LENGTH
                },
                'annotations': self.annotations,
                'annotation_count': {
                    'total_frames': len(self.annotations),
                    'player_0_positions': len([f for f, ann in self.annotations.items() if 'player_0' in ann]),
                    'player_1_positions': len([f for f, ann in self.annotations.items() if 'player_1' in ann])
                }
            }

            with open(self.annotation_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"Annotations saved to: {self.annotation_file}")
            messagebox.showinfo("Success", f"Annotations saved to:\n{self.annotation_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")

    def load_existing_annotations(self):
        """Load existing annotations if they exist."""
        if self.annotation_file.exists():
            try:
                with open(self.annotation_file, 'r') as f:
                    data = json.load(f)

                self.annotations = data.get('annotations', {})
                # Convert string keys back to integers
                self.annotations = {int(k): v for k, v in self.annotations.items()}

                print(f"Loaded {len(self.annotations)} annotated frames")

            except Exception as e:
                print(f"Warning: Could not load existing annotations: {e}")

    def load_annotations_dialog(self):
        """Load annotations from file dialog."""
        file_path = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                self.annotations = data.get('annotations', {})
                # Convert string keys back to integers
                self.annotations = {int(k): v for k, v in self.annotations.items()}

                self.update_display()
                messagebox.showinfo("Success", f"Loaded {len(self.annotations)} annotated frames")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations: {e}")

    def export_for_evaluation(self):
        """Export annotations in format compatible with tracking scripts."""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to export")
            return

        try:
            # Convert to evaluation format (similar to positions.json)
            player_positions = []

            for frame_index, frame_annotations in self.annotations.items():
                for player_key, (x, y) in frame_annotations.items():
                    player_id = int(player_key.split('_')[1])

                    position_entry = {
                        'frame_index': frame_index,
                        'player_id': player_id,
                        'hip_world_X': x,
                        'hip_world_Y': y,
                        'left_ankle_world_X': x,  # Same as hip for ground truth
                        'left_ankle_world_Y': y,
                        'right_ankle_world_X': x,
                        'right_ankle_world_Y': y,
                        'annotation_type': 'manual_ground_truth'
                    }

                    player_positions.append(position_entry)

            # Sort by frame and player
            player_positions.sort(key=lambda x: (x['frame_index'], x['player_id']))

            # Create export data
            export_data = {
                'video_info': {
                    'path': str(self.video_path),
                    'name': self.video_name,
                    'total_frames': self.video_player.total_frames,
                    'fps': self.video_player.fps
                },
                'court_dimensions': {
                    'width': self.court.COURT_WIDTH,
                    'length': self.court.COURT_LENGTH
                },
                'player_positions': player_positions,
                'annotation_method': 'manual_ground_truth',
                'annotation_info': {
                    'total_annotated_frames': len(self.annotations),
                    'total_positions': len(player_positions),
                    'annotation_coverage': len(self.annotations) / self.video_player.total_frames
                }
            }

            # Save export file
            export_file = self.results_dir / "ground_truth_positions.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            messagebox.showinfo("Success", f"Ground truth exported to:\n{export_file}")
            print(f"Exported {len(player_positions)} positions to: {export_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def start_auto_save(self):
        """Start auto-save timer."""
        def auto_save():
            if self.annotations:  # Only save if there are annotations
                try:
                    self.save_annotations()
                except:
                    pass  # Ignore auto-save errors

            # Schedule next auto-save
            self.auto_save_timer = threading.Timer(60.0, auto_save)  # Every 60 seconds
            self.auto_save_timer.start()

        # Start the auto-save
        auto_save()

    def stop_auto_save(self):
        """Stop auto-save timer."""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()

    def on_closing(self):
        """Handle application closing."""
        # Save annotations before closing
        if self.annotations:
            result = messagebox.askyesnocancel("Save", "Save annotations before closing?")
            if result is True:
                self.save_annotations()
            elif result is None:  # Cancel
                return

        # Clean up
        self.stop_auto_save()
        self.video_player.close()
        self.root.destroy()

    def run(self):
        """Run the annotation tool."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python manual_annotation_tool.py <video_file_path>")
        print("\nThis tool allows you to manually annotate player positions for evaluation.")
        print("Controls:")
        print("  - Click on court to place current player")
        print("  - Use arrow keys or buttons to navigate frames")
        print("  - Space bar to switch between players")
        print("  - Ctrl+S to save annotations")
        print("  - Auto-save every 60 seconds")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Starting annotation tool for: {video_path}")
    print("\nInstructions:")
    print("1. Use the video controls to navigate to frames with clear player positions")
    print("2. Select the current player (0 or 1) using radio buttons")
    print("3. Click on the court where the player is positioned")
    print("4. Annotations are auto-saved every 60 seconds")
    print("5. Use 'Export for Evaluation' to create ground truth data")
    print("\nKeyboard shortcuts:")
    print("  Left/Right arrows: Navigate frames")
    print("  Spacebar: Switch between players")
    print("  Ctrl+S: Save annotations")

    try:
        tool = AnnotationTool(video_path)
        tool.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()