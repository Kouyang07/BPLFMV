#!/usr/bin/env python3
"""
Video-Click Badminton Player Position Annotation Tool - Two Player Version
Modified to work with the standard pipeline outputs

GUI tool for annotating both player positions by clicking directly on the video frame.
Converts pixel coordinates to world coordinates using homography from court detection.
Supports annotating both Player 1 and Player 2 for evaluation purposes.

Usage: python video_annotation_tool.py <video_file_path>

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
from PIL import Image, ImageTk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import time


class VideoClickAnnotator:
    """Annotation tool that converts video clicks to world coordinates for both players."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    def __init__(self, video_path: str):
        """Initialize the video click annotator."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem

        # Look for results in standard pipeline location first
        self.results_dir = Path("results") / self.video_name
        if not self.results_dir.exists():
            # Fallback to current directory structure
            self.results_dir = Path(self.video_name)

        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # File paths to match pipeline outputs
        self.pose_file = self.results_dir / "pose.json"
        self.calibration_file = self.results_dir / "calibration.csv"
        self.detections_file = self.results_dir / "detections.csv"
        self.annotation_file = self.results_dir / "video_click_annotations.json"

        # Load court detection data for coordinate transformation
        self.court_points = None
        self.homography_matrix = None
        self.inverse_homography = None
        self.load_court_data()

        # Video player
        self.video_player = None
        self.init_video_player()

        # Annotation data - supports both players
        self.annotations = {}  # frame_number -> {player_id: (world_x, world_y)}
        self.current_frame = 0
        self.current_player = 1  # Currently selected player (1 or 2)

        # GUI elements
        self.root = None
        self.video_canvas = None
        self.video_image = None
        self.display_scale = 1.0

        # Auto-save
        self.auto_save_timer = None

        # Player colors for visualization
        self.player_colors = {
            1: (255, 100, 100),  # Light red for Player 1
            2: (100, 100, 255)   # Light blue for Player 2
        }

    def load_court_data(self):
        """Load court points and calculate homography for coordinate transformation."""
        # Try to load from pose.json first (standard pipeline output)
        if self.pose_file.exists():
            self.load_from_pose_json()
        # Fallback to separate files if pose.json doesn't exist
        elif self.detections_file.exists():
            self.load_from_detections_csv()
        else:
            raise FileNotFoundError(f"Court detection data not found. Expected: {self.pose_file} or {self.detections_file}")

        # Calculate homography matrix
        self.calculate_homography()

        print("Court coordinate system loaded:")
        print("  P1: Upper baseline + LEFT sideline")
        print("  P2: Lower baseline + LEFT sideline")
        print("  P3: Lower baseline + RIGHT sideline")
        print("  P4: Upper baseline + RIGHT sideline")

    def load_from_pose_json(self):
        """Load court points from pose.json (standard pipeline output)."""
        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        # Try different possible locations for court points in pose.json
        self.court_points = None

        if 'court_points' in data:
            self.court_points = data['court_points']
        elif 'all_court_points' in data:
            self.court_points = data['all_court_points']
        elif 'enlarged_court_points' in data:
            if 'court_points' in data:
                self.court_points = data['court_points']
            else:
                self.court_points = data['enlarged_court_points']

        if not self.court_points:
            raise ValueError("No court points found in pose.json")

        # Ensure we have the required corner points
        required_points = ['P1', 'P2', 'P3', 'P4']
        missing_points = [p for p in required_points if p not in self.court_points]

        if missing_points:
            # If we don't have P1-P4, try to extract them from available points
            if len(self.court_points) >= 4:
                print(f"Warning: Missing {missing_points}, using first 4 available points")
                available_points = list(self.court_points.items())[:4]
                self.court_points = {f'P{i+1}': coords for i, (_, coords) in enumerate(available_points)}
            else:
                raise ValueError(f"Missing required court points: {missing_points}")

        print(f"Loaded court points from pose.json: {list(self.court_points.keys())}")

    def load_from_detections_csv(self):
        """Load court points from detections.csv (fallback method)."""
        import csv

        self.court_points = {}

        with open(self.detections_file, 'r') as f:
            first_line = f.readline().strip()
            f.seek(0)

            if 'Point' in first_line and 'X' in first_line:
                # Header format
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        point_name = row['Point'].strip()
                        x = float(row['X'])
                        y = float(row['Y'])
                        self.court_points[point_name] = [x, y]
                    except (ValueError, KeyError):
                        continue
            else:
                # No header format
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            point_name = row[0].strip()
                            x = float(row[1])
                            y = float(row[2])
                            self.court_points[point_name] = [x, y]
                        except (ValueError, IndexError):
                            continue

        if not self.court_points:
            raise ValueError("No court points loaded from detections.csv")

        print(f"Loaded court points from detections.csv: {list(self.court_points.keys())}")

    def normalize_point_coordinates(self, point_coords):
        """Normalize point coordinates to consistent format [x, y]."""
        if isinstance(point_coords, list) and len(point_coords) >= 2:
            return [float(point_coords[0]), float(point_coords[1])]
        elif isinstance(point_coords, dict):
            if 'x' in point_coords and 'y' in point_coords:
                return [float(point_coords['x']), float(point_coords['y'])]
            elif 0 in point_coords and 1 in point_coords:
                return [float(point_coords[0]), float(point_coords[1])]

        raise ValueError(f"Invalid point coordinate format: {point_coords}")

    def calculate_homography(self):
        """Calculate homography matrix from video pixels to world coordinates."""
        # Required corner points for homography
        required_corners = ['P1', 'P2', 'P3', 'P4']

        # Check if we have all required points
        missing_corners = [corner for corner in required_corners if corner not in self.court_points]
        if missing_corners:
            raise ValueError(f"Missing required court corners: {missing_corners}")

        # Extract image points (pixels) from court detection
        image_points = []
        for corner in required_corners:
            coords = self.court_points[corner]
            normalized_coords = self.normalize_point_coordinates(coords)
            image_points.append(normalized_coords)

        image_points = np.array(image_points, dtype=np.float32)

        # Corresponding world coordinates (meters)
        world_points = np.array([
            [0, 0],                      # P1: Left side, top
            [0, self.COURT_LENGTH],      # P2: Left side, bottom
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Right side, bottom
            [self.COURT_WIDTH, 0]        # P4: Right side, top
        ], dtype=np.float32)

        # Calculate homography matrix
        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography matrix")

        # Also calculate inverse for displaying court overlay
        self.inverse_homography = np.linalg.inv(self.homography_matrix)

        print("Homography matrix calculated successfully")

    def init_video_player(self):
        """Initialize video capture."""
        self.video_player = cv2.VideoCapture(str(self.video_path))

        if not self.video_player.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.total_frames = int(self.video_player.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_player.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.video_player.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_player.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {self.total_frames} frames at {self.fps:.2f} FPS")
        print(f"Resolution: {self.video_width}x{self.video_height}")

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert video pixel coordinates to world coordinates."""
        # Adjust for display scaling
        actual_x = pixel_x / self.display_scale
        actual_y = pixel_y / self.display_scale

        # Transform using homography
        point = np.array([[actual_x, actual_y]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.homography_matrix)

        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates to video pixel coordinates."""
        point = np.array([[world_x, world_y]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.inverse_homography)

        # Adjust for display scaling
        display_x = pixel_point[0][0][0] * self.display_scale
        display_y = pixel_point[0][0][1] * self.display_scale

        return float(display_x), float(display_y)

    def world_to_pixel_original(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates to original video pixel coordinates (before scaling)."""
        point = np.array([[world_x, world_y]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.inverse_homography)
        return float(pixel_point[0][0][0]), float(pixel_point[0][0][1])

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current video frame."""
        self.video_player.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_player.read()
        return frame if ret else None

    def setup_gui(self):
        """Setup the main GUI."""
        self.root = tk.Tk()
        self.root.title(f"Two-Player Video Annotation - {self.video_name}")
        self.root.geometry("1400x900")
        try:
            self.root.state('zoomed')  # Start maximized on Windows
        except:
            pass  # Ignore if not available on this platform

        # Configure main grid - give video much more space
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=4)  # Video gets 4x more space
        self.root.grid_columnconfigure(1, weight=1)  # Controls get 1x space

        # Left panel - Video (takes up most of the screen)
        left_panel = ttk.Frame(self.root)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)

        # Configure left panel grid weights
        left_panel.grid_rowconfigure(1, weight=1)  # Video gets all the space
        left_panel.grid_rowconfigure(0, weight=0)  # Title fixed size
        left_panel.grid_rowconfigure(2, weight=0)  # Controls fixed size
        left_panel.grid_rowconfigure(3, weight=0)  # Zoom fixed size
        left_panel.grid_rowconfigure(4, weight=0)  # Info fixed size
        left_panel.grid_columnconfigure(0, weight=1)

        # Video title
        video_title = ttk.Label(left_panel, text="Click on video to mark player positions",
                                font=('Arial', 11, 'bold'))
        video_title.grid(row=0, column=0, pady=(0, 5), sticky="ew")

        # Video canvas container
        video_container = ttk.Frame(left_panel)
        video_container.grid(row=1, column=0, sticky="nsew", pady=2)
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        # Video canvas with scrollbars
        self.video_canvas = tk.Canvas(video_container, bg='black', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(video_container, orient="vertical", command=self.video_canvas.yview)
        h_scrollbar = ttk.Scrollbar(video_container, orient="horizontal", command=self.video_canvas.xview)

        # Configure canvas scrolling
        self.video_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid the canvas and scrollbars
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Bind events to canvas
        self.video_canvas.bind('<Button-1>', self.on_video_click)
        self.video_canvas.bind('<Motion>', self.on_mouse_move)
        self.video_canvas.bind('<Enter>', lambda e: self.video_canvas.focus_set())
        self.video_canvas.focus_set()

        # Video controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.grid(row=2, column=0, sticky="ew", pady=3)
        controls_frame.grid_columnconfigure(2, weight=1)

        # Navigation buttons
        btn_prev_10 = ttk.Button(controls_frame, text="◀◀", width=4, command=self.prev_frame_10)
        btn_prev_10.grid(row=0, column=0, padx=2)

        btn_prev = ttk.Button(controls_frame, text="◀", width=4, command=self.prev_frame)
        btn_prev.grid(row=0, column=1, padx=2)

        btn_next = ttk.Button(controls_frame, text="▶", width=4, command=self.next_frame)
        btn_next.grid(row=0, column=3, padx=2)

        btn_next_10 = ttk.Button(controls_frame, text="▶▶", width=4, command=self.next_frame_10)
        btn_next_10.grid(row=0, column=4, padx=2)

        # Frame entry in center
        frame_controls = ttk.Frame(controls_frame)
        frame_controls.grid(row=0, column=2, padx=10, sticky="ew")
        frame_controls.grid_columnconfigure(1, weight=1)

        ttk.Label(frame_controls, text="Frame:").grid(row=0, column=0, sticky="w")

        self.frame_var = tk.StringVar(value="0")
        frame_entry = ttk.Entry(frame_controls, textvariable=self.frame_var, width=10)
        frame_entry.grid(row=0, column=1, padx=5, sticky="ew")
        frame_entry.bind('<Return>', self.goto_frame)
        frame_entry.bind('<KeyRelease>', self.on_frame_entry_change)

        btn_go = ttk.Button(frame_controls, text="Go", command=self.goto_frame)
        btn_go.grid(row=0, column=2, padx=2)

        # Zoom controls
        zoom_frame = ttk.Frame(left_panel)
        zoom_frame.grid(row=3, column=0, sticky="ew", pady=2)
        zoom_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(zoom_frame, text="Zoom:").grid(row=0, column=0, sticky="w")

        self.scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(zoom_frame, from_=0.2, to=3.0, variable=self.scale_var,
                                 orient=tk.HORIZONTAL, command=self.on_scale_change)
        scale_slider.grid(row=0, column=1, sticky="ew", padx=5)

        btn_100 = ttk.Button(zoom_frame, text="100%", width=6, command=lambda: self.set_scale(1.0))
        btn_100.grid(row=0, column=2, padx=2)

        btn_fit = ttk.Button(zoom_frame, text="Fit", width=6, command=self.fit_to_window)
        btn_fit.grid(row=0, column=3, padx=2)

        # Status info
        status_frame = ttk.Frame(left_panel)
        status_frame.grid(row=4, column=0, sticky="ew", pady=2)
        status_frame.grid_columnconfigure(0, weight=1)

        self.frame_info_label = ttk.Label(status_frame, text="", font=('Arial', 9))
        self.frame_info_label.grid(row=0, column=0, sticky="w")

        self.mouse_info_label = ttk.Label(status_frame, text="", font=('Arial', 9))
        self.mouse_info_label.grid(row=0, column=1, sticky="e")

        # Right panel - Controls
        right_panel = ttk.Frame(self.root)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)

        # Data source info
        info_frame = ttk.LabelFrame(right_panel, text="Data Sources")
        info_frame.pack(fill=tk.X, pady=(0, 5))

        # Display info about loaded data sources
        source_info = f"Results dir: {self.results_dir}\n"
        if self.pose_file.exists():
            source_info += "✓ pose.json\n"
        if self.calibration_file.exists():
            source_info += "✓ calibration.csv\n"
        if self.detections_file.exists():
            source_info += "✓ detections.csv"

        source_label = ttk.Label(info_frame, text=source_info, font=('Arial', 8))
        source_label.pack(pady=3)

        # Player selection
        player_frame = ttk.LabelFrame(right_panel, text="Player Selection")
        player_frame.pack(fill=tk.X, pady=(0, 5))

        self.player_var = tk.IntVar(value=1)
        player1_radio = ttk.Radiobutton(player_frame, text="Player 1 (Red)",
                                        variable=self.player_var, value=1,
                                        command=self.on_player_change)
        player1_radio.pack(anchor=tk.W, padx=5, pady=2)

        player2_radio = ttk.Radiobutton(player_frame, text="Player 2 (Blue)",
                                        variable=self.player_var, value=2,
                                        command=self.on_player_change)
        player2_radio.pack(anchor=tk.W, padx=5, pady=2)

        # Current annotation info
        current_frame = ttk.LabelFrame(right_panel, text="Current Positions")
        current_frame.pack(fill=tk.X, pady=(0, 5))

        self.position_info_label = ttk.Label(current_frame, text="No positions marked",
                                             wraplength=280, justify=tk.LEFT, font=('Arial', 9))
        self.position_info_label.pack(pady=8)

        # Court visualization
        court_frame = ttk.LabelFrame(right_panel, text="Court Reference")
        court_frame.pack(fill=tk.X, pady=(0, 5))

        self.court_canvas = tk.Canvas(court_frame, width=160, height=220, bg='white')
        self.court_canvas.pack(pady=5)

        # Controls
        control_frame = ttk.LabelFrame(right_panel, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # Clear buttons
        clear_frame = ttk.Frame(control_frame)
        clear_frame.pack(fill=tk.X, pady=2)

        btn_clear_current = ttk.Button(clear_frame, text="Clear Current Player", command=self.clear_current_player)
        btn_clear_current.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))

        btn_clear_all = ttk.Button(clear_frame, text="Clear All", command=self.clear_current_frame)
        btn_clear_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=2)

        btn_prev_ann = ttk.Button(nav_frame, text="← Previous", command=self.goto_prev_annotation)
        btn_prev_ann.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))

        btn_next_ann = ttk.Button(nav_frame, text="Next →", command=self.goto_next_annotation)
        btn_next_ann.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT, font=('Arial', 9))
        self.stats_label.pack(pady=5)

        # File operations
        file_frame = ttk.LabelFrame(right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=(0, 5))

        btn_save = ttk.Button(file_frame, text="💾 Save", command=self.save_annotations)
        btn_save.pack(fill=tk.X, pady=1)

        btn_load = ttk.Button(file_frame, text="📁 Load", command=self.load_annotations_dialog)
        btn_load.pack(fill=tk.X, pady=1)

        btn_export = ttk.Button(file_frame, text="📤 Export", command=self.export_for_evaluation)
        btn_export.pack(fill=tk.X, pady=1)

        # Keyboard bindings
        self.video_canvas.bind('<Left>', lambda e: self.prev_frame())
        self.video_canvas.bind('<Right>', lambda e: self.next_frame())
        self.video_canvas.bind('<Control-s>', lambda e: self.save_annotations())
        self.video_canvas.bind('<Delete>', lambda e: self.clear_current_player())
        self.video_canvas.bind('<BackSpace>', lambda e: self.clear_current_player())
        self.video_canvas.bind('<1>', lambda e: self.set_current_player(1))
        self.video_canvas.bind('<2>', lambda e: self.set_current_player(2))

        # Also bind to root for when other widgets have focus
        self.root.bind('<Control-s>', lambda e: self.save_annotations())

        # Initialize displays
        self.load_existing_annotations()
        self.update_display()

        # Test coordinate transformation
        self.debug_coordinate_system()

        # Set initial focus to canvas
        self.root.after(100, lambda: self.video_canvas.focus_set())

    def debug_coordinate_system(self):
        """Debug the coordinate transformation system."""
        print("\n=== COORDINATE SYSTEM DEBUG ===")
        print(f"Court points loaded: {list(self.court_points.keys())}")

        # Print actual court point coordinates
        for corner in ['P1', 'P2', 'P3', 'P4']:
            if corner in self.court_points:
                coords = self.normalize_point_coordinates(self.court_points[corner])
                print(f"{corner}: {coords}")

        print(f"Homography matrix exists: {self.homography_matrix is not None}")
        if self.homography_matrix is not None:
            print(f"Homography matrix shape: {self.homography_matrix.shape}")

        # Test transformation of corner points
        print("\nTesting corner transformations:")
        for corner in ['P1', 'P2', 'P3', 'P4']:
            if corner in self.court_points:
                coords = self.normalize_point_coordinates(self.court_points[corner])
                try:
                    world_x, world_y = self.pixel_to_world(coords[0] * self.display_scale, coords[1] * self.display_scale)
                    print(f"{corner}: pixel {coords} -> world ({world_x:.2f}, {world_y:.2f})")
                except Exception as e:
                    print(f"{corner}: FAILED - {e}")

        print("================================\n")

    def on_player_change(self):
        """Handle player selection change."""
        self.current_player = self.player_var.get()
        self.update_display()

    def set_current_player(self, player_id):
        """Set current player via keyboard shortcut."""
        self.player_var.set(player_id)
        self.current_player = player_id
        self.update_display()

    def on_video_click(self, event):
        """Handle clicks on the video canvas."""
        # Get click coordinates relative to the canvas scroll region
        canvas_x = self.video_canvas.canvasx(event.x)
        canvas_y = self.video_canvas.canvasy(event.y)

        try:
            # Convert to world coordinates (accounting for display scaling)
            world_x, world_y = self.pixel_to_world(canvas_x, canvas_y)

            print(f"Click at canvas ({canvas_x:.1f}, {canvas_y:.1f}) -> world ({world_x:.2f}, {world_y:.2f})")

            # Check if the click is within the court bounds (with some tolerance)
            margin = 0.5  # Allow clicks slightly outside court
            if (-margin <= world_x <= self.COURT_WIDTH + margin and
                    -margin <= world_y <= self.COURT_LENGTH + margin):

                # Clamp coordinates to court bounds
                world_x = max(0, min(self.COURT_WIDTH, world_x))
                world_y = max(0, min(self.COURT_LENGTH, world_y))

                # Initialize frame annotations if needed
                if self.current_frame not in self.annotations:
                    self.annotations[self.current_frame] = {}

                # Store annotation for current player
                self.annotations[self.current_frame][self.current_player] = (world_x, world_y)

                print(f"Frame {self.current_frame}: Player {self.current_player} at ({world_x:.2f}, {world_y:.2f})")

                # Update displays
                self.update_display()

                # Auto-advance logic
                frame_annotations = self.annotations.get(self.current_frame, {})
                if len(frame_annotations) >= 2 and self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                    self.update_display()
                elif self.current_player in frame_annotations and self.current_frame < self.total_frames - 1:
                    other_player = 2 if self.current_player == 1 else 1
                    if other_player not in frame_annotations:
                        self.player_var.set(other_player)
                        self.current_player = other_player
                        self.update_display()
                    else:
                        self.current_frame += 1
                        self.update_display()
            else:
                # Click outside court
                messagebox.showwarning("Invalid Position",
                                       f"Click is outside court bounds.\n"
                                       f"Clicked: ({world_x:.2f}, {world_y:.2f})\n"
                                       f"Court bounds: (0, 0) to ({self.COURT_WIDTH}, {self.COURT_LENGTH})")

        except Exception as e:
            print(f"Click conversion error: {e}")
            messagebox.showerror("Error", f"Failed to convert coordinates: {e}")

    def on_mouse_move(self, event):
        """Show mouse position in world coordinates."""
        try:
            canvas_x = self.video_canvas.canvasx(event.x)
            canvas_y = self.video_canvas.canvasy(event.y)

            world_x, world_y = self.pixel_to_world(canvas_x, canvas_y)

            if 0 <= world_x <= self.COURT_WIDTH and 0 <= world_y <= self.COURT_LENGTH:
                self.mouse_info_label.config(
                    text=f"Court: ({world_x:.2f}, {world_y:.2f}) | Player {self.current_player}"
                )
            else:
                self.mouse_info_label.config(
                    text=f"Outside: ({world_x:.2f}, {world_y:.2f}) | Player {self.current_player}"
                )

        except Exception as e:
            self.mouse_info_label.config(text=f"Player {self.current_player} | Error: {e}")

    def on_frame_entry_change(self, event=None):
        """Handle real-time frame entry changes."""
        try:
            frame_text = self.frame_var.get()
            if frame_text.isdigit():
                frame_num = int(frame_text)
                if 0 <= frame_num < self.total_frames:
                    self.current_frame = frame_num
                    self.update_display()
        except:
            pass

    def on_scale_change(self, value):
        """Handle zoom scale changes."""
        self.display_scale = float(value)
        self.update_video_display()

    def set_scale(self, scale):
        """Set specific scale value."""
        self.scale_var.set(scale)
        self.display_scale = scale
        self.update_video_display()

    def fit_to_window(self):
        """Fit video to current window size."""
        self.root.update_idletasks()

        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:
            scale_x = canvas_width / self.video_width
            scale_y = canvas_height / self.video_height
            scale = min(scale_x, scale_y) * 0.95

            scale = max(0.1, min(3.0, scale))
            self.set_scale(scale)
        else:
            self.set_scale(0.8)

    def update_display(self):
        """Update all displays."""
        self.update_video_display()
        self.update_info_display()
        self.update_court_display()
        self.update_statistics()

    def update_video_display(self):
        """Update the video display with current frame."""
        frame = self.get_current_frame()
        if frame is None:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw court overlay
        self.draw_court_overlay(frame_rgb)

        # Draw annotation markers if they exist
        self.draw_annotation_markers(frame_rgb)

        # Scale the frame
        height, width = frame_rgb.shape[:2]
        new_width = int(width * self.display_scale)
        new_height = int(height * self.display_scale)

        if new_width > 0 and new_height > 0:
            frame_scaled = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL and then to PhotoImage
            pil_image = Image.fromarray(frame_scaled)
            self.video_image = ImageTk.PhotoImage(pil_image)

            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_image)
            self.video_canvas.configure(scrollregion=self.video_canvas.bbox("all"))

    def draw_court_overlay(self, frame):
        """Draw court lines overlay on the video frame."""
        try:
            # Draw court boundary using the corner points
            corner_points = []
            for corner in ['P1', 'P2', 'P3', 'P4']:
                if corner in self.court_points:
                    coords = self.normalize_point_coordinates(self.court_points[corner])
                    # These are already in video pixel coordinates, so use directly
                    corner_points.append([int(coords[0]), int(coords[1])])

            if len(corner_points) == 4:
                court_corners = np.array(corner_points, dtype=np.int32)
                cv2.polylines(frame, [court_corners], True, (0, 255, 0), 2)

            # Draw center line (net) using world coordinates
            try:
                # Convert world coordinates to pixel coordinates
                net_left_world = (0, self.COURT_LENGTH/2)
                net_right_world = (self.COURT_WIDTH, self.COURT_LENGTH/2)

                # Transform to pixel coordinates
                net_left_pixel = self.world_to_pixel_original(*net_left_world)
                net_right_pixel = self.world_to_pixel_original(*net_right_world)

                net_left_pt = (int(net_left_pixel[0]), int(net_left_pixel[1]))
                net_right_pt = (int(net_right_pixel[0]), int(net_right_pixel[1]))

                cv2.line(frame, net_left_pt, net_right_pt, (255, 0, 0), 2)
            except Exception:
                pass  # Skip net line if transformation fails

        except Exception as e:
            print(f"Court overlay error: {e}")

    def draw_annotation_markers(self, frame):
        """Draw markers for all player annotations."""
        if self.current_frame not in self.annotations:
            return

        frame_annotations = self.annotations[self.current_frame]

        for player_id, (world_x, world_y) in frame_annotations.items():
            try:
                # Convert world coordinates to original pixel coordinates
                pixel_x, pixel_y = self.world_to_pixel_original(world_x, world_y)

                # Convert to frame coordinates (these are already at original scale)
                display_x = int(pixel_x)
                display_y = int(pixel_y)

                # Get player color
                color = self.player_colors.get(player_id, (255, 255, 255))

                # Draw crosshair marker
                cv2.drawMarker(frame, (display_x, display_y), color,
                               cv2.MARKER_CROSS, 30, 3)

                # Draw circle
                cv2.circle(frame, (display_x, display_y), 15, color, 2)

                # Add text label
                label = f"P{player_id}"
                cv2.putText(frame, label, (display_x - 10, display_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Highlight current player
                if player_id == self.current_player:
                    cv2.circle(frame, (display_x, display_y), 25, color, 3)

            except Exception as e:
                print(f"Marker drawing error for player {player_id}: {e}")

    def update_info_display(self):
        """Update frame and position information."""
        # Frame info
        timestamp = self.current_frame / self.fps if self.fps > 0 else 0
        self.frame_var.set(str(self.current_frame))

        frame_annotations = self.annotations.get(self.current_frame, {})
        annotation_count = len(frame_annotations)

        if annotation_count == 0:
            status = "○ Not annotated"
        elif annotation_count == 1:
            status = "◐ Partially annotated"
        else:
            status = "✓ Fully annotated"

        self.frame_info_label.config(
            text=f"Frame {self.current_frame} / {self.total_frames - 1} | "
                 f"Time: {timestamp:.2f}s | {status}"
        )

        # Position info
        if frame_annotations:
            info_lines = [f"Frame {self.current_frame}:"]
            for player_id in [1, 2]:
                if player_id in frame_annotations:
                    world_x, world_y = frame_annotations[player_id]
                    marker = "▶ " if player_id == self.current_player else "  "
                    info_lines.append(f"{marker}Player {player_id}: ({world_x:.2f}, {world_y:.2f})")
                else:
                    marker = "▶ " if player_id == self.current_player else "  "
                    info_lines.append(f"{marker}Player {player_id}: Not marked")

            info_text = "\n".join(info_lines)
        else:
            info_text = f"Frame {self.current_frame}:\nNo positions marked\n(Click on video to annotate Player {self.current_player})"

        self.position_info_label.config(text=info_text)

    def update_court_display(self):
        """Update the small court reference display."""
        self.court_canvas.delete("all")

        # Court dimensions
        margin = 10
        court_width = 140
        court_height = 200

        # Draw court outline
        self.court_canvas.create_rectangle(margin, margin, margin + court_width, margin + court_height,
                                           outline='black', width=2)

        # Draw net line
        net_y = margin + court_height // 2
        self.court_canvas.create_line(margin, net_y, margin + court_width, net_y,
                                      fill='red', width=2)

        # Add labels
        self.court_canvas.create_text(margin + court_width//2, 5, text="Court",
                                      font=('Arial', 8, 'bold'))
        self.court_canvas.create_text(margin + court_width//2, margin - 5, text="Top",
                                      font=('Arial', 7))
        self.court_canvas.create_text(margin + court_width//2, margin + court_height + 8, text="Bottom",
                                      font=('Arial', 7))
        self.court_canvas.create_text(margin + court_width//2, net_y, text="NET",
                                      fill='white', font=('Arial', 7, 'bold'))

        # Draw current annotations if they exist
        if self.current_frame in self.annotations:
            frame_annotations = self.annotations[self.current_frame]

            for player_id, (world_x, world_y) in frame_annotations.items():
                # Scale to mini court
                court_x = margin + (world_x / self.COURT_WIDTH) * court_width
                court_y = margin + (world_y / self.COURT_LENGTH) * court_height

                # Choose color
                if player_id == 1:
                    color = 'red'
                    outline_color = 'darkred'
                else:
                    color = 'blue'
                    outline_color = 'darkblue'

                # Draw player marker
                size = 6 if player_id == self.current_player else 4
                self.court_canvas.create_oval(court_x-size, court_y-size, court_x+size, court_y+size,
                                              fill=color, outline=outline_color, width=2)

                # Add player number
                self.court_canvas.create_text(court_x, court_y, text=str(player_id),
                                              fill='white', font=('Arial', 6, 'bold'))

    def update_statistics(self):
        """Update statistics display."""
        # Count annotations per player
        player1_count = 0
        player2_count = 0
        both_players_count = 0

        for frame_annotations in self.annotations.values():
            if 1 in frame_annotations:
                player1_count += 1
            if 2 in frame_annotations:
                player2_count += 1
            if 1 in frame_annotations and 2 in frame_annotations:
                both_players_count += 1

        total_annotated_frames = len(self.annotations)
        coverage = (total_annotated_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

        stats_text = f"Annotated frames: {total_annotated_frames}\n"
        stats_text += f"Player 1 annotations: {player1_count}\n"
        stats_text += f"Player 2 annotations: {player2_count}\n"
        stats_text += f"Both players: {both_players_count}\n"
        stats_text += f"Total frames: {self.total_frames}\n"
        stats_text += f"Coverage: {coverage:.1f}%"

        if total_annotated_frames > 0:
            frames = sorted(self.annotations.keys())
            stats_text += f"\nRange: {frames[0]} - {frames[-1]}"

        self.stats_label.config(text=stats_text)

    def clear_current_player(self):
        """Clear annotation for current player in current frame."""
        if (self.current_frame in self.annotations and
                self.current_player in self.annotations[self.current_frame]):
            del self.annotations[self.current_frame][self.current_player]

            # Remove frame entry if no players left
            if not self.annotations[self.current_frame]:
                del self.annotations[self.current_frame]

            self.update_display()
            print(f"Cleared Player {self.current_player} annotation for frame {self.current_frame}")

    def clear_current_frame(self):
        """Clear all annotations for current frame."""
        if self.current_frame in self.annotations:
            del self.annotations[self.current_frame]
            self.update_display()
            print(f"Cleared all annotations for frame {self.current_frame}")

    def goto_prev_annotation(self):
        """Go to previous annotated frame."""
        annotated_frames = sorted([f for f in self.annotations.keys() if f < self.current_frame])
        if annotated_frames:
            self.current_frame = annotated_frames[-1]
            self.update_display()

    def goto_next_annotation(self):
        """Go to next annotated frame."""
        annotated_frames = sorted([f for f in self.annotations.keys() if f > self.current_frame])
        if annotated_frames:
            self.current_frame = annotated_frames[0]
            self.update_display()

    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_display()

    def prev_frame_10(self):
        """Go back 10 frames."""
        self.current_frame = max(0, self.current_frame - 10)
        self.update_display()

    def next_frame_10(self):
        """Go forward 10 frames."""
        self.current_frame = min(self.total_frames - 1, self.current_frame + 10)
        self.update_display()

    def goto_frame(self, event=None):
        """Go to specific frame."""
        try:
            frame_num = int(self.frame_var.get())
            if 0 <= frame_num < self.total_frames:
                self.current_frame = frame_num
                self.update_display()
            else:
                messagebox.showerror("Error", f"Frame must be between 0 and {self.total_frames - 1}")
                self.frame_var.set(str(self.current_frame))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number")
            self.frame_var.set(str(self.current_frame))

    def save_annotations(self):
        """Save annotations to file."""
        try:
            # Convert annotations to serializable format
            serializable_annotations = {}
            for frame_num, frame_annotations in self.annotations.items():
                serializable_annotations[str(frame_num)] = {
                    str(player_id): list(position) for player_id, position in frame_annotations.items()
                }

            save_data = {
                'video_path': str(self.video_path),
                'video_name': self.video_name,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'video_resolution': {
                    'width': self.video_width,
                    'height': self.video_height
                },
                'court_dimensions': {
                    'width': self.COURT_WIDTH,
                    'length': self.COURT_LENGTH
                },
                'court_points': self.court_points,
                'annotations': serializable_annotations,
                'annotation_info': {
                    'total_annotated_frames': len(self.annotations),
                    'annotation_method': 'video_click_with_homography_two_players',
                    'coverage_percentage': (len(self.annotations) / self.total_frames) * 100,
                    'players_supported': [1, 2],
                    'data_source': 'pipeline_compatible'
                }
            }

            with open(self.annotation_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"Annotations saved to: {self.annotation_file}")
            messagebox.showinfo("Success", f"Saved annotations for {len(self.annotations)} frames")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")

    def load_existing_annotations(self):
        """Load existing annotations if they exist."""
        if self.annotation_file.exists():
            try:
                with open(self.annotation_file, 'r') as f:
                    data = json.load(f)

                # Convert string keys back to integers and tuples
                annotations_data = data.get('annotations', {})
                self.annotations = {}

                for frame_str, frame_data in annotations_data.items():
                    frame_num = int(frame_str)
                    self.annotations[frame_num] = {}

                    for player_str, position_list in frame_data.items():
                        player_id = int(player_str)
                        self.annotations[frame_num][player_id] = tuple(position_list)

                print(f"Loaded existing annotations for {len(self.annotations)} frames")

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

                # Convert loaded data
                annotations_data = data.get('annotations', {})
                self.annotations = {}

                for frame_str, frame_data in annotations_data.items():
                    frame_num = int(frame_str)
                    self.annotations[frame_num] = {}

                    if isinstance(frame_data, dict):  # New format with player IDs
                        for player_str, position_list in frame_data.items():
                            player_id = int(player_str)
                            self.annotations[frame_num][player_id] = tuple(position_list)
                    else:  # Old format - assume single player
                        self.annotations[frame_num][1] = tuple(frame_data)

                self.update_display()
                messagebox.showinfo("Success", f"Loaded annotations for {len(self.annotations)} frames")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations: {e}")

    def export_for_evaluation(self):
        """Export annotations in format compatible with tracking evaluation."""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to export")
            return

        try:
            # Convert to evaluation format
            player_positions = []

            for frame_index, frame_annotations in self.annotations.items():
                for player_id, (world_x, world_y) in frame_annotations.items():
                    position_entry = {
                        'frame_index': frame_index,
                        'player_id': player_id,
                        'hip_world_X': world_x,
                        'hip_world_Y': world_y,
                        'left_ankle_world_X': world_x,  # Same as hip for ground truth
                        'left_ankle_world_Y': world_y,
                        'right_ankle_world_X': world_x,
                        'right_ankle_world_Y': world_y,
                        'annotation_method': 'video_click_homography'
                    }

                    player_positions.append(position_entry)

            # Sort by frame and then by player
            player_positions.sort(key=lambda x: (x['frame_index'], x['player_id']))

            # Count statistics
            player1_count = sum(1 for p in player_positions if p['player_id'] == 1)
            player2_count = sum(1 for p in player_positions if p['player_id'] == 2)

            # Create export data
            export_data = {
                'video_info': {
                    'path': str(self.video_path),
                    'name': self.video_name,
                    'total_frames': self.total_frames,
                    'fps': self.fps,
                    'width': self.video_width,
                    'height': self.video_height
                },
                'court_dimensions': {
                    'width': self.COURT_WIDTH,
                    'length': self.COURT_LENGTH
                },
                'court_points': self.court_points,
                'player_positions': player_positions,
                'annotation_method': 'video_click_with_homography_transformation_two_players',
                'annotation_info': {
                    'total_annotated_frames': len(self.annotations),
                    'total_positions': len(player_positions),
                    'player1_positions': player1_count,
                    'player2_positions': player2_count,
                    'annotation_coverage': len(self.annotations) / self.total_frames,
                    'players_supported': [1, 2],
                    'coordinate_transformation': 'homography_matrix',
                    'pipeline_compatible': True
                }
            }

            # Save export file
            export_file = self.results_dir / "video_click_ground_truth.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            messagebox.showinfo("Success",
                                f"Ground truth exported to:\n{export_file}\n\n"
                                f"Player 1: {player1_count} positions\n"
                                f"Player 2: {player2_count} positions\n"
                                f"Total: {len(player_positions)} positions")
            print(f"Exported {len(player_positions)} positions to: {export_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def start_auto_save(self):
        """Start auto-save timer."""
        def auto_save():
            if self.annotations:  # Only save if there are annotations
                try:
                    self.save_annotations()
                    print(f"Auto-saved annotations for {len(self.annotations)} frames")
                except Exception as e:
                    print(f"Auto-save failed: {e}")

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
        if self.video_player:
            self.video_player.release()
        self.root.destroy()

    def run(self):
        """Run the annotation tool."""
        self.setup_gui()
        self.start_auto_save()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python video_annotation_tool.py <video_file_path>")
        print("\nTwo-Player Video Click Badminton Annotation Tool")
        print("=" * 60)
        print("Features:")
        print("  ✓ Click directly on video to mark both player positions")
        print("  ✓ Uses homography for accurate coordinate transformation")
        print("  ✓ Real-time court overlay with boundaries and net")
        print("  ✓ Zoom and pan controls for precise positioning")
        print("  ✓ Player selection (Player 1: Red, Player 2: Blue)")
        print("  ✓ Auto-advance logic for efficient annotation")
        print("  ✓ Mouse coordinates shown in real-time")
        print("  ✓ Visual markers and validation for both players")
        print("  ✓ Compatible with standard pipeline outputs")
        print("\nData Sources (checked in order):")
        print("  1. results/[video_name]/pose.json (standard pipeline)")
        print("  2. results/[video_name]/detections.csv (fallback)")
        print("  3. [video_name]/pose.json (legacy format)")
        print("\nRequirements:")
        print("  - Court detection must be run first")
        print("  - Video file in supported format")
        print("\nPipeline Integration:")
        print("  1. Run: python detect_court.py <video>")
        print("  2. Run: python detect_pose.py <video>")
        print("  3. Run: python video_annotation_tool.py <video>")
        print("\nControls:")
        print("  - Select Player 1 or Player 2 with radio buttons")
        print("  - Click on video where selected player is positioned")
        print("  - Use zoom slider or buttons to adjust view")
        print("  - Arrow keys or buttons to navigate frames")
        print("  - Delete/Backspace to clear current player annotation")
        print("  - Ctrl+S to save annotations")
        print("\nKeyboard Shortcuts:")
        print("  Left/Right arrows: Navigate frames")
        print("  1/2 keys: Switch to Player 1/2")
        print("  Delete/Backspace: Clear current player annotation")
        print("  Ctrl+S: Save annotations")
        print("\nAuto-advance Logic:")
        print("  - After annotating a player, switches to other player if not annotated")
        print("  - After both players annotated, advances to next frame")
        print("  - Efficient workflow for full game annotation")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Starting two-player video click annotation tool for: {video_path}")

    # Check for required data files
    video_name = Path(video_path).stem
    results_dir = Path("results") / video_name

    print(f"Looking for pipeline data in: {results_dir}")

    # Check what files are available
    available_files = []
    if (results_dir / "pose.json").exists():
        available_files.append("pose.json")
    if (results_dir / "detections.csv").exists():
        available_files.append("detections.csv")
    if (results_dir / "calibration.csv").exists():
        available_files.append("calibration.csv")

    if available_files:
        print(f"Found: {', '.join(available_files)}")
    else:
        print("Warning: No pipeline data found. Make sure to run court detection first:")
        print(f"  python detect_court.py {video_path}")
        print(f"  python detect_pose.py {video_path}")

    print("\nInstructions:")
    print("1. The video will display with court boundary overlay (green lines)")
    print("2. Select Player 1 (Red) or Player 2 (Blue) using radio buttons")
    print("3. Click directly on the video where the selected player is positioned")
    print("4. The tool will automatically switch players or advance frames")
    print("5. Use zoom controls if you need to see details more clearly")
    print("6. Mouse coordinates are shown in real-time in the bottom right")
    print("7. Annotations are auto-saved every 60 seconds")
    print("8. Export creates evaluation-ready ground truth data for both players")

    try:
        annotator = VideoClickAnnotator(video_path)
        annotator.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()