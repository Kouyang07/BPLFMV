#!/usr/bin/env python3
"""
Video-Click Badminton Player Position Annotation Tool

GUI tool for annotating player positions by clicking directly on the video frame.
Converts pixel coordinates to world coordinates using homography from court detection.
Focuses on annotating the closest player position for evaluation purposes.

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
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import time


class VideoClickAnnotator:
    """Annotation tool that converts video clicks to world coordinates."""

    # Court dimensions (meters)
    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    def __init__(self, video_path: str):
        """Initialize the video click annotator."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem

        # Results directory structure
        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"
        self.annotation_file = self.results_dir / "video_click_annotations.json"

        # Load court detection data for coordinate transformation
        self.court_points = None
        self.homography_matrix = None
        self.inverse_homography = None
        self.load_court_data()

        # Video player
        self.video_player = None
        self.init_video_player()

        # Annotation data
        self.annotations = {}  # frame_number -> (world_x, world_y)
        self.current_frame = 0

        # GUI elements
        self.root = None
        self.video_canvas = None
        self.video_image = None
        self.display_scale = 1.0
        self.click_marker = None

        # Auto-save
        self.auto_save_timer = None

    def load_court_data(self):
        """Load court points and calculate homography for coordinate transformation."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Court detection file not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        self.court_points = data.get('court_points', {})

        # Check for required court points
        required_points = ['P1', 'P2', 'P3', 'P4']
        if not all(point in self.court_points for point in required_points):
            raise ValueError(f"Missing required court points: {required_points}")

        # Calculate homography matrix
        self.calculate_homography()

        print("Court coordinate system loaded:")
        print("  P1: Upper baseline + LEFT sideline")
        print("  P2: Lower baseline + LEFT sideline")
        print("  P3: Lower baseline + RIGHT sideline")
        print("  P4: Upper baseline + RIGHT sideline")

    def calculate_homography(self):
        """Calculate homography matrix from video pixels to world coordinates."""
        # Image points (pixels) from court detection
        image_points = np.array([
            self.court_points['P1'],  # Upper baseline + LEFT sideline
            self.court_points['P2'],  # Lower baseline + LEFT sideline
            self.court_points['P3'],  # Lower baseline + RIGHT sideline
            self.court_points['P4']   # Upper baseline + RIGHT sideline
        ], dtype=np.float32)

        # Corresponding world coordinates (meters)
        world_points = np.array([
            [0, 0],                      # P1: Left side, top (0, 0)
            [0, self.COURT_LENGTH],      # P2: Left side, bottom (0, 13.4)
            [self.COURT_WIDTH, self.COURT_LENGTH],  # P3: Right side, bottom (6.1, 13.4)
            [self.COURT_WIDTH, 0]        # P4: Right side, top (6.1, 0)
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
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

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current video frame."""
        self.video_player.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_player.read()
        return frame if ret else None

    def setup_gui(self):
        """Setup the main GUI."""
        self.root = tk.Tk()
        self.root.title(f"Video Click Annotation - {self.video_name}")
        self.root.geometry("1400x900")
        self.root.state('zoomed')  # Start maximized on Windows

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

        # Video title - compact
        video_title = ttk.Label(left_panel, text="Click on video to mark closest player position",
                                font=('Arial', 11, 'bold'))
        video_title.grid(row=0, column=0, pady=(0, 5), sticky="ew")

        # Video canvas container - this gets most of the space
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

        # Bind events to canvas with better responsiveness
        self.video_canvas.bind('<Button-1>', self.on_video_click)
        self.video_canvas.bind('<Motion>', self.on_mouse_move)
        self.video_canvas.bind('<Enter>', lambda e: self.video_canvas.focus_set())
        self.video_canvas.focus_set()  # Allow canvas to receive keyboard focus

        # Video controls - compact layout
        controls_frame = ttk.Frame(left_panel)
        controls_frame.grid(row=2, column=0, sticky="ew", pady=3)
        controls_frame.grid_columnconfigure(2, weight=1)

        # Navigation buttons - make them actually work
        btn_prev_10 = ttk.Button(controls_frame, text="◀◀", width=4)
        btn_prev_10.configure(command=self.prev_frame_10)
        btn_prev_10.grid(row=0, column=0, padx=2)

        btn_prev = ttk.Button(controls_frame, text="◀", width=4)
        btn_prev.configure(command=self.prev_frame)
        btn_prev.grid(row=0, column=1, padx=2)

        btn_next = ttk.Button(controls_frame, text="▶", width=4)
        btn_next.configure(command=self.next_frame)
        btn_next.grid(row=0, column=3, padx=2)

        btn_next_10 = ttk.Button(controls_frame, text="▶▶", width=4)
        btn_next_10.configure(command=self.next_frame_10)
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

        btn_go = ttk.Button(frame_controls, text="Go")
        btn_go.configure(command=self.goto_frame)
        btn_go.grid(row=0, column=2, padx=2)

        # Zoom controls - compact single row
        zoom_frame = ttk.Frame(left_panel)
        zoom_frame.grid(row=3, column=0, sticky="ew", pady=2)
        zoom_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(zoom_frame, text="Zoom:").grid(row=0, column=0, sticky="w")

        self.scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(zoom_frame, from_=0.2, to=3.0, variable=self.scale_var,
                                 orient=tk.HORIZONTAL, command=self.on_scale_change)
        scale_slider.grid(row=0, column=1, sticky="ew", padx=5)

        btn_100 = ttk.Button(zoom_frame, text="100%", width=6)
        btn_100.configure(command=lambda: self.set_scale(1.0))
        btn_100.grid(row=0, column=2, padx=2)

        btn_fit = ttk.Button(zoom_frame, text="Fit", width=6)
        btn_fit.configure(command=self.fit_to_window)
        btn_fit.grid(row=0, column=3, padx=2)

        # Status info - single compact row
        status_frame = ttk.Frame(left_panel)
        status_frame.grid(row=4, column=0, sticky="ew", pady=2)
        status_frame.grid_columnconfigure(0, weight=1)

        self.frame_info_label = ttk.Label(status_frame, text="", font=('Arial', 9))
        self.frame_info_label.grid(row=0, column=0, sticky="w")

        self.mouse_info_label = ttk.Label(status_frame, text="", font=('Arial', 9))
        self.mouse_info_label.grid(row=0, column=1, sticky="e")

        # Right panel - Controls (compact, less space)
        right_panel = ttk.Frame(self.root)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)

        # Current annotation info
        current_frame = ttk.LabelFrame(right_panel, text="Current Position")
        current_frame.pack(fill=tk.X, pady=(0, 5))

        self.position_info_label = ttk.Label(current_frame, text="No position marked",
                                             wraplength=280, justify=tk.LEFT, font=('Arial', 9))
        self.position_info_label.pack(pady=8)

        # Court visualization (smaller)
        court_frame = ttk.LabelFrame(right_panel, text="Court Reference")
        court_frame.pack(fill=tk.X, pady=(0, 5))

        self.court_canvas = tk.Canvas(court_frame, width=160, height=220, bg='white')
        self.court_canvas.pack(pady=5)

        # Controls - more compact
        control_frame = ttk.LabelFrame(right_panel, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # Clear button
        btn_clear = ttk.Button(control_frame, text="Clear Current Position")
        btn_clear.configure(command=self.clear_current_frame)
        btn_clear.pack(fill=tk.X, pady=2)

        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=2)

        btn_prev_ann = ttk.Button(nav_frame, text="← Previous")
        btn_prev_ann.configure(command=self.goto_prev_annotation)
        btn_prev_ann.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))

        btn_next_ann = ttk.Button(nav_frame, text="Next →")
        btn_next_ann.configure(command=self.goto_next_annotation)
        btn_next_ann.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT, font=('Arial', 9))
        self.stats_label.pack(pady=5)

        # File operations
        file_frame = ttk.LabelFrame(right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=(0, 5))

        btn_save = ttk.Button(file_frame, text="💾 Save")
        btn_save.configure(command=self.save_annotations)
        btn_save.pack(fill=tk.X, pady=1)

        btn_load = ttk.Button(file_frame, text="📁 Load")
        btn_load.configure(command=self.load_annotations_dialog)
        btn_load.pack(fill=tk.X, pady=1)

        btn_export = ttk.Button(file_frame, text="📤 Export")
        btn_export.configure(command=self.export_for_evaluation)
        btn_export.pack(fill=tk.X, pady=1)

        # Keyboard bindings - bind to canvas instead of root for better focus
        self.video_canvas.bind('<Left>', lambda e: self.prev_frame())
        self.video_canvas.bind('<Right>', lambda e: self.next_frame())
        self.video_canvas.bind('<Control-s>', lambda e: self.save_annotations())
        self.video_canvas.bind('<Delete>', lambda e: self.clear_current_frame())
        self.video_canvas.bind('<BackSpace>', lambda e: self.clear_current_frame())

        # Also bind to root for when other widgets have focus
        self.root.bind('<Control-s>', lambda e: self.save_annotations())

        # Initialize displays
        self.load_existing_annotations()
        self.update_display()

        # Set initial focus to canvas
        self.root.after(100, lambda: self.video_canvas.focus_set())

    def on_video_click(self, event):
        """Handle clicks on the video canvas."""
        # Get click coordinates relative to the video image
        canvas_x = self.video_canvas.canvasx(event.x)
        canvas_y = self.video_canvas.canvasy(event.y)

        try:
            # Convert to world coordinates
            world_x, world_y = self.pixel_to_world(canvas_x, canvas_y)

            # Check if the click is within the court bounds
            if 0 <= world_x <= self.COURT_WIDTH and 0 <= world_y <= self.COURT_LENGTH:
                # Store annotation
                self.annotations[self.current_frame] = (world_x, world_y)

                print(f"Frame {self.current_frame}: Closest player at ({world_x:.2f}, {world_y:.2f})")

                # Update displays
                self.update_display()

                # Auto-advance to next frame
                if self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                    self.update_display()
            else:
                # Click outside court
                messagebox.showwarning("Invalid Position",
                                       f"Click is outside court bounds.\n"
                                       f"Clicked: ({world_x:.2f}, {world_y:.2f})\n"
                                       f"Court bounds: (0, 0) to ({self.COURT_WIDTH}, {self.COURT_LENGTH})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert coordinates: {e}")

    def on_mouse_move(self, event):
        """Show mouse position in world coordinates."""
        try:
            canvas_x = self.video_canvas.canvasx(event.x)
            canvas_y = self.video_canvas.canvasy(event.y)

            world_x, world_y = self.pixel_to_world(canvas_x, canvas_y)

            if 0 <= world_x <= self.COURT_WIDTH and 0 <= world_y <= self.COURT_LENGTH:
                self.mouse_info_label.config(text=f"Court: ({world_x:.2f}, {world_y:.2f})")
            else:
                self.mouse_info_label.config(text="Outside court")

        except:
            self.mouse_info_label.config(text="")

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
        # Force update to get actual canvas size
        self.root.update_idletasks()

        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:  # Make sure canvas is rendered
            scale_x = canvas_width / self.video_width
            scale_y = canvas_height / self.video_height
            scale = min(scale_x, scale_y) * 0.95  # Leave some margin

            # Clamp to reasonable bounds
            scale = max(0.1, min(3.0, scale))
            self.set_scale(scale)
        else:
            # Fallback if canvas size not available
            self.set_scale(0.8)

    def update_display_fast(self):
        """Fast update for better responsiveness during annotation."""
        if self.fast_update_mode:
            # Only update video and essential info
            self.update_video_display()
            self.update_info_display()
            # Update other displays less frequently
            self.root.after_idle(self.update_court_display)
            self.root.after_idle(self.update_statistics)
        else:
            # Full update
            self.update_display()

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

        # Draw annotation marker if exists
        if self.current_frame in self.annotations:
            self.draw_annotation_marker(frame_rgb)

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
        # Draw court boundary
        court_corners = np.array([
            self.court_points['P1'],
            self.court_points['P2'],
            self.court_points['P3'],
            self.court_points['P4']
        ], dtype=np.int32)

        cv2.polylines(frame, [court_corners], True, (0, 255, 0), 2)

        # Draw center line
        try:
            # Net line (center of court)
            net_left = self.world_to_pixel(0, self.COURT_LENGTH/2)
            net_right = self.world_to_pixel(self.COURT_WIDTH, self.COURT_LENGTH/2)

            net_left_pixel = (int(net_left[0] / self.display_scale), int(net_left[1] / self.display_scale))
            net_right_pixel = (int(net_right[0] / self.display_scale), int(net_right[1] / self.display_scale))

            cv2.line(frame, net_left_pixel, net_right_pixel, (255, 0, 0), 2)

        except:
            pass  # Skip overlay if transformation fails

    def draw_annotation_marker(self, frame):
        """Draw marker for current annotation."""
        if self.current_frame not in self.annotations:
            return

        try:
            world_x, world_y = self.annotations[self.current_frame]
            pixel_x, pixel_y = self.world_to_pixel(world_x, world_y)

            # Adjust for display scale
            display_x = int(pixel_x / self.display_scale)
            display_y = int(pixel_y / self.display_scale)

            # Draw crosshair marker
            cv2.drawMarker(frame, (display_x, display_y), (255, 255, 0),
                           cv2.MARKER_CROSS, 30, 3)

            # Draw circle
            cv2.circle(frame, (display_x, display_y), 15, (255, 255, 0), 2)

            # Add text label
            cv2.putText(frame, "CLOSEST PLAYER", (display_x - 50, display_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        except:
            pass  # Skip marker if transformation fails

    def update_info_display(self):
        """Update frame and position information."""
        # Frame info
        timestamp = self.current_frame / self.fps if self.fps > 0 else 0
        self.frame_var.set(str(self.current_frame))

        status = "✓ Annotated" if self.current_frame in self.annotations else "○ Not annotated"
        self.frame_info_label.config(
            text=f"Frame {self.current_frame} / {self.total_frames - 1} | "
                 f"Time: {timestamp:.2f}s | {status}"
        )

        # Position info
        if self.current_frame in self.annotations:
            world_x, world_y = self.annotations[self.current_frame]
            info_text = f"Frame {self.current_frame}:\nClosest player at ({world_x:.2f}, {world_y:.2f})"
        else:
            info_text = f"Frame {self.current_frame}:\nNo position marked\n(Click on video to annotate)"

        self.position_info_label.config(text=info_text)

    def update_court_display(self):
        """Update the small court reference display."""
        self.court_canvas.delete("all")

        # Smaller court dimensions
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

        # Add compact labels
        self.court_canvas.create_text(margin + court_width//2, 5, text="Court",
                                      font=('Arial', 8, 'bold'))
        self.court_canvas.create_text(margin + court_width//2, margin - 5, text="Top",
                                      font=('Arial', 7))
        self.court_canvas.create_text(margin + court_width//2, margin + court_height + 8, text="Bottom",
                                      font=('Arial', 7))
        self.court_canvas.create_text(margin + court_width//2, net_y, text="NET",
                                      fill='white', font=('Arial', 7, 'bold'))

        # Draw current annotation if exists
        if self.current_frame in self.annotations:
            world_x, world_y = self.annotations[self.current_frame]

            # Scale to mini court
            court_x = margin + (world_x / self.COURT_WIDTH) * court_width
            court_y = margin + (world_y / self.COURT_LENGTH) * court_height

            # Draw player marker
            self.court_canvas.create_oval(court_x-4, court_y-4, court_x+4, court_y+4,
                                          fill='red', outline='darkred', width=2)

    def update_statistics(self):
        """Update statistics display."""
        total_annotations = len(self.annotations)
        coverage = (total_annotations / self.total_frames) * 100 if self.total_frames > 0 else 0

        stats_text = f"Annotated frames: {total_annotations}\n"
        stats_text += f"Total frames: {self.total_frames}\n"
        stats_text += f"Coverage: {coverage:.1f}%\n"

        if total_annotations > 0:
            frames = sorted(self.annotations.keys())
            stats_text += f"Range: {frames[0]} - {frames[-1]}"

        self.stats_label.config(text=stats_text)

    def clear_current_frame(self):
        """Clear annotation for current frame."""
        if self.current_frame in self.annotations:
            del self.annotations[self.current_frame]
            self.update_display()
            print(f"Cleared annotation for frame {self.current_frame}")

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
                'annotations': {str(k): v for k, v in self.annotations.items()},
                'annotation_info': {
                    'total_annotated_frames': len(self.annotations),
                    'annotation_method': 'video_click_with_homography',
                    'coverage_percentage': (len(self.annotations) / self.total_frames) * 100
                }
            }

            with open(self.annotation_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"Annotations saved to: {self.annotation_file}")
            messagebox.showinfo("Success", f"Saved {len(self.annotations)} annotations")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")

    def load_existing_annotations(self):
        """Load existing annotations if they exist."""
        if self.annotation_file.exists():
            try:
                with open(self.annotation_file, 'r') as f:
                    data = json.load(f)

                # Convert string keys back to integers
                annotations_data = data.get('annotations', {})
                self.annotations = {int(k): tuple(v) for k, v in annotations_data.items()}

                print(f"Loaded {len(self.annotations)} existing annotations")

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

                annotations_data = data.get('annotations', {})
                self.annotations = {int(k): tuple(v) for k, v in annotations_data.items()}

                self.update_display()
                messagebox.showinfo("Success", f"Loaded {len(self.annotations)} annotations")

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

            for frame_index, (world_x, world_y) in self.annotations.items():
                position_entry = {
                    'frame_index': frame_index,
                    'player_id': 0,  # Always 0 for closest player
                    'hip_world_X': world_x,
                    'hip_world_Y': world_y,
                    'left_ankle_world_X': world_x,  # Same as hip for ground truth
                    'left_ankle_world_Y': world_y,
                    'right_ankle_world_X': world_x,
                    'right_ankle_world_Y': world_y,
                    'annotation_method': 'video_click_homography'
                }

                player_positions.append(position_entry)

            # Sort by frame
            player_positions.sort(key=lambda x: x['frame_index'])

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
                'annotation_method': 'video_click_with_homography_transformation',
                'annotation_info': {
                    'total_annotated_frames': len(self.annotations),
                    'total_positions': len(player_positions),
                    'annotation_coverage': len(self.annotations) / self.total_frames,
                    'player_type': 'closest_to_camera',
                    'coordinate_transformation': 'homography_matrix'
                }
            }

            # Save export file
            export_file = self.results_dir / "video_click_ground_truth.json"
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
                    print(f"Auto-saved {len(self.annotations)} annotations")
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
        print("\nVideo Click Badminton Annotation Tool")
        print("=" * 50)
        print("Features:")
        print("  ✓ Click directly on video to mark player positions")
        print("  ✓ Uses homography for accurate coordinate transformation")
        print("  ✓ Real-time court overlay with boundaries and net")
        print("  ✓ Zoom and pan controls for precise positioning")
        print("  ✓ Auto-advance to next frame after annotation")
        print("  ✓ Mouse coordinates shown in real-time")
        print("  ✓ Visual markers and validation")
        print("\nRequirements:")
        print("  - Court detection must be run first (pose.json with court points)")
        print("  - Video file in supported format")
        print("\nControls:")
        print("  - Click on video where closest player is positioned")
        print("  - Use zoom slider or buttons to adjust view")
        print("  - Arrow keys or buttons to navigate frames")
        print("  - Delete/Backspace to clear current annotation")
        print("  - Ctrl+S to save annotations")
        print("\nKeyboard Shortcuts:")
        print("  Left/Right arrows: Navigate frames")
        print("  Delete/Backspace: Clear current annotation")
        print("  Ctrl+S: Save annotations")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Starting video click annotation tool for: {video_path}")
    print("\nInstructions:")
    print("1. The video will display with court boundary overlay (green lines)")
    print("2. Click directly on the video where the closest player is positioned")
    print("3. Use zoom controls if you need to see details more clearly")
    print("4. The tool will auto-advance to the next frame after each annotation")
    print("5. Mouse coordinates are shown in real-time in the bottom right")
    print("6. Annotations are auto-saved every 60 seconds")
    print("7. Export creates evaluation-ready ground truth data")

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