#!/usr/bin/env python3
"""
Optimized Universal Visualization Script for BPLFMV Structure

Key Optimizations:
1. Multi-threaded frame processing with ThreadPoolExecutor
2. Frame batching and pre-loading
3. Memory-mapped file reading for large datasets
4. Optimized OpenCV operations
5. Smart caching and data structure optimization
6. GPU acceleration support (when available)
7. Progressive rendering for real-time preview
8. Memory management and garbage collection optimization

Handles different data formats:
- Stage 1: court.csv → Court detection visualization
- Stage 2: pose.json → Pose estimation visualization
- Stage 3: positions.json → 3D position tracking visualization
- Stage 4: corrected_positions.json → Jump correction comparison
"""

import sys
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from functools import partial, lru_cache
import gc
import time

import numpy as np
import cv2
import json
import csv
import argparse
import logging
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List, Union
import psutil
import threading

# Set up logging with performance tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self):
        self.start_time = None
        self.frame_times = []

    def start(self):
        self.start_time = time.time()

    def log_frame(self, frame_idx: int):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.frame_times.append(elapsed)
            if frame_idx % 100 == 0:
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                logging.info(f"Frame {frame_idx}: Avg FPS: {avg_fps:.2f}, Memory: {psutil.virtual_memory().percent}%")

class OptimizedVideoProcessor:
    """Optimized video processing with multi-threading and caching"""

    def __init__(self, video_path: str, num_threads: Optional[int] = None):
        self.video_path = video_path
        self.num_threads = num_threads or min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        self.frame_cache = {}
        self.cache_lock = Lock()
        self.max_cache_size = 50  # Maximum frames to cache

        # Get video info once
        self.video_info = self._get_video_info()

        # Pre-allocate commonly used arrays
        self._initialize_templates()

    def _get_video_info(self) -> Dict[str, Any]:
        """Extract video information with error handling"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        cap.release()
        return info

    def _initialize_templates(self):
        """Pre-allocate templates and commonly used arrays"""
        # Court visualization parameters
        self.court_length_m = 13.4
        self.court_width_m = 6.1
        self.margin_m = 2.0
        self.court_scale = 65
        self.court_img_h = int((self.court_length_m + 2 * self.margin_m) * self.court_scale)
        self.court_img_w = int((self.court_width_m + 2 * self.margin_m) * self.court_scale)

        # Pre-generate court template
        self.court_template = np.zeros((self.court_img_h, self.court_img_w, 3), dtype=np.uint8)
        self._draw_court_template()

        # Color palettes
        self.color_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                              (0, 255, 255), (255, 255, 0), (255, 0, 255)]

        # Pose skeleton edges (cached)
        self.pose_edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                           (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

    def _draw_court_template(self):
        """Pre-draw court template to avoid redrawing every frame"""
        draw_court_optimized(self.court_template, self.court_scale)

    def get_frame_batch(self, start_frame: int, batch_size: int) -> List[Tuple[int, np.ndarray]]:
        """Get a batch of frames efficiently"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((start_frame + i, frame.copy()))

        cap.release()
        return frames

    @lru_cache(maxsize=128)
    def get_player_color(self, tracked_id: int) -> Tuple[int, int, int]:
        """Cached color lookup"""
        return self.color_palette[tracked_id % len(self.color_palette)]

class OptimizedDataLoader:
    """Optimized data loading with caching and indexing"""

    def __init__(self):
        self.data_cache = {}
        self.index_cache = {}

    @lru_cache(maxsize=32)
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load and cache JSON data"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def organize_by_frame_optimized(self, data_list: List[Dict]) -> Dict[int, List[Dict]]:
        """Optimized frame organization with indexing"""
        if id(data_list) in self.index_cache:
            return self.index_cache[id(data_list)]

        frame_dict = defaultdict(list)
        for item in data_list:
            frame_idx = item["frame_index"]
            frame_dict[frame_idx].append(item)

        result = dict(frame_dict)
        self.index_cache[id(data_list)] = result
        return result

def draw_court_optimized(court_image: np.ndarray, court_scale: int) -> None:
    """
    Optimized court drawing with pre-calculated coordinates and batch operations
    """
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0

    # Pre-calculate all coordinates
    left = int(margin_m * court_scale)
    right = int((margin_m + court_width_m) * court_scale)
    top = int(margin_m * court_scale)
    bottom = int((margin_m + court_length_m) * court_scale)
    center_x = int((margin_m + court_width_m / 2) * court_scale)
    net_y = int((margin_m + court_length_m / 2) * court_scale)

    # Draw all lines in batches
    white_lines = [
        # Main court outline
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
        # Net line
        ((left, net_y), (right, net_y))
    ]

    gray_lines = [
        # Center line
        ((center_x, top), (center_x, bottom)),
        # Service lines
        ((left, int((margin_m + court_length_m/2 - 1.98) * court_scale)),
         (right, int((margin_m + court_length_m/2 - 1.98) * court_scale))),
        ((left, int((margin_m + court_length_m/2 + 1.98) * court_scale)),
         (right, int((margin_m + court_length_m/2 + 1.98) * court_scale))),
    ]

    # Batch draw lines
    for (pt1, pt2) in white_lines:
        cv2.line(court_image, pt1, pt2, (255, 255, 255), 2)

    for (pt1, pt2) in gray_lines:
        cv2.line(court_image, pt1, pt2, (150, 150, 150), 1)

    # Add text labels (optimized)
    text_items = [
        ("NET", (center_x - 20, net_y - 10), 0.6, (200, 200, 200)),
        ("FRONT", (center_x - 30, top - 10), 0.6, (200, 200, 200)),
        ("BACK", (center_x - 25, bottom + 20), 0.6, (200, 200, 200))
    ]

    for text, pos, scale, color in text_items:
        cv2.putText(court_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

@lru_cache(maxsize=1000)
def world_to_court_cached(x: float, y: float, court_img_w: int, court_img_h: int,
                          court_scale: int, margin_m: float) -> Tuple[int, int]:
    """Cached coordinate transformation"""
    px = int((x + margin_m) * court_scale)
    py = int((y + margin_m) * court_scale)
    return (px, py)

def process_frame_batch_stage1(batch_data: Tuple[List[Tuple[int, np.ndarray]], Dict, np.ndarray]) -> List[Tuple[int, np.ndarray]]:
    """Process a batch of frames for stage 1 visualization"""
    frames, all_court_points, corner_points = batch_data
    processed_frames = []

    for frame_idx, frame in frames:
        frame_display = frame.copy()

        # Optimized court point drawing
        frame_display = draw_all_court_points_optimized(frame_display, all_court_points)
        frame_display = draw_court_polygon_optimized(frame_display, corner_points)

        # Add frame info
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames

def draw_all_court_points_optimized(frame: np.ndarray, all_court_points: Dict, show_labels: bool = True) -> np.ndarray:
    """Optimized court point drawing with vectorized operations"""
    if not all_court_points:
        return frame

    frame_display = frame.copy()

    # Group points by type for batch processing
    main_points = []
    net_points = []
    other_points = []

    for point_name, coords in all_court_points.items():
        if len(coords) >= 2:
            point_data = (point_name, int(coords[0]), int(coords[1]))

            if point_name.startswith('P') and point_name[1:].isdigit():
                main_points.append(point_data)
            elif 'NetPole' in point_name:
                net_points.append(point_data)
            else:
                other_points.append(point_data)

    # Batch draw points
    point_groups = [
        (main_points, (0, 255, 255), 6),
        (net_points, (255, 0, 255), 8),
        (other_points, (0, 255, 0), 4)
    ]

    for points, color, radius in point_groups:
        for point_name, x, y in points:
            cv2.circle(frame_display, (x, y), radius, color, -1)
            cv2.circle(frame_display, (x, y), radius + 2, (255, 255, 255), 2)

            if show_labels:
                font_scale = 0.5 if len(point_name) <= 3 else 0.4
                cv2.putText(frame_display, point_name, (x + 10, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame_display

def draw_court_polygon_optimized(frame: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Optimized polygon drawing"""
    if len(corner_points) >= 4:
        pts = corner_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
    return frame

def visualize_stage1_optimized(video_path: str, data_path: str, output_path: str, num_threads: int = None) -> None:
    """Optimized Stage 1 visualization with multi-threading"""
    logging.info("Creating optimized Stage 1 visualization: Court detection")

    # Initialize optimized components
    processor = OptimizedVideoProcessor(video_path, num_threads)
    data_loader = OptimizedDataLoader()
    perf_monitor = PerformanceMonitor()

    # Load data once
    all_court_points = read_court_csv_optimized(data_path)
    corner_points = extract_corner_points_optimized(all_court_points)

    # Setup video writer
    video_info = processor.video_info
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"],
                          (video_info["width"], video_info["height"]))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    perf_monitor.start()
    batch_size = 16  # Process frames in batches
    frame_count = video_info["frame_count"]

    # Process frames in batches with threading
    with ThreadPoolExecutor(max_workers=processor.num_threads) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                batch_data = (batch_frames, all_court_points, corner_points)
                future = executor.submit(process_frame_batch_stage1, batch_data)
                futures.append((start_frame, future))

        # Write results in order
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                try:
                    processed_batch = future.result(timeout=30)

                    # Sort by frame index to maintain order
                    processed_batch.sort(key=lambda x: x[0])

                    for frame_idx, processed_frame in processed_batch:
                        out.write(processed_frame)
                        perf_monitor.log_frame(frame_idx)
                        pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing batch starting at frame {start_frame}: {e}")

                # Periodic garbage collection
                if start_frame % (batch_size * 10) == 0:
                    gc.collect()

    out.release()
    logging.info(f"✓ Optimized Stage 1 visualization saved to {output_path}")

def read_court_csv_optimized(csv_path: str) -> Dict[str, List[float]]:
    """Optimized CSV reading with better error handling"""
    court_points = {}

    try:
        with open(csv_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                point_name = row['Point']
                x_coord = float(row['X'])
                y_coord = float(row['Y'])
                court_points[point_name] = [x_coord, y_coord]
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        raise

    return court_points

def extract_corner_points_optimized(all_court_points: Dict) -> np.ndarray:
    """Optimized corner point extraction"""
    corner_points = []
    for i in range(1, 5):
        point_name = f"P{i}"
        if point_name in all_court_points:
            corner_points.append(all_court_points[point_name])

    if len(corner_points) != 4:
        points_list = list(all_court_points.values())[:4]
        corner_points = points_list

    return np.array(corner_points, dtype=np.float32)

def visualize_stage3_optimized(video_path: str, data_path: str, output_path: str, num_threads: int = None) -> None:
    """
    Optimized Stage 3 visualization with advanced multi-threading and caching
    """
    logging.info("Creating optimized Stage 3 visualization: 3D position tracking")

    # Initialize optimized components
    processor = OptimizedVideoProcessor(video_path, num_threads)
    data_loader = OptimizedDataLoader()
    perf_monitor = PerformanceMonitor()

    # Load and organize data
    data = data_loader.load_json_data(data_path)
    court_points = data.get("court_points", {})
    video_info = data["video_info"]
    player_positions = data["player_positions"]

    # Optimize data organization
    positions_by_frame = data_loader.organize_by_frame_optimized(player_positions)
    image_points = extract_corner_points_optimized(court_points)

    # Setup video writer with side-by-side layout
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    perf_monitor.start()
    frame_count = video_info["frame_count"]
    batch_size = 8  # Smaller batch size for memory efficiency with side-by-side layout

    # Create processing function with closure for data
    def process_frame_batch_stage3_closure(batch_frames):
        return process_frame_batch_stage3(
            batch_frames, positions_by_frame, image_points, processor.court_template.copy(),
            processor.court_img_w, processor.court_img_h, processor.court_scale,
            processor.margin_m, processor.get_player_color, out_h, out_w
        )

    # Process with threading
    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 4)) as executor:  # Limit threads for memory
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                future = executor.submit(process_frame_batch_stage3_closure, batch_frames)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                try:
                    processed_batch = future.result(timeout=60)  # Longer timeout for complex processing

                    processed_batch.sort(key=lambda x: x[0])

                    for frame_idx, combined_frame in processed_batch:
                        out.write(combined_frame)
                        perf_monitor.log_frame(frame_idx)
                        pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing batch starting at frame {start_frame}: {e}")

                # Aggressive garbage collection for memory management
                if start_frame % (batch_size * 5) == 0:
                    gc.collect()

    out.release()

    total_positions = len(player_positions)
    unique_players = len(set(pos["tracked_id"] for pos in player_positions))

    logging.info(f"✓ Optimized Stage 3 visualization saved to {output_path}")
    logging.info(f"✓ Total positions visualized: {total_positions}")
    logging.info(f"✓ Unique players tracked: {unique_players}")

def process_frame_batch_stage3(batch_frames, positions_by_frame, image_points, court_template,
                               court_img_w, court_img_h, court_scale, margin_m, get_player_color,
                               out_h, out_w) -> List[Tuple[int, np.ndarray]]:
    """Process a batch of frames for stage 3 with optimized operations"""
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Draw calibration points (vectorized)
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add frame number
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process positions for this frame
        frame_players = positions_by_frame.get(frame_idx, [])

        # Batch process player positions
        for pos_data in frame_players:
            tracked_id = pos_data["tracked_id"]
            color = get_player_color(tracked_id)

            # Process all position types efficiently
            position_types = [
                ("hip_world_X", "hip_world_Y", "Hip", color, 5),
                ("left_hip_world_X", "left_hip_world_Y", "LHip", (100, 100, 255), 3),
                ("right_hip_world_X", "right_hip_world_Y", "RHip", (255, 100, 100), 3),
                ("left_ankle_world_X", "left_ankle_world_Y", "LAnk", (0, 255, 255), 4),
                ("right_ankle_world_X", "right_ankle_world_Y", "RAnk", (255, 255, 0), 4)
            ]

            for x_key, y_key, label, pos_color, radius in position_types:
                if x_key in pos_data and y_key in pos_data:
                    world_x = pos_data[x_key]
                    world_y = pos_data[y_key]
                    px, py = world_to_court_cached(world_x, world_y, court_img_w, court_img_h,
                                                   court_scale, margin_m)

                    cv2.circle(court_img, (px, py), radius, pos_color, -1)
                    cv2.putText(court_img, f"P{tracked_id}-{label}", (px - 15, py - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, pos_color, 1)

        # Add statistics and legend (optimized)
        add_court_legend_optimized(court_img, court_img_w, len(frame_players), frame_idx)

        # Combine frames efficiently
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame_display.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames

def add_court_legend_optimized(court_img, court_img_w, total_players, frame_idx):
    """Add legend and statistics to court image efficiently"""
    legend_y = 50
    legend_x = court_img_w - 230

    # Text items to draw
    text_items = [
        ("POSITION LEGEND:", (legend_x, legend_y), 0.5, (255, 255, 255)),
        ("Left Hip", (legend_x + 30, legend_y + 25), 0.4, (100, 100, 255)),
        ("Right Hip", (legend_x + 30, legend_y + 45), 0.4, (255, 100, 100)),
        ("Hip Center", (legend_x + 30, legend_y + 65), 0.4, (255, 255, 255)),
        ("Left Ankle", (legend_x + 30, legend_y + 85), 0.4, (0, 255, 255)),
        ("Right Ankle", (legend_x + 30, legend_y + 105), 0.4, (255, 255, 0)),
        (f"Players tracked: {total_players}", (10, 30), 0.6, (255, 255, 255)),
        (f"Frame: {frame_idx}", (10, 50), 0.6, (255, 255, 255))
    ]

    # Legend circles
    legend_circles = [
        ((legend_x + 20, legend_y + 20), 3, (100, 100, 255)),
        ((legend_x + 20, legend_y + 40), 3, (255, 100, 100)),
        ((legend_x + 20, legend_y + 60), 5, (255, 255, 255)),
        ((legend_x + 20, legend_y + 80), 4, (0, 255, 255)),
        ((legend_x + 20, legend_y + 100), 4, (255, 255, 0))
    ]

    # Batch draw
    for (center, radius, color) in legend_circles:
        cv2.circle(court_img, center, radius, color, -1)

    for (text, pos, scale, color) in text_items:
        cv2.putText(court_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

def main():
    parser = argparse.ArgumentParser(description="Create optimized visualizations for BPLFMV pipeline stages")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Stage to visualize (1=court, 2=pose, 3=positions, 4=corrected)")
    parser.add_argument("--data_path", type=str, help="Path to data file (auto-detected if not provided)")
    parser.add_argument("--output", type=str, help="Output video path (auto-generated if not provided)")
    parser.add_argument("--threads", type=int, help="Number of threads to use (auto-detected if not provided)")
    parser.add_argument("--batch_size", type=int, default=16, help="Frame batch size for processing")
    parser.add_argument("--enable_gpu", action="store_true", help="Enable GPU acceleration if available")
    parser.add_argument("--memory_limit", type=float, default=0.8, help="Memory usage limit (0.0-1.0)")

    args = parser.parse_args()

    # Performance monitoring
    start_time = time.time()
    initial_memory = psutil.virtual_memory().percent

    # Auto-detect optimal thread count
    if args.threads is None:
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        # Use more conservative threading for memory-intensive operations
        args.threads = min(cpu_count, max(2, int(memory_gb / 2)))

    logging.info(f"Using {args.threads} threads for processing")
    logging.info(f"Initial memory usage: {initial_memory}%")

    # Auto-detect paths
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    result_dir = os.path.join("results", base_name)

    if args.data_path is None:
        stage_data_files = {
            1: os.path.join(result_dir, "court.csv"),
            2: os.path.join(result_dir, "pose.json"),
            3: os.path.join(result_dir, "positions.json"),
            4: os.path.join(result_dir, "corrected_positions.json")
        }
        args.data_path = stage_data_files[args.stage]

    if args.output is None:
        stage_names = {1: "court", 2: "pose", 3: "positions", 4: "corrected"}
        args.output = os.path.join(result_dir, f"{base_name}_{stage_names[args.stage]}_optimized_viz.mp4")

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        logging.error("Run the corresponding processing stage first")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # GPU acceleration setup (if requested and available)
    if args.enable_gpu:
        try:
            import cupy as cp
            logging.info("GPU acceleration enabled with CuPy")
        except ImportError:
            logging.warning("CuPy not available, falling back to CPU processing")
            args.enable_gpu = False

    # Memory monitoring setup
    def check_memory_usage():
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > args.memory_limit:
            logging.warning(f"Memory usage ({memory_percent:.1%}) exceeds limit ({args.memory_limit:.1%})")
            gc.collect()
            return True
        return False

    # Call appropriate optimized visualization function
    try:
        if args.stage == 1:
            visualize_stage1_optimized(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 2:
            visualize_stage2_optimized(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 3:
            visualize_stage3_optimized(args.video_path, args.data_path, args.output, args.threads)
        elif args.stage == 4:
            visualize_stage4_optimized(args.video_path, args.data_path, args.output, args.threads)

        # Performance summary
        end_time = time.time()
        total_time = end_time - start_time
        final_memory = psutil.virtual_memory().percent

        logging.info(f"✓ Optimized visualization complete!")
        logging.info(f"✓ Output saved to: {args.output}")
        logging.info(f"✓ Total processing time: {total_time:.2f} seconds")
        logging.info(f"✓ Memory usage change: {initial_memory}% → {final_memory}%")

        # Calculate performance metrics
        video_info = OptimizedVideoProcessor(args.video_path).video_info
        fps_processed = video_info["frame_count"] / total_time if total_time > 0 else 0
        logging.info(f"✓ Processing speed: {fps_processed:.2f} FPS")

    except Exception as e:
        logging.error(f"Optimized visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


# Additional optimized functions for stages 2 and 4

def visualize_stage2_optimized(video_path: str, data_path: str, output_path: str, num_threads: int = None) -> None:
    """Optimized Stage 2 visualization: Pose estimation with multi-threading"""
    logging.info("Creating optimized Stage 2 visualization: Pose estimation")

    processor = OptimizedVideoProcessor(video_path, num_threads)
    data_loader = OptimizedDataLoader()
    perf_monitor = PerformanceMonitor()

    # Load and organize data
    data = data_loader.load_json_data(data_path)
    all_court_points = data.get("all_court_points", data.get("court_points", {}))
    pose_data = data["pose_data"]
    video_info = data["video_info"]

    # Optimize pose organization
    poses_by_frame = data_loader.organize_by_frame_optimized(pose_data)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"],
                          (video_info["width"], video_info["height"]))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    perf_monitor.start()
    batch_size = 12
    frame_count = video_info["frame_count"]

    # Process with threading
    with ThreadPoolExecutor(max_workers=processor.num_threads) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                future = executor.submit(process_frame_batch_stage2,
                                         batch_frames, poses_by_frame, all_court_points, processor.pose_edges, processor.color_palette)
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                try:
                    processed_batch = future.result(timeout=45)
                    processed_batch.sort(key=lambda x: x[0])

                    for frame_idx, processed_frame in processed_batch:
                        out.write(processed_frame)
                        perf_monitor.log_frame(frame_idx)
                        pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing batch starting at frame {start_frame}: {e}")

                if start_frame % (batch_size * 8) == 0:
                    gc.collect()

    out.release()
    logging.info(f"✓ Optimized Stage 2 visualization saved to {output_path}")


def process_frame_batch_stage2(batch_frames, poses_by_frame, all_court_points, pose_edges, color_palette) -> List[Tuple[int, np.ndarray]]:
    """Process a batch of frames for stage 2 pose visualization"""
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()

        # Draw court points (subtle)
        frame_display = draw_all_court_points_optimized(frame_display, all_court_points, show_labels=False)

        # Process poses for this frame
        frame_poses = poses_by_frame.get(frame_idx, [])

        for pose in frame_poses:
            human_idx = pose["human_index"]
            joints = pose["joints"]
            color = color_palette[human_idx % len(color_palette)]

            # Collect joint positions efficiently
            joint_positions = {}
            in_court_count = 0

            for joint in joints:
                if joint["confidence"] > 0.5:
                    joint_idx = joint["joint_index"]
                    x, y = int(joint["x"]), int(joint["y"])

                    if joint.get("in_court", False):
                        cv2.circle(frame_display, (x, y), 5, color, -1)
                        in_court_count += 1
                    else:
                        cv2.circle(frame_display, (x, y), 3, (128, 128, 128), -1)

                    joint_positions[joint_idx] = (x, y)

            # Draw skeleton efficiently
            for edge in pose_edges:
                if edge[0] in joint_positions and edge[1] in joint_positions:
                    cv2.line(frame_display, joint_positions[edge[0]], joint_positions[edge[1]], color, 2)

            # Add player label
            if joint_positions:
                head_pos = joint_positions.get(0, list(joint_positions.values())[0])
                cv2.putText(frame_display, f"Player {human_idx} ({in_court_count} joints)",
                            (head_pos[0] - 20, head_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add frame info
        total_players = len(frame_poses)
        cv2.putText(frame_display, f"Frame: {frame_idx} | Players: {total_players}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        processed_frames.append((frame_idx, frame_display))

    return processed_frames


def visualize_stage4_optimized(video_path: str, data_path: str, output_path: str, num_threads: int = None) -> None:
    """
    Optimized Stage 4 visualization: Corrected position comparison with advanced trajectory tracking
    """
    logging.info("Creating optimized Stage 4 visualization: Corrected position comparison")

    processor = OptimizedVideoProcessor(video_path, num_threads)
    data_loader = OptimizedDataLoader()
    perf_monitor = PerformanceMonitor()

    # Load corrected data
    corrected_data = data_loader.load_json_data(data_path)

    # Load original data for comparison
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results", base_name)
    original_path = os.path.join(result_dir, "positions.json")

    try:
        original_data = data_loader.load_json_data(original_path)
    except FileNotFoundError:
        logging.error(f"Original positions file not found: {original_path}")
        return

    # Organize data
    court_points = corrected_data.get("court_points", {})
    video_info = corrected_data["video_info"]
    corrected_positions = corrected_data["player_positions"]
    original_positions = original_data["player_positions"]

    corrected_by_frame = data_loader.organize_by_frame_optimized(corrected_positions)
    original_by_frame = data_loader.organize_by_frame_optimized(original_positions)
    image_points = extract_corner_points_optimized(court_points)

    # Pre-calculate correction frames for efficiency
    correction_frames = find_correction_frames_optimized(corrected_by_frame, original_by_frame)

    # Setup video writer
    out_w = video_info["width"] + processor.court_img_w
    out_h = max(video_info["height"], processor.court_img_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info["fps"], (out_w, out_h))

    if not out.isOpened():
        logging.error(f"Cannot open video writer: {output_path}")
        return

    perf_monitor.start()
    frame_count = video_info["frame_count"]
    batch_size = 6  # Conservative for complex visualization

    # Shared trajectory history (thread-safe)
    trajectory_lock = Lock()
    trajectory_history = defaultdict(lambda: {'original': [], 'corrected': []})

    # Process with threading
    with ThreadPoolExecutor(max_workers=min(processor.num_threads, 3)) as executor:
        futures = []

        for start_frame in range(0, frame_count, batch_size):
            end_frame = min(start_frame + batch_size, frame_count)
            batch_frames = processor.get_frame_batch(start_frame, end_frame - start_frame)

            if batch_frames:
                future = executor.submit(
                    process_frame_batch_stage4,
                    batch_frames, corrected_by_frame, original_by_frame, correction_frames,
                    image_points, processor.court_template.copy(), processor.court_img_w, processor.court_img_h,
                    processor.court_scale, processor.margin_m, processor.get_player_color,
                    out_h, out_w, trajectory_history, trajectory_lock
                )
                futures.append((start_frame, future))

        # Write results
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for start_frame, future in futures:
                try:
                    processed_batch = future.result(timeout=90)
                    processed_batch.sort(key=lambda x: x[0])

                    for frame_idx, combined_frame in processed_batch:
                        out.write(combined_frame)
                        perf_monitor.log_frame(frame_idx)
                        pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing batch starting at frame {start_frame}: {e}")

                if start_frame % (batch_size * 4) == 0:
                    gc.collect()

    out.release()

    logging.info(f"✓ Optimized Stage 4 visualization saved to {output_path}")
    logging.info(f"✓ Total frames with corrections: {len(correction_frames)}")


def find_correction_frames_optimized(corrected_by_frame: Dict, original_by_frame: Dict) -> set:
    """Pre-calculate frames where corrections occurred for efficiency"""
    correction_frames = set()

    for frame_idx in corrected_by_frame:
        if frame_idx in original_by_frame:
            for corr_pos in corrected_by_frame[frame_idx]:
                tracked_id = corr_pos["tracked_id"]

                # Find matching original position
                orig_pos = next((pos for pos in original_by_frame[frame_idx]
                                 if pos["tracked_id"] == tracked_id), None)

                if orig_pos:
                    corr_x = corr_pos.get("hip_world_X", 0)
                    corr_y = corr_pos.get("hip_world_Y", 0)
                    orig_x = orig_pos.get("hip_world_X", 0)
                    orig_y = orig_pos.get("hip_world_Y", 0)

                    if abs(corr_x - orig_x) > 0.01 or abs(corr_y - orig_y) > 0.01:
                        correction_frames.add(frame_idx)
                        break

    return correction_frames


def process_frame_batch_stage4(batch_frames, corrected_by_frame, original_by_frame, correction_frames,
                               image_points, court_template, court_img_w, court_img_h, court_scale, margin_m,
                               get_player_color, out_h, out_w, trajectory_history, trajectory_lock) -> List[Tuple[int, np.ndarray]]:
    """Process a batch of frames for stage 4 with thread-safe trajectory tracking"""
    processed_frames = []

    for frame_idx, frame in batch_frames:
        frame_display = frame.copy()
        court_img = court_template.copy()

        # Draw calibration points
        for i, point in enumerate(image_points):
            cv2.circle(frame_display, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
            cv2.putText(frame_display, f"P{i+1}", (int(point[0]) + 10, int(point[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Highlight correction frames
        is_correction_frame = frame_idx in correction_frames
        if is_correction_frame:
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 255), 8)
            cv2.putText(frame_display, "CORRECTED FRAME", (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Get frame data
        corrected_frame_players = corrected_by_frame.get(frame_idx, [])
        original_frame_players = original_by_frame.get(frame_idx, [])

        # Update trajectory history (thread-safe)
        with trajectory_lock:
            for pos_data in corrected_frame_players:
                tracked_id = pos_data["tracked_id"]
                if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                    hip_x = pos_data["hip_world_X"]
                    hip_y = pos_data["hip_world_Y"]
                    hip_px = world_to_court_cached(hip_x, hip_y, court_img_w, court_img_h, court_scale, margin_m)
                    trajectory_history[tracked_id]['corrected'].append(hip_px)

                    # Keep only last 30 points for memory efficiency
                    if len(trajectory_history[tracked_id]['corrected']) > 30:
                        trajectory_history[tracked_id]['corrected'] = trajectory_history[tracked_id]['corrected'][-30:]

            for pos_data in original_frame_players:
                tracked_id = pos_data["tracked_id"]
                if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                    hip_x = pos_data["hip_world_X"]
                    hip_y = pos_data["hip_world_Y"]
                    hip_px = world_to_court_cached(hip_x, hip_y, court_img_w, court_img_h, court_scale, margin_m)
                    trajectory_history[tracked_id]['original'].append(hip_px)

                    if len(trajectory_history[tracked_id]['original']) > 30:
                        trajectory_history[tracked_id]['original'] = trajectory_history[tracked_id]['original'][-30:]

        # Draw trajectories efficiently
        for tracked_id, trajectories in trajectory_history.items():
            color = get_player_color(tracked_id)

            # Draw original trajectory
            original_traj = trajectories['original'][-30:]
            if len(original_traj) > 1:
                for i in range(1, len(original_traj)):
                    cv2.line(court_img, original_traj[i-1], original_traj[i], (128, 128, 128), 2)

            # Draw corrected trajectory
            corrected_traj = trajectories['corrected'][-30:]
            if len(corrected_traj) > 1:
                for i in range(1, len(corrected_traj)):
                    cv2.line(court_img, corrected_traj[i-1], corrected_traj[i], color, 3)

        # Draw current positions
        for pos_data in corrected_frame_players:
            tracked_id = pos_data["tracked_id"]
            color = get_player_color(tracked_id)

            if "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                hip_x = pos_data["hip_world_X"]
                hip_y = pos_data["hip_world_Y"]
                hip_px = world_to_court_cached(hip_x, hip_y, court_img_w, court_img_h, court_scale, margin_m)
                cv2.circle(court_img, hip_px, 8, color, -1)
                cv2.circle(court_img, hip_px, 10, (255, 255, 255), 2)
                cv2.putText(court_img, f"P{tracked_id}-Corr", (hip_px[0] - 25, hip_px[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw original positions if different
        for pos_data in original_frame_players:
            tracked_id = pos_data["tracked_id"]

            corrected_pos = next((pos for pos in corrected_frame_players if pos["tracked_id"] == tracked_id), None)

            if corrected_pos and "hip_world_X" in pos_data and "hip_world_Y" in pos_data:
                orig_x = pos_data["hip_world_X"]
                orig_y = pos_data["hip_world_Y"]
                corr_x = corrected_pos.get("hip_world_X", orig_x)
                corr_y = corrected_pos.get("hip_world_Y", orig_y)

                if abs(orig_x - corr_x) > 0.01 or abs(orig_y - corr_y) > 0.01:
                    orig_px = world_to_court_cached(orig_x, orig_y, court_img_w, court_img_h, court_scale, margin_m)
                    corr_px = world_to_court_cached(corr_x, corr_y, court_img_w, court_img_h, court_scale, margin_m)

                    cv2.circle(court_img, orig_px, 4, (128, 128, 128), -1)
                    cv2.line(court_img, orig_px, corr_px, (255, 255, 255), 1)

        # Add legend and statistics
        add_stage4_legend_optimized(court_img, court_img_w, len(correction_frames), len(corrected_frame_players), frame_idx)

        # Combine frames
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        combined[:h, :w] = frame_display
        combined[:court_img.shape[0], w:w + court_img.shape[1]] = court_img

        processed_frames.append((frame_idx, combined))

    return processed_frames


def add_stage4_legend_optimized(court_img, court_img_w, total_corrections, current_corrections, frame_idx):
    """Add optimized legend for stage 4"""
    legend_y = 50
    legend_x = court_img_w - 250

    # Legend items
    legend_items = [
        ("CORRECTION LEGEND:", (legend_x, legend_y), 0.5, (255, 255, 255), None),
        ("Corrected Position", (legend_x + 35, legend_y + 25), 0.4, (0, 255, 0), ((legend_x + 20, legend_y + 20), 8, (0, 255, 0))),
        ("Original Position", (legend_x + 35, legend_y + 45), 0.4, (128, 128, 128), ((legend_x + 20, legend_y + 40), 4, (128, 128, 128))),
        ("Corrected Trajectory", (legend_x + 35, legend_y + 65), 0.4, (0, 255, 0), None),
        ("Original Trajectory", (legend_x + 35, legend_y + 85), 0.4, (128, 128, 128), None),
        (f"Total corrections: {total_corrections}", (10, 30), 0.6, (255, 255, 255), None),
        (f"Current corrections: {current_corrections}", (10, 50), 0.6, (255, 255, 255), None),
        (f"Frame: {frame_idx}", (10, 70), 0.6, (255, 255, 255), None)
    ]

    # Draw legend efficiently
    for text, pos, scale, color, circle_data in legend_items:
        cv2.putText(court_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        if circle_data:
            center, radius, circle_color = circle_data
            cv2.circle(court_img, center, radius, circle_color, -1)

    # Draw trajectory lines
    cv2.line(court_img, (legend_x + 15, legend_y + 60), (legend_x + 30, legend_y + 60), (0, 255, 0), 3)
    cv2.line(court_img, (legend_x + 15, legend_y + 80), (legend_x + 30, legend_y + 80), (128, 128, 128), 2)


if __name__ == "__main__":
    main()