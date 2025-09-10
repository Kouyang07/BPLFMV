#!/usr/bin/env python3
"""
CVAT to World Coordinates Converter

Converts CVAT point annotations to world coordinates using homography.
Usage: python cvat_converter.py <cvat_annotations.xml> <video_path>
"""

import sys
import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CVATAnnotationConverter:
    """Convert CVAT annotations to world coordinates using homography."""

    COURT_WIDTH = 6.1
    COURT_LENGTH = 13.4

    def __init__(self, video_path: str):
        """Initialize converter with video path."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem

        self.results_dir = Path("results") / self.video_name
        self.pose_file = self.results_dir / "pose.json"

        self.court_points = None
        self.homography_matrix = None
        self.load_court_data()

    def load_court_data(self):
        """Load court points and calculate homography."""
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Court data not found: {self.pose_file}")

        with open(self.pose_file, 'r') as f:
            data = json.load(f)

        if 'court_points' in data:
            self.court_points = data['court_points']
        elif 'all_court_points' in data:
            self.court_points = data['all_court_points']
        else:
            raise ValueError("No court points found")

        self.calculate_homography()

    def calculate_homography(self):
        """Calculate homography matrix."""
        required_corners = ['P1', 'P2', 'P3', 'P4']

        image_points = []
        for corner in required_corners:
            coords = self.court_points[corner]
            if isinstance(coords, list):
                image_points.append([float(coords[0]), float(coords[1])])
            else:
                image_points.append([float(coords['x']), float(coords['y'])])

        image_points = np.array(image_points, dtype=np.float32)

        world_points = np.array([
            [0, 0],                                    # P1
            [0, self.COURT_LENGTH],                   # P2
            [self.COURT_WIDTH, self.COURT_LENGTH],    # P3
            [self.COURT_WIDTH, 0]                     # P4
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(
            image_points, world_points, cv2.RANSAC
        )

        if self.homography_matrix is None:
            raise ValueError("Failed to calculate homography")

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.homography_matrix)
        return float(world_point[0][0][0]), float(world_point[0][0][1])

    def parse_cvat_xml(self, xml_path: str) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Parse CVAT XML annotations."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = {}  # frame -> {player_id: (x, y)}

        # Parse track elements (video format)
        for track in root.findall('.//track'):
            label = track.get('label', 'player')

            # Map labels to player IDs: P0->0, P1->1
            if label == 'P0' or 'p0' in label.lower() or label == '0':
                player_id = 0
            elif label == 'P1' or 'p1' in label.lower() or label == '1':
                player_id = 1
            else:
                import re
                numbers = re.findall(r'\d+', label)
                if numbers:
                    player_id = int(numbers[0])
                else:
                    player_id = 0

            for points in track.findall('.//points'):
                frame_num = int(points.get('frame', 0))
                outside = int(points.get('outside', 0))

                if outside:
                    continue

                points_str = points.get('points', '')

                if points_str:
                    coords = points_str.split(',')
                    if len(coords) >= 2:
                        pixel_x = float(coords[0])
                        pixel_y = float(coords[1])

                        if frame_num not in annotations:
                            annotations[frame_num] = {}

                        annotations[frame_num][player_id] = (pixel_x, pixel_y)

        # Fallback: parse image elements (image format)
        if not annotations:
            for image in root.findall('.//image'):
                frame_id = int(image.get('id'))
                frame_name = image.get('name', '')

                try:
                    if 'frame_' in frame_name:
                        frame_num = int(frame_name.split('frame_')[1].split('.')[0])
                    else:
                        frame_num = frame_id
                except:
                    frame_num = frame_id

                frame_annotations = {}

                for points in image.findall('.//points'):
                    label = points.get('label', 'player')
                    points_str = points.get('points', '')

                    if points_str:
                        coords = points_str.split(',')
                        if len(coords) >= 2:
                            pixel_x = float(coords[0])
                            pixel_y = float(coords[1])

                            # Map labels to player IDs
                            if 'player0' in label.lower() or label == '0':
                                player_id = 0
                            elif 'player1' in label.lower() or label == '1':
                                player_id = 1
                            else:
                                player_id = len(frame_annotations)

                            frame_annotations[player_id] = (pixel_x, pixel_y)

                if frame_annotations:
                    annotations[frame_num] = frame_annotations

        return annotations

    def convert_to_world_coordinates(self, annotations: Dict[int, Dict[int, Tuple[float, float]]]) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Convert pixel annotations to world coordinates."""
        world_annotations = {}

        for frame_num, frame_data in annotations.items():
            world_frame = {}

            for player_id, (pixel_x, pixel_y) in frame_data.items():
                try:
                    world_x, world_y = self.pixel_to_world(pixel_x, pixel_y)

                    if (0 <= world_x <= self.COURT_WIDTH and
                        0 <= world_y <= self.COURT_LENGTH):
                        world_frame[player_id] = (world_x, world_y)

                except Exception:
                    continue

            if world_frame:
                world_annotations[frame_num] = world_frame

        return world_annotations

    def save_annotations(self, annotations: Dict[int, Dict[int, Tuple[float, float]]]):
        """Save annotations in pipeline format."""
        serializable_annotations = {}
        for frame_num, frame_data in annotations.items():
            serializable_annotations[str(frame_num)] = {
                str(player_id): list(position) for player_id, position in frame_data.items()
            }

        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        save_data = {
            'video_path': str(self.video_path),
            'video_name': self.video_name,
            'total_frames': total_frames,
            'fps': fps,
            'video_resolution': {'width': width, 'height': height},
            'court_dimensions': {'width': self.COURT_WIDTH, 'length': self.COURT_LENGTH},
            'court_points': self.court_points,
            'annotations': serializable_annotations,
            'annotation_info': {
                'total_annotated_frames': len(annotations),
                'annotation_method': 'cvat_converted_to_world_coordinates',
                'coverage_percentage': (len(annotations) / total_frames) * 100,
                'players_supported': [0, 1],
                'data_source': 'cvat_xml_export'
            }
        }

        annotation_file = self.results_dir / "cvat_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        self.export_for_evaluation(annotations)

    def export_for_evaluation(self, annotations: Dict[int, Dict[int, Tuple[float, float]]]):
        """Export in evaluation format."""
        player_positions = []

        for frame_index, frame_data in annotations.items():
            for player_id, (world_x, world_y) in frame_data.items():
                position_entry = {
                    'frame_index': frame_index,
                    'player_id': player_id,
                    'world_x': world_x,
                    'world_y': world_y,
                    'annotation_method': 'cvat_converted'
                }
                player_positions.append(position_entry)

        player_positions.sort(key=lambda x: (x['frame_index'], x['player_id']))

        player0_count = sum(1 for p in player_positions if p['player_id'] == 0)
        player1_count = sum(1 for p in player_positions if p['player_id'] == 1)

        export_data = {
            'video_info': {
                'path': str(self.video_path),
                'name': self.video_name
            },
            'court_dimensions': {
                'width': self.COURT_WIDTH,
                'length': self.COURT_LENGTH
            },
            'court_points': self.court_points,
            'player_positions': player_positions,
            'annotation_method': 'cvat_with_homography_conversion',
            'annotation_info': {
                'total_annotated_frames': len(annotations),
                'total_positions': len(player_positions),
                'player0_positions': player0_count,
                'player1_positions': player1_count,
                'coordinate_transformation': 'homography_matrix'
            }
        }

        export_file = self.results_dir / "cvat_ground_truth.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python cvat_converter.py <cvat_annotations.xml> <video_path>")
        sys.exit(1)

    xml_path = sys.argv[1]
    video_path = sys.argv[2]

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    try:
        converter = CVATAnnotationConverter(video_path)
        pixel_annotations = converter.parse_cvat_xml(xml_path)
        world_annotations = converter.convert_to_world_coordinates(pixel_annotations)
        converter.save_annotations(world_annotations)

        total_positions = sum(len(frame_data) for frame_data in world_annotations.values())
        player0_count = sum(1 for frame_data in world_annotations.values() if 0 in frame_data)
        player1_count = sum(1 for frame_data in world_annotations.values() if 1 in frame_data)

        print(f"Converted {len(world_annotations)} frames")
        print(f"Player 0: {player0_count} positions")
        print(f"Player 1: {player1_count} positions")
        print(f"Total: {total_positions} positions")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()