import argparse
import logging
import os
import subprocess
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


class CourtPointStandardizer:
    """Standardizes court point coordinates to consistent clockwise order."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def load_court_points_from_csv(self, csv_path: str) -> Dict[str, List[float]]:
        """Load court points from CSV file."""
        court_points = {}

        with open(csv_path, 'r') as f:
            # Try DictReader first (for files with headers)
            f.seek(0)
            sample_line = f.readline().strip()
            f.seek(0)

            if 'Point' in sample_line and 'X' in sample_line and 'Y' in sample_line:
                # File has headers, use DictReader
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        point_name = row['Point'].strip()
                        x_coord = float(row['X'])
                        y_coord = float(row['Y'])
                        court_points[point_name] = [x_coord, y_coord]
                    except (ValueError, KeyError) as e:
                        if self.debug:
                            print(f"Skipping invalid row: {row} - {e}")
                        continue
            else:
                # File without headers, use regular reader
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            point_name = row[0].strip()
                            x_coord = float(row[1])
                            y_coord = float(row[2])
                            court_points[point_name] = [x_coord, y_coord]
                        except (ValueError, IndexError) as e:
                            if self.debug:
                                print(f"Skipping invalid row: {row} - {e}")
                            continue

        if self.debug:
            print(f"Loaded {len(court_points)} court points from CSV:")
            for name, coords in court_points.items():
                print(f"  {name}: ({coords[0]:.1f}, {coords[1]:.1f})")

        return court_points

    def standardize_court_point_order(self, court_points: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Standardize court points to clockwise order 1,4,3,2 from top-left."""
        print("\n=== Court Point Order Standardization ===")

        # Extract all four points
        points = {
            'P1': court_points.get('P1', court_points.get('1')),
            'P2': court_points.get('P2', court_points.get('2')),
            'P3': court_points.get('P3', court_points.get('3')),
            'P4': court_points.get('P4', court_points.get('4'))
        }

        # Handle different naming conventions
        if any(v is None for v in points.values()):
            # Try numeric keys
            points = {
                'P1': court_points.get('1'),
                'P2': court_points.get('2'),
                'P3': court_points.get('3'),
                'P4': court_points.get('4')
            }

        if any(v is None for v in points.values()):
            raise ValueError("Missing required court points. Expected P1-P4 or 1-4.")

        if self.debug:
            print("Original points:")
            for name, point in points.items():
                print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")

        # Convert to list of (x, y, original_name) for analysis
        point_data = [(point[0], point[1], name) for name, point in points.items()]

        # Find corners based on spatial relationships
        # Sort by Y coordinate first (top vs bottom)
        point_data.sort(key=lambda p: p[1])

        # Split into top two and bottom two points
        top_points = point_data[:2]
        bottom_points = point_data[2:]

        # Within each pair, sort by X coordinate (left vs right)
        top_points.sort(key=lambda p: p[0])  # Left to right
        bottom_points.sort(key=lambda p: p[0])  # Left to right

        # Assign based on clockwise order from top-left: 1(TL), 4(TR), 3(BR), 2(BL)
        top_left = top_points[0]      # P1: Top-left
        top_right = top_points[1]     # P4: Top-right
        bottom_left = bottom_points[0]  # P2: Bottom-left
        bottom_right = bottom_points[1] # P3: Bottom-right

        # Create the standardized mapping
        standardized_points = {
            'P1': [top_left[0], top_left[1]],      # Top-left
            'P2': [bottom_left[0], bottom_left[1]], # Bottom-left
            'P3': [bottom_right[0], bottom_right[1]], # Bottom-right
            'P4': [top_right[0], top_right[1]]     # Top-right
        }

        # Check if reordering was needed
        original_order = [points['P1'], points['P2'], points['P3'], points['P4']]
        new_order = [standardized_points['P1'], standardized_points['P2'],
                     standardized_points['P3'], standardized_points['P4']]

        reordering_needed = not all(
            abs(orig[0] - new[0]) < 1 and abs(orig[1] - new[1]) < 1
            for orig, new in zip(original_order, new_order)
        )

        if reordering_needed:
            print("🔄 Court point reordering detected and applied:")
            print(f"  Original P1 ({points['P1'][0]:.1f}, {points['P1'][1]:.1f}) -> New P1 ({standardized_points['P1'][0]:.1f}, {standardized_points['P1'][1]:.1f}) [{top_left[2]}]")
            print(f"  Original P2 ({points['P2'][0]:.1f}, {points['P2'][1]:.1f}) -> New P2 ({standardized_points['P2'][0]:.1f}, {standardized_points['P2'][1]:.1f}) [{bottom_left[2]}]")
            print(f"  Original P3 ({points['P3'][0]:.1f}, {points['P3'][1]:.1f}) -> New P3 ({standardized_points['P3'][0]:.1f}, {standardized_points['P3'][1]:.1f}) [{bottom_right[2]}]")
            print(f"  Original P4 ({points['P4'][0]:.1f}, {points['P4'][1]:.1f}) -> New P4 ({standardized_points['P4'][0]:.1f}, {standardized_points['P4'][1]:.1f}) [{top_right[2]}]")
        else:
            print("✓ Court points are already in correct clockwise order (1,4,3,2 from top-left)")

        print("Standardized clockwise order: P1(TL) -> P4(TR) -> P3(BR) -> P2(BL)")
        print("==========================================\n")

        return standardized_points

    def save_standardized_court_points(self, standardized_points: Dict[str, List[float]],
                                       output_path: str, original_csv_path: str) -> None:
        """Save standardized court points back to CSV and create JSON backup."""
        # First, read the original CSV to preserve all points (not just P1-P4)
        original_points = {}

        with open(original_csv_path, 'r') as f:
            reader = csv.reader(f)
            header_skipped = False

            for row in reader:
                if len(row) >= 3:
                    # Skip header row if it contains non-numeric data
                    if not header_skipped:
                        try:
                            float(row[1])
                            header_skipped = True
                        except ValueError:
                            header_skipped = True
                            continue

                    try:
                        point_name = row[0].strip()
                        x_coord = float(row[1])
                        y_coord = float(row[2])
                        original_points[point_name] = [x_coord, y_coord]
                    except (ValueError, IndexError):
                        continue

        # Update with standardized P1-P4 points, keep all other points
        for point_name, coords in standardized_points.items():
            original_points[point_name] = coords

        # Save to CSV with proper header format expected by pose detection
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header that matches pose detection expectations
            writer.writerow(['Point', 'X', 'Y'])

            # Write standardized corner points first (P1-P4)
            for point_name in ['P1', 'P2', 'P3', 'P4']:
                if point_name in original_points:
                    point = original_points[point_name]
                    writer.writerow([point_name, point[0], point[1]])

            # Write all other points (P5, P6, etc., NetPole1, NetPole2, etc.)
            for point_name, point in sorted(original_points.items()):
                if point_name not in ['P1', 'P2', 'P3', 'P4']:
                    writer.writerow([point_name, point[0], point[1]])

        # Also save JSON version for easier integration
        json_path = output_path.replace('.csv', '_standardized.json')
        with open(json_path, 'w') as f:
            json.dump({
                'court_points': standardized_points,
                'all_court_points': original_points,
                'standardization_info': {
                    'order': 'Clockwise from top-left: P1(TL) -> P4(TR) -> P3(BR) -> P2(BL)',
                    'source_csv': original_csv_path,
                    'standardized': True,
                    'format': 'Point,X,Y header maintained for pose detection compatibility'
                }
            }, f, indent=2)

        print(f"Standardized court points saved to: {output_path}")
        print(f"JSON backup saved to: {json_path}")
        print(f"Maintained format: Point,X,Y with {len(original_points)} total points")


def run_detect_script(video_path, output_path):
    """Run the court detection script with retry logic."""
    while True:
        print("Attempting to detect court")
        result = subprocess.run(f'./resources/detect {video_path} {output_path}',
                                shell=True, capture_output=True, text=True)
        print(result.stdout)
        if "Processing error: Not enough line candidates were found." not in result.stdout:
            break

    return result.returncode == 0


def main(video_path, debug=False):
    """Main function with court point standardization."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results/", f"{base_name}")
    os.makedirs(result_dir, exist_ok=True)

    court_csv_path = os.path.join(result_dir, "court.csv")

    logging.info(f"Result directory for {base_name} will be at {result_dir}")

    # Run court detection
    detection_success = run_detect_script(video_path, court_csv_path)

    if not detection_success:
        logging.error("Court detection failed")
        return False

    # Check if court points were detected
    if not os.path.exists(court_csv_path):
        logging.error(f"Court detection output not found: {court_csv_path}")
        return False

    # Debug: Show CSV content before processing
    if debug:
        print(f"\n=== CSV Content Debug ===")
        with open(court_csv_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print(f"Line {i+1}: {line.strip()}")
        print("========================\n")

    try:
        # Initialize standardizer
        standardizer = CourtPointStandardizer(debug=debug)

        # Load court points from CSV
        print(f"Loading court points from: {court_csv_path}")
        court_points = standardizer.load_court_points_from_csv(court_csv_path)

        if len(court_points) < 4:
            logging.error(f"Insufficient court points detected: {len(court_points)}. Expected at least 4.")
            return False

        # Standardize court point order
        standardized_points = standardizer.standardize_court_point_order(court_points)

        # Save standardized points
        standardizer.save_standardized_court_points(
            standardized_points, court_csv_path, court_csv_path
        )

        logging.info("Court detection and standardization completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Error during court point standardization: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video for court detection with standardization")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    success = main(args.video_path, debug=args.debug)
    exit(0 if success else 1)