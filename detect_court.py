import argparse
import logging
import os
import subprocess


def run_detect_script(video_path, output_path):
    while True:
        print("Attempting to detect court")
        result = subprocess.run(f'./resources/detect {video_path} {output_path}', shell=True, capture_output=True, text=True)
        print(result.stdout)
        if "Processing error: Not enough line candidates were found." not in result.stdout:
            break

def main(video_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join("results/", f"{base_name}")
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f"Result directory for {base_name} will be at {result_dir}")
    run_detect_script(video_path, os.path.join(result_dir, "court.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video for court detection")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    args = parser.parse_args()
    main(args.video_path)