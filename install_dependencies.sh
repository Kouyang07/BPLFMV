#!/bin/bash

# BPLFMV (Badminton Player Location and Feature Motion Vector) Dependencies Installation Script
# This script installs all dependencies required for the badminton analysis pipeline

set -e  # Exit on error

echo "=============================================================="
echo "BPLFMV Dependencies Installation Script"
echo "=============================================================="
echo "This script will install dependencies for:"
echo "1. Court detection"
echo "2. Pose estimation (YOLO11 and ViTPose-G)"
echo "3. 3D position tracking"
echo "4. Jump analysis and correction"
echo "5. Visualization"
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Python command
detect_python() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$PYTHON_VERSION" = "3" ]; then
            echo "python"
        else
            print_error "Python 3 is required but not found"
            exit 1
        fi
    else
        print_error "Python 3 is required but not found"
        exit 1
    fi
}

# Function to detect pip command
detect_pip() {
    if command_exists pip3; then
        echo "pip3"
    elif command_exists pip; then
        echo "pip"
    else
        print_error "pip is required but not found"
        exit 1
    fi
}

# Detect system
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/' | head -1)
        if [ ! -z "$CUDA_VERSION" ]; then
            print_status "CUDA detected: version $CUDA_VERSION"
            return 0
        fi
    fi
    print_warning "CUDA not detected, will install CPU-only versions"
    return 1
}

# Function to check PyTorch installation
check_pytorch() {
    $PYTHON -c "import torch; print(f'PyTorch {torch.__version__} detected')" 2>/dev/null
}

# Initialize variables
PYTHON=$(detect_python)
PIP=$(detect_pip)
SYSTEM=$(detect_system)
HAS_CUDA=false

print_status "Detected Python: $PYTHON"
print_status "Detected pip: $PIP"
print_status "Detected system: $SYSTEM"

# Check CUDA
if check_cuda; then
    HAS_CUDA=true
fi

print_section "Checking existing Python environment"

# Check Python version
PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check if in virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_status "Virtual environment detected: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected. It's recommended to use a virtual environment."
    echo "To create one: python3 -m venv bplfmv_env && source bplfmv_env/bin/activate"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_section "Installing core dependencies"

# Upgrade pip
print_status "Upgrading pip..."
$PIP install --upgrade pip

# Install essential Python packages
print_status "Installing essential Python packages..."
$PIP install --upgrade setuptools wheel

print_section "Installing PyTorch and related packages"

# Check if PyTorch is already installed
if check_pytorch; then
    print_warning "PyTorch is already installed. Skipping PyTorch installation."
    read -p "Do you want to reinstall PyTorch? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        INSTALL_PYTORCH=true
    else
        INSTALL_PYTORCH=false
    fi
else
    INSTALL_PYTORCH=true
fi

if [ "$INSTALL_PYTORCH" = true ]; then
    print_status "Installing PyTorch..."
    if [ "$HAS_CUDA" = true ]; then
        print_status "Installing PyTorch with CUDA support..."
        $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU-only version..."
        $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

print_section "Installing OpenMMLab ecosystem"

# Install OpenMIM (OpenMMLab package manager)
print_status "Installing OpenMIM..."
$PIP install -U openmim

# Install MMEngine
print_status "Installing MMEngine..."
mim install mmengine

# Install MMCV
print_status "Installing MMCV..."
if [ "$HAS_CUDA" = true ]; then
    print_status "Installing MMCV with CUDA support..."
    mim install "mmcv>=2.0.1"
else
    print_status "Installing MMCV CPU-only version..."
    mim install "mmcv-lite>=2.0.1"
fi

# Install MMPose
print_status "Installing MMPose..."
mim install mmpose

print_section "Installing YOLO and pose estimation dependencies"

# Install Ultralytics YOLO
print_status "Installing Ultralytics YOLO..."
$PIP install ultralytics

# Install additional pose estimation dependencies
print_status "Installing pose estimation dependencies..."
$PIP install chumpy  # Required for some MMPose models

print_section "Installing computer vision and image processing libraries"

# OpenCV
print_status "Installing OpenCV..."
$PIP install opencv-python

# Additional CV libraries
print_status "Installing additional computer vision libraries..."
$PIP install Pillow
$PIP install imageio
$PIP install scikit-image

print_section "Installing scientific computing libraries"

# NumPy and SciPy
print_status "Installing NumPy and SciPy..."
$PIP install numpy scipy

# Matplotlib and visualization
print_status "Installing visualization libraries..."
$PIP install matplotlib seaborn

print_section "Installing data processing libraries"

# Pandas and data processing
print_status "Installing data processing libraries..."
$PIP install pandas

# JSON and CSV processing (built-in, but ensuring compatibility)
print_status "Installing additional data processing libraries..."
$PIP install jsonschema

print_section "Installing video processing dependencies"

# FFmpeg-python for video processing
print_status "Installing video processing libraries..."
$PIP install ffmpeg-python

print_section "Installing ML and analysis libraries"

# Scikit-learn for machine learning utilities
print_status "Installing machine learning libraries..."
$PIP install scikit-learn

# Progress bars and utilities
print_status "Installing utility libraries..."
$PIP install tqdm
$PIP install pathlib2

print_section "Installing optional dependencies for enhanced functionality"

# Install additional packages that might be useful
print_status "Installing optional dependencies..."

# For better numeric computations
$PIP install numba

# For advanced signal processing (used in jump detection)
$PIP install scipy

# For configuration management
$PIP install pyyaml

# For argument parsing enhancements
$PIP install argparse

# For logging enhancements
$PIP install colorlog

print_section "Verifying installations"

# Function to check if a package is importable
check_import() {
    $PYTHON -c "import $1" 2>/dev/null && echo -e "${GREEN}✓${NC} $1" || echo -e "${RED}✗${NC} $1"
}

print_status "Checking core dependencies..."
check_import torch
check_import torchvision
check_import cv2
check_import numpy
check_import scipy
check_import matplotlib
check_import pandas
check_import tqdm
check_import json
check_import argparse

print_status "Checking OpenMMLab dependencies..."
check_import mmengine
check_import mmcv
check_import mmpose

print_status "Checking YOLO dependencies..."
check_import ultralytics

print_status "Checking additional dependencies..."
check_import pathlib
check_import collections
check_import platform

print_section "Testing PyTorch functionality"

print_status "Testing PyTorch installation..."
$PYTHON -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('Running on CPU mode')
"

print_section "Testing MMPose functionality"

print_status "Testing MMPose installation..."
$PYTHON -c "
try:
    import mmpose
    print(f'MMPose version: {mmpose.__version__}')
    print('MMPose installation successful!')
except Exception as e:
    print(f'MMPose test failed: {e}')
"

print_section "Creating directory structure"

print_status "Creating required directories..."
mkdir -p results
mkdir -p samples
mkdir -p resources

print_status "Directory structure created:"
echo "  results/     - For storing analysis results"
echo "  samples/     - For input video files"
echo "  resources/   - For model weights and resources"

print_section "Installation summary"

echo "=============================================================="
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "=============================================================="
echo ""
echo "Installed components:"
echo "✓ PyTorch and torchvision"
echo "✓ OpenMMLab ecosystem (MMEngine, MMCV, MMPose)"
echo "✓ Ultralytics YOLO"
echo "✓ OpenCV and image processing libraries"
echo "✓ Scientific computing libraries (NumPy, SciPy, Matplotlib)"
echo "✓ Data processing libraries (Pandas)"
echo "✓ Video processing libraries"
echo "✓ Utility libraries"
echo ""
echo "Next steps:"
echo "1. Place your model weights in the 'resources/' directory"
echo "2. Place your video files in the 'samples/' directory"
echo "3. Run the analysis scripts:"
echo "   - python detect_court.py <video_path>"
echo "   - python detect_pose.py <video_path>"
echo "   - python calculate_location.py <video_path>"
echo "   - python remove_artifact.py <video_path>"
echo "   - python visualize.py <video_path> --stage <1-4>"
echo ""
echo "For enhanced pose estimation with ViTPose-G:"
echo "   - python detect_pose_enhanced.py <video_path>"
echo ""
echo -e "${YELLOW}Note:${NC} If you encounter any issues:"
echo "1. Make sure you're in the correct virtual environment"
echo "2. Check that all required model files are downloaded"
echo "3. Verify video file formats are supported"
echo "4. Check GPU memory if using CUDA"
echo ""
echo "=============================================================="

# Optional: Download YOLO model
read -p "Do you want to download the default YOLO pose model? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading YOLO11 pose model..."
    $PYTHON -c "
from ultralytics import YOLO
model = YOLO('yolo11x-pose.pt')
print('YOLO11 pose model downloaded successfully!')
"
fi

print_status "Installation script completed!"