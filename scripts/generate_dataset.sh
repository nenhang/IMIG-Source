#!/bin/bash

# ==============================================
# Script: generate_dataset.sh
# Description: Automated pipeline for image generation and processing
# Usage: ./scripts/generate_dataset.sh
# ==============================================

# --- Directory Setup ---
# Get the absolute path of the script's directory (scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Change to project root directory with error handling
cd "$PROJECT_ROOT" || {
    echo "Failed to change to project root directory"
    exit 1
}

# --- Logging Functions ---
# Helper function for consistent logging with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling function that logs and exits
exit_on_error() {
    log "ERROR: $1"
    exit 1
}

# --- Pre-flight Checks ---
# Verify conda is available in the system
if ! command -v conda &> /dev/null; then
    exit_on_error "Conda not found. Please install Anaconda/Miniconda first."
fi

# Initialize conda for shell script (needed for some environments)
eval "$(conda shell.bash hook)" 2>/dev/null

# ==================== PHASE 1: Image Generation and Annotation ====================
log "=== Starting Phase 1: Image Generation and Annotation ==="

# Activate the generation environment
# Using conda activate in a script requires proper shell initialization
conda activate imig-gen || exit_on_error "Failed to activate imig-gen environment"

# Generate initial images from prompts
python src/generate_dataset.py --task generate_images || \
    exit_on_error "generate_images failed"

# Annotate generated images with bounding boxes
conda activate imig-tool || python src/generate_dataset.py --task annotate_images || \
    exit_on_error "annotate_images failed"

# Crop objects based on bounding boxes
python src/generate_dataset.py --task crop_images || \
    exit_on_error "crop_images failed"

# Repaint instances with Kontext
conda activate imig-gen || python src/generate_dataset.py --task repaint_with_kontext || \
    exit_on_error "repaint_with_kontext failed"

# Validate bounding boxes after repainting
conda activate imig-tool || python src/generate_dataset.py --task get_valid_bboxes || \
    exit_on_error "get_valid_bboxes failed"

# ==================== PHASE 2: Image Segmentation ====================
log "=== Starting Phase 2: Image Segmentation ==="

# Switch to the tool environment for segmentation tasks
exit_on_error "Failed to activate imig-tool environment"

# Segment object instances from cropped images
python src/generate_dataset.py --task segment_instance_images || \
    exit_on_error "segment_instance_images failed"

# Segment repainted instances
python src/generate_dataset.py --task segment_repainted_images || \
    exit_on_error "segment_repainted_images failed"

# ==================== PHASE 3: Data Filtering ====================
log "=== Starting Phase 3: Data Filtering ==="

# Switch back to generation environment for final filtering
conda activate imig-gen || exit_on_error "Failed to activate imig-gen environment"

# Filter out failed samples to produce final dataset
python src/generate_dataset.py --task filter_prompts || \
    exit_on_error "filter_prompts failed"

# ==================== COMPLETION ====================
log "=== Pipeline completed successfully ==="