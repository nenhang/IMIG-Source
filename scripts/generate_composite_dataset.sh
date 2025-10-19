#!/bin/bash

# ==============================================
# Script: generate_composite_dataset.sh
# Description: Automated pipeline for composite image generation and processing
# Usage: ./scripts/generate_composite_dataset.sh
# ==============================================

# --- Directory Setup ---
# Get absolute path of script directory for reliable execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Change to project root directory with error handling
cd "$PROJECT_ROOT" || {
    echo "Failed to change to project root directory"
    exit 1
}

# --- Logging Functions ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

exit_on_error() {
    log "ERROR: $1"
    exit 1
}

# --- Pre-flight Checks ---
if ! command -v conda &> /dev/null; then
    exit_on_error "Conda not found. Please install Anaconda/Miniconda first."
fi

# Initialize conda for shell script (needed for some environments)
eval "$(conda shell.bash hook)" 2>/dev/null

# ==================== PHASE 1: Reference Image Generation ====================
log "=== Starting Phase 1: Reference Image Generation ==="

conda activate imig-gen || exit_on_error "Failed to activate imig-gen environment"

# Generate reference images from prompts
python src/generate_composite_dataset.py --task generate_reference_images || \
    exit_on_error "generate_reference_images failed"

# Crop objects from reference images
python src/generate_composite_dataset.py --task crop_reference_images || \
    exit_on_error "crop_reference_images failed"

# ==================== PHASE 2: Reference Image Segmentation ====================
log "=== Starting Phase 2: Reference Image Segmentation ==="

conda activate imig-tool || exit_on_error "Failed to activate imig-tool environment"

# Segment reference images
python src/generate_composite_dataset.py --task segment_reference_images || \
    exit_on_error "segment_reference_images failed"

# ==================== PHASE 3: Composite Image Generation ====================
log "=== Starting Phase 3: Composite Image Generation ==="

conda activate imig-gen || exit_on_error "Failed to activate imig-gen environment"

# Generate composite images
python src/generate_composite_dataset.py --task generate_composite_images || \
    exit_on_error "generate_composite_images failed"

# Annotate composite images with bounding boxes
python src/generate_composite_dataset.py --task annotate_images || \
    exit_on_error "annotate_images failed"

# Crop objects from composite images
python src/generate_composite_dataset.py --task crop_composite_images || \
    exit_on_error "crop_composite_images failed"

# ==================== PHASE 4: Composite Image Segmentation ====================
log "=== Starting Phase 4: Composite Image Segmentation ==="

conda activate imig-tool || exit_on_error "Failed to activate imig-tool environment"

# Segment instances from composite images
python src/generate_composite_dataset.py --task segment_instance_images || \
    exit_on_error "segment_instance_images failed"

# ==================== PHASE 5: Quality Assessment ====================
log "=== Starting Phase 5: Quality Assessment ==="

conda activate imig-gen || exit_on_error "Failed to activate imig-gen environment"

# Calculate DINO similarity scores
python src/generate_composite_dataset.py --task cal_dino_score || \
    exit_on_error "cal_dino_score failed"

# Filter final dataset
python src/generate_composite_dataset.py --task filter_prompts || \
    exit_on_error "filter_prompts failed"

# ==================== COMPLETION ====================
log "=== Composite Pipeline completed successfully ==="