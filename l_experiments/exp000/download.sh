#!/bin/bash
# exp000: prepare manually downloaded Solafune data archives.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR" "$DATA_DIR/train_dataset" "$DATA_DIR/evaluation_dataset" "$DATA_DIR/sample_submission"

echo "========================================"
echo "exp000: Data Preparation"
echo "Raw dir: $RAW_DIR"
echo "========================================"

prepare_archive() {
  local archive="$1"
  local target="$2"

  if [ ! -f "$archive" ]; then
    echo "Missing: $archive"
    return
  fi

  echo "Extracting $(basename "$archive") -> $target"
  unzip -q -o "$archive" -d "$target"
}

prepare_archive "$RAW_DIR/train_dataset.zip" "$DATA_DIR/train_dataset"
prepare_archive "$RAW_DIR/evaluation_dataset.zip" "$DATA_DIR/evaluation_dataset"
prepare_archive "$RAW_DIR/sample_submission.zip" "$DATA_DIR/sample_submission"

echo ""
echo "Directory summary:"
find "$DATA_DIR" -maxdepth 2 -type f | sort | head -50

echo ""
echo "If files are missing, place the Solafune zip archives in:"
echo "$RAW_DIR"
