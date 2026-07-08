#!/bin/bash
# exp000: prepare manually downloaded Solafune data archives.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
RAW_DIR="$DATA_DIR/raw"
DRY_RUN="${DRY_RUN:-0}"
FORCE_UNZIP="${FORCE_UNZIP:-0}"

mkdir -p "$RAW_DIR" "$DATA_DIR/train_dataset" "$DATA_DIR/evaluation_dataset" "$DATA_DIR/sample_submission"

echo "========================================"
echo "exp000: Data Preparation"
echo "Project dir: $PROJECT_DIR"
echo "Data dir:    $DATA_DIR"
echo "Raw dir:     $RAW_DIR"
echo "========================================"

find_archive() {
  local standard_name="$1"
  local long_name="$2"
  local standard_path="$RAW_DIR/$standard_name"

  if [ -f "$standard_path" ]; then
    echo "$standard_path"
    return 0
  fi
  if [ -f "$DATA_DIR/$standard_name" ]; then
    echo "$DATA_DIR/$standard_name"
    return 0
  fi
  if [ -f "$DATA_DIR/$long_name" ]; then
    ln -sfn "../$long_name" "$standard_path"
    echo "$standard_path"
    return 0
  fi
  return 1
}

extract_archive() {
  local label="$1"
  local archive="$2"
  local target="$3"
  local marker="$4"

  if [ "$FORCE_UNZIP" != "1" ] && [ -f "$marker" ]; then
    echo "Skip $label: already extracted ($marker)"
    return 0
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "Would extract $label: $archive -> $target"
    return 0
  fi
  echo "Extracting $label: $archive -> $target"
  unzip -q -o "$archive" -d "$target"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: required file missing: $path"
    return 1
  fi
}

TRAIN_ARCHIVE="$(find_archive "train_dataset.zip" "train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip" || true)"
EVAL_ARCHIVE="$(find_archive "evaluation_dataset.zip" "evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip" || true)"
SAMPLE_ARCHIVE="$(find_archive "sample_submission.zip" "sample_submission_95c3b1e094034f5fbba421f5e5310f8a.zip" || true)"

if [ -z "$TRAIN_ARCHIVE" ] || [ -z "$EVAL_ARCHIVE" ] || [ -z "$SAMPLE_ARCHIVE" ]; then
  echo "ERROR: one or more Solafune archives are missing."
  echo "Place these files in either $RAW_DIR or $DATA_DIR:"
  echo "  train_dataset.zip or train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip"
  echo "  evaluation_dataset.zip or evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip"
  echo "  sample_submission.zip or sample_submission_95c3b1e094034f5fbba421f5e5310f8a.zip"
  exit 1
fi

extract_archive "train_dataset" "$TRAIN_ARCHIVE" "$DATA_DIR/train_dataset" "$DATA_DIR/train_dataset/train_dataset.csv"
extract_archive "evaluation_dataset" "$EVAL_ARCHIVE" "$DATA_DIR/evaluation_dataset" "$DATA_DIR/evaluation_dataset/evaluation_target.csv"
extract_archive "sample_submission" "$SAMPLE_ARCHIVE" "$DATA_DIR/sample_submission" "$DATA_DIR/sample_submission/evaluation_target.csv"

if [ "$DRY_RUN" != "1" ]; then
  require_file "$DATA_DIR/train_dataset/train_dataset.csv"
  require_file "$DATA_DIR/evaluation_dataset/evaluation_target.csv"
  require_file "$DATA_DIR/sample_submission/evaluation_target.csv"
fi

echo ""
echo "Directory summary:"
find "$DATA_DIR" -maxdepth 2 -type f | sort | head -50
