#!/usr/bin/env python
"""
Test Data Preparation Script

This script extracts sample frames from test videos and creates synthetic test images
for comprehensive testing of the face detection and anonymization system.

Usage:
    python tests/prepare_test_data.py
"""

import cv2
import json
import numpy as np
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, num_frames=10, prefix="frame"):
    """
    Extract sample frames from a video file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        prefix: Prefix for output filenames

    Returns:
        List of extracted frame paths
    """
    print(f"Extracting {num_frames} frames from {video_path}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)

    extracted_paths = []
    frame_idx = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            output_path = output_dir / f"{prefix}_{extracted_count:03d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted_paths.append(output_path)
            extracted_count += 1
            print(f"  Extracted frame {extracted_count}/{num_frames}")

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(extracted_paths)} frames")

    return extracted_paths


def create_synthetic_images(output_dir, num_images=5):
    """
    Create synthetic test images with various characteristics.

    Args:
        output_dir: Directory to save synthetic images
        num_images: Number of images to create

    Returns:
        Dictionary mapping image paths to metadata
    """
    print(f"Creating {num_images} synthetic test images...")

    synthetic_data = {}

    # 1. Empty frame (no faces)
    img_path = output_dir / "synthetic_empty.jpg"
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(empty_img, (0, 300), (640, 480), (100, 200, 100), -1)  # Ground
    cv2.rectangle(empty_img, (0, 0), (640, 300), (150, 200, 255), -1)    # Sky
    cv2.imwrite(str(img_path), empty_img)
    synthetic_data[str(img_path)] = {"faces": [], "description": "Empty landscape"}
    print(f"  Created: {img_path.name}")

    # 2. Single face (centered)
    img_path = output_dir / "synthetic_single_face.jpg"
    single_face_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    x, y, w, h = 270, 190, 100, 120
    cv2.rectangle(single_face_img, (x, y), (x+w, y+h), (255, 200, 150), -1)
    # Add eyes
    cv2.circle(single_face_img, (x + w//3, y + h//3), 5, (0, 0, 0), -1)
    cv2.circle(single_face_img, (x + 2*w//3, y + h//3), 5, (0, 0, 0), -1)
    cv2.imwrite(str(img_path), single_face_img)
    synthetic_data[str(img_path)] = {
        "faces": [{"bbox": [x, y, x+w, y+h], "difficulty": "easy"}],
        "description": "Single centered face"
    }
    print(f"  Created: {img_path.name}")

    # 3. Multiple faces
    img_path = output_dir / "synthetic_multiple_faces.jpg"
    multi_face_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    face_positions = [
        (50, 50, 80, 100),
        (200, 100, 80, 100),
        (400, 50, 80, 100),
        (150, 250, 80, 100),
        (450, 300, 80, 100),
    ]
    faces = []
    for x, y, w, h in face_positions:
        cv2.rectangle(multi_face_img, (x, y), (x+w, y+h), (255, 200, 150), -1)
        cv2.circle(multi_face_img, (x + w//3, y + h//3), 3, (0, 0, 0), -1)
        cv2.circle(multi_face_img, (x + 2*w//3, y + h//3), 3, (0, 0, 0), -1)
        faces.append({"bbox": [x, y, x+w, y+h], "difficulty": "easy"})
    cv2.imwrite(str(img_path), multi_face_img)
    synthetic_data[str(img_path)] = {
        "faces": faces,
        "description": "Multiple faces (5)"
    }
    print(f"  Created: {img_path.name}")

    # 4. Edge case: Face at boundary
    img_path = output_dir / "synthetic_edge_face.jpg"
    edge_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    # Face partially at top-left corner
    x, y, w, h = 0, 0, 80, 100
    cv2.rectangle(edge_img, (x, y), (x+w, y+h), (255, 200, 150), -1)
    cv2.imwrite(str(img_path), edge_img)
    synthetic_data[str(img_path)] = {
        "faces": [{"bbox": [x, y, x+w, y+h], "difficulty": "hard_boundary"}],
        "description": "Face at image boundary"
    }
    print(f"  Created: {img_path.name}")

    # 5. Small faces
    img_path = output_dir / "synthetic_small_faces.jpg"
    small_face_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    small_faces = []
    for i in range(10):
        x = np.random.randint(0, 620)
        y = np.random.randint(0, 460)
        w, h = 20, 25  # Very small
        cv2.rectangle(small_face_img, (x, y), (x+w, y+h), (255, 200, 150), -1)
        small_faces.append({"bbox": [x, y, x+w, y+h], "difficulty": "hard_small"})
    cv2.imwrite(str(img_path), small_face_img)
    synthetic_data[str(img_path)] = {
        "faces": small_faces,
        "description": "Many small faces"
    }
    print(f"  Created: {img_path.name}")

    print(f"Created {len(synthetic_data)} synthetic images")
    return synthetic_data


def create_ground_truth_file(output_path, data):
    """
    Create ground truth JSON file with face annotations.

    Args:
        output_path: Path to output JSON file
        data: Dictionary mapping image paths to metadata
    """
    print(f"Creating ground truth file: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Ground truth file created with {len(data)} entries")


def main():
    """Main function to prepare all test data."""
    print("=" * 60)
    print("Test Data Preparation Script")
    print("=" * 60)

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    sample_frames_dir = data_dir / "sample_frames"
    synthetic_dir = data_dir / "synthetic"
    testdata_dir = script_dir.parent / "testData"

    # Create directories
    sample_frames_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Sample frames: {sample_frames_dir}")
    print(f"Synthetic images: {synthetic_dir}")
    print(f"Source videos: {testdata_dir}")
    print()

    # Extract frames from test videos
    ground_truth_data = {}

    if testdata_dir.exists():
        video_files = list(testdata_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} test video(s)")

        for video_path in video_files:
            video_name = video_path.stem
            prefix = f"{video_name}_frame"

            extracted = extract_frames_from_video(
                video_path,
                sample_frames_dir,
                num_frames=10,
                prefix=prefix
            )

            # Add to ground truth (no annotations for real video frames)
            for frame_path in extracted:
                ground_truth_data[str(frame_path)] = {
                    "source": str(video_path),
                    "description": f"Real frame from {video_name}"
                }

        print()
    else:
        print(f"Warning: Test video directory not found: {testdata_dir}")
        print("Skipping video frame extraction")
        print()

    # Create synthetic images
    synthetic_data = create_synthetic_images(synthetic_dir, num_images=5)
    ground_truth_data.update(synthetic_data)

    print()

    # Create ground truth JSON
    ground_truth_path = data_dir / "ground_truth.json"
    create_ground_truth_file(ground_truth_path, ground_truth_data)

    print()
    print("=" * 60)
    print("Test Data Preparation Complete!")
    print("=" * 60)
    print(f"Sample frames: {len(list(sample_frames_dir.glob('*.jpg')))}")
    print(f"Synthetic images: {len(list(synthetic_dir.glob('*.jpg')))}")
    print(f"Ground truth entries: {len(ground_truth_data)}")
    print()
    print("You can now run the test suite:")
    print("  pytest tests/")
    print()


if __name__ == "__main__":
    main()
