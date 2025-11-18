import logging
from pathlib import Path
from anonator.core.processor import VideoProcessor, ProgressData

logging.basicConfig(level=logging.INFO, format='%(message)s')

def progress_callback(data: ProgressData):
    progress_pct = (data.frame_number / data.total_frames * 100) if data.total_frames > 0 else 0
    print(f"[{progress_pct:5.1f}%] Frame {data.frame_number:3d}/{data.total_frames} | {data.fps:.1f} fps | Elapsed: {data.elapsed_time:.1f}s")

input_video = Path("testData/WhatsApp Video 2025-10-21 at 19.27.27_09f5a5f4.mp4")
output_video = input_video.parent / f"{input_video.stem}_threshold_0.05{input_video.suffix}"

print("=" * 80)
print("TESTING WITH THRESHOLD=0.05 (More Reasonable for HIPAA)")
print("=" * 80)
print(f"Input:  {input_video}")
print(f"Output: {output_video}")
print()
print("Settings:")
print("  Mode:        solid")
print("  Threshold:   0.05 (still very sensitive)")
print("  Mask Scale:  2.0")
print("  Multi-Pass:  YES")
print("  Keep Audio:  NO")
print()
print("Expected: Much faster processing with fewer false positives")
print("=" * 80)
print()

processor = VideoProcessor(progress_callback=progress_callback)
processor.process_video(
    input_path=str(input_video),
    output_path=str(output_video),
    anonymization_mode="solid",
    threshold=0.05,
    mask_scale=2.0,
    multi_pass=True,
    keep_audio=False
)

print("\nProcessing in progress...")

import time
start = time.time()
while processor._processing_thread and processor._processing_thread.is_alive():
    time.sleep(0.5)

elapsed = time.time() - start
print()
print("=" * 80)
print(f"COMPLETE! Total time: {elapsed:.1f} seconds")
print(f"Output: {output_video}")
print("=" * 80)
