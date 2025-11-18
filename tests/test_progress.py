import logging
from pathlib import Path
from anonator.core.processor import VideoProcessor, ProgressData

logging.basicConfig(level=logging.INFO, format='%(message)s')

def progress_callback(data: ProgressData):
    progress_pct = (data.frame_number / data.total_frames * 100) if data.total_frames > 0 else 0
    print(f"[{progress_pct:5.1f}%] Frame {data.frame_number:3d}/{data.total_frames} | {data.fps:.1f} fps | Elapsed: {data.elapsed_time:.1f}s")

input_video = Path("testData/WhatsApp Video 2025-10-21 at 19.27.27_09f5a5f4.mp4")
output_video = input_video.parent / f"{input_video.stem}_test_progress{input_video.suffix}"

print("=" * 80)
print("TESTING PROGRESS DISPLAY WITH HIPAA MODE")
print("=" * 80)
print(f"Input:  {input_video}")
print(f"Output: {output_video}")
print()
print("Settings:")
print("  Mode:        solid")
print("  Threshold:   0.01 (ultra-sensitive)")
print("  Mask Scale:  2.0 (maximum coverage)")
print("  Multi-Pass:  YES (3 detection passes per frame)")
print("  Keep Audio:  NO")
print()
print("Expected processing time: ~45-60 seconds (3x slower due to multi-pass)")
print("=" * 80)
print()

processor = VideoProcessor(progress_callback=progress_callback)
processor.process_video(
    input_path=str(input_video),
    output_path=str(output_video),
    anonymization_mode="solid",
    threshold=0.01,
    mask_scale=2.0,
    multi_pass=True,
    keep_audio=False
)

print("\nProcessing in progress...")
print("Watch for progress updates above...")
print()

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
