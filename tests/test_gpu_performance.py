import logging
from pathlib import Path
from anonator.core.processor import VideoProcessor, ProgressData

logging.basicConfig(level=logging.INFO, format='%(message)s')

def progress_callback(data: ProgressData):
    progress_pct = (data.frame_number / data.total_frames * 100) if data.total_frames > 0 else 0
    print(f"[{progress_pct:5.1f}%] Frame {data.frame_number:3d}/{data.total_frames} | {data.fps:.1f} fps | Elapsed: {data.elapsed_time:.1f}s")

input_video = Path("testData/WhatsApp Video 2025-10-21 at 19.27.27_09f5a5f4.mp4")
output_video = input_video.parent / f"{input_video.stem}_GPU_test{input_video.suffix}"

print("=" * 80)
print("GPU ACCELERATION TEST - HIPAA MODE")
print("=" * 80)
print(f"Input:  {input_video}")
print(f"Output: {output_video}")
print()
print("Settings:")
print("  Mode:        solid")
print("  Threshold:   0.05 (optimized)")
print("  Mask Scale:  2.0")
print("  Multi-Pass:  YES (3 passes)")
print("  Keep Audio:  NO")
print()
print("Expected: GPU should provide 10-15x speedup vs CPU")
print("Previous CPU: 4.3 fps (60 seconds)")
print("Expected GPU: 40-60+ fps (5-10 seconds)")
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
print(f"Average FPS: {256 / elapsed:.1f}")
print(f"Output: {output_video}")
print()
print("Performance comparison:")
print(f"  CPU (threshold 0.05):  4.3 fps, 60 seconds")
print(f"  GPU (threshold 0.05):  {256 / elapsed:.1f} fps, {elapsed:.1f} seconds")
print(f"  Speedup: {(60 / elapsed):.1f}x faster!")
print("=" * 80)
