import logging
from pathlib import Path
from anonator.core.processor import VideoProcessor, ProgressData

logging.basicConfig(level=logging.INFO)

def progress_callback(data: ProgressData):
    progress_pct = (data.frame_number / data.total_frames * 100) if data.total_frames > 0 else 0
    print(f"HIPAA Mode Progress: {progress_pct:.1f}% - Frame {data.frame_number}/{data.total_frames} - {data.fps:.1f} fps")

input_video = Path("testData/WhatsApp Video 2025-10-21 at 19.27.27_09f5a5f4.mp4")
output_video = input_video.parent / f"{input_video.stem}_HIPAA_anonymized{input_video.suffix}"

print("=" * 70)
print("HIPAA MODE TEST - Medical-Grade Video Anonymization")
print("=" * 70)
print(f"Input:  {input_video}")
print(f"Output: {output_video}")
print()
print("HIPAA Mode Settings:")
print("  Threshold:       0.01 (ultra-sensitive)")
print("  Mask Scale:      2.0 (maximum coverage)")
print("  Mode:            solid (complete blackout)")
print("  Multi-Pass:      Yes (3 detection passes)")
print("  Keep Audio:      No (removes voice PHI)")
print()
print("Starting HIPAA-compliant processing...")
print("=" * 70)
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

print("Processing started in background thread...")
print("This will take ~3x longer due to multi-pass detection")
print()

import time
start_time = time.time()
while processor._processing_thread and processor._processing_thread.is_alive():
    time.sleep(1)

elapsed = time.time() - start_time
print()
print("=" * 70)
print(f"HIPAA Mode Processing Complete!")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Output saved to: {output_video}")
print("=" * 70)
