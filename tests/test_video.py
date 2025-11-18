import logging
from pathlib import Path
from anonator.core.processor import VideoProcessor, ProgressData

logging.basicConfig(level=logging.INFO)

def progress_callback(data: ProgressData):
    progress_pct = (data.frame_number / data.total_frames * 100) if data.total_frames > 0 else 0
    print(f"Progress: {progress_pct:.1f}% - Frame {data.frame_number}/{data.total_frames} - {data.fps:.1f} fps")

input_video = Path("testData/WhatsApp Video 2025-10-21 at 19.27.27_09f5a5f4.mp4")
output_video = input_video.parent / f"{input_video.stem}_anonymized{input_video.suffix}"

print(f"Input: {input_video}")
print(f"Output: {output_video}")
print("Starting processing...")

processor = VideoProcessor(progress_callback=progress_callback)
processor.process_video(
    input_path=str(input_video),
    output_path=str(output_video),
    anonymization_mode="blur",
    threshold=0.2,
    keep_audio=True
)

print("Processing started in background thread")
print("Waiting for completion...")

import time
while processor._processing_thread and processor._processing_thread.is_alive():
    time.sleep(1)

print("Done!")
