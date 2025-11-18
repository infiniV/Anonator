import numpy as np
import cv2
from typing import Tuple
from anonator.core.config import PROCESSOR_CONFIG


def scale_bb(x1: int, y1: int, x2: int, y2: int, mask_scale: float = 1.0) -> Tuple[int, int, int, int]:
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 = int(y1 - h * s)
    y2 = int(y2 + h * s)
    x1 = int(x1 - w * s)
    x2 = int(x2 + w * s)
    return x1, y1, x2, y2


def anonymize_frame(
    frame: np.ndarray,
    dets: np.ndarray,
    mask_scale: float = 1.3,
    replacewith: str = "blur",
    ellipse: bool = True,
    draw_scores: bool = False,
    replaceimg: np.ndarray = None,
    mosaicsize: int = 20
) -> np.ndarray:
    if dets is None or len(dets) == 0:
        return frame

    frame = frame.copy()

    for detection in dets:
        x1, y1, x2, y2, score = detection[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        if replacewith == "blur":
            roi = cv2.GaussianBlur(roi, (PROCESSOR_CONFIG.blur_kernel_size, PROCESSOR_CONFIG.blur_kernel_size), PROCESSOR_CONFIG.blur_sigma)
            if ellipse:
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.ellipse(mask, (roi.shape[1]//2, roi.shape[0]//2),
                           (roi.shape[1]//2, roi.shape[0]//2), 0, 0, 360, 255, -1)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                frame[y1:y2, x1:x2] = (roi * mask_3ch + frame[y1:y2, x1:x2] * (1 - mask_3ch)).astype(np.uint8)
            else:
                frame[y1:y2, x1:x2] = roi

        elif replacewith == "solid":
            frame[y1:y2, x1:x2] = 0

        elif replacewith == "mosaic":
            small = cv2.resize(roi, (mosaicsize, mosaicsize), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

        elif replacewith == "img" and replaceimg is not None:
            replaceimg_resized = cv2.resize(replaceimg, (x2-x1, y2-y1))
            frame[y1:y2, x1:x2] = replaceimg_resized

        if draw_scores:
            cv2.putText(frame, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
