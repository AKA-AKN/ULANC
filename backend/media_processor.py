# backend/media_processor.py
import cv2
import mediapipe as mp
import numpy as np
import logging # <-- ADD THIS LINE

# --- Processor Interface ---
class BaseProcessor:
    def process_frame(self, frame):
        """Processes a single video frame and returns the processed frame."""
        raise NotImplementedError

# --- Initial Implementation (Placeholder for ULANC) ---
class StandardProcessor(BaseProcessor):
    def __init__(self):
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        logging.info("StandardProcessor initialized with blur effect.")

    def process_frame(self, img):
        # The blur logic we've already built
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb_img)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        blurred_background = cv2.GaussianBlur(img, (99, 99), 0)
        output_img = np.where(condition, img, blurred_background)
        return output_img