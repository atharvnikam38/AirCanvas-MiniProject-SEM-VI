# canvas_manager.py
import cv2
import numpy as np
import time
from datetime import datetime
import os

class CanvasManager:
    def __init__(self, frame_shape):
        self.canvas = np.zeros_like(frame_shape)
        self.permanent_drawing = np.zeros_like(frame_shape)
        self.history = []
        self.drawing_path = []
        self.disappearing_elements = []
        self.prev_pos = None
        self.is_drawing = False
        self.current_path = None
        self.current_path_start = None

    def save_canvas(self, filename=None):
        if filename is None:
            filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cv2.imwrite(f"{filename}.png", self.canvas)
        return f"Saved as {filename}.png"

    def load_canvas(self):
        try:
            files = [f for f in os.listdir() if f.endswith('.png')]
            if files:
                latest = max(files, key=os.path.getctime)
                self.canvas = cv2.imread(latest)
                return f"Loaded {latest}"
            return "No files found"
        except Exception as e:
            return f"Error loading: {str(e)}"

    def update_history(self):
        if len(self.history) > 10:
            self.history.pop(0)
        self.history.append(self.canvas.copy())

    def undo(self):
        if self.history:
            self.canvas = self.history.pop()
            return True
        return False

    def clear(self):
        self.update_history()
        self.canvas = np.zeros_like(self.canvas)
        self.permanent_drawing = np.zeros_like(self.canvas)
        self.disappearing_elements = []
        self.drawing_path = []
        return True