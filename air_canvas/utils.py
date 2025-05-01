# utils.py
import cv2
import streamlit as st

@st.cache_resource
def initialize_camera():
    for i in range(0, 3):  # Try 3 camera indices (0, 1, 2)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
    return None

def calculate_fps(prev_time, frame_count):
    curr_time = time.time()
    if curr_time - prev_time >= 1:
        fps = frame_count
        frame_count = 0
        prev_time = curr_time
        return fps, prev_time, frame_count
    return None, prev_time, frame_count