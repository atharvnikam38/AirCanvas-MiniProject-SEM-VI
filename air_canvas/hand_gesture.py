# hand_gesture.py
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

class HandGestureDetector:
    def __init__(self):
        self.detector = HandDetector(
            mode=["mode"],
            maxHands=["maxHands"],
            detectionCon=["detectionCon"],
            minTrackCon=["minTrackCon"]
        )

    def get_hand_info(self, img):
        hands = self.detector.findHands(img, draw=False, flipType=True)
        if hands and len(hands) > 0:
            hand = hands[0]
            return {
                'fingers': self.detector.fingersUp(hand),
                'landmarks': hand["lmList"],
                'type': hand.get("type", "Unknown")
            }
        return None

    @staticmethod
    def draw_eraser_preview(frame, position, size):
        overlay = frame.copy()
        cv2.circle(overlay, position, size // 2, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, position, size // 2, (0, 0, 255), 2)
        return frame