# config.py
COLORS = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
}

GEMINI_CONFIG = {
    "api_key": "AIzaSyAuW8PdguB7eF1vwD2cM8mvWtO1EKmicrk",
    "model_name": "gemini-2.0-flash"
}

CAMERA_CONFIG = {
    "width": 1280,
    "height": 720,
    "attempts": 3
}

HAND_DETECTOR_CONFIG = {
    "mode": False,          # Boolean
    "maxHands": 1,          # Integer
    "detectionCon": 0.8,    # Float
    "minTrackCon": 0.6      # Float
}