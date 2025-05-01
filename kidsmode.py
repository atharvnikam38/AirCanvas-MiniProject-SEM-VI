import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import time
import os
from datetime import datetime
import math
import random

def main():
    st.set_page_config(layout="wide")
    st.title("🌟 Kids Learning Air Canvas with Gemini AI ✨")

    # Create columns for layout (wider left column for camera)
    col1, col2 = st.columns([4, 2])

    # Initialize Gemini AI
    try:
        genai.configure(api_key="AIzaSyAuW8PdguB7eF1vwD2cM8mvWtO1EKmicrk")
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {str(e)}")
        st.stop()

    # Webcam Initialization
    @st.cache_resource
    def initialize_camera():
        for i in range(0, 3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
        return None

    cap = initialize_camera()
    if not cap:
        st.error("Could not initialize webcam. Please check your camera connection.")
        st.stop()

    # Hand Detector Setup
    detector = HandDetector(
        mode=False,
        maxHands=1,
        detectionCon=0.8,
        minTrackCon=0.6
    )

    # Define colors dictionary
    colors = {
        "Red": (0, 0, 255),
        "Blue": (255, 0, 0),
        "Green": (0, 255, 0),
        "Yellow": (0, 255, 255),
        "Purple": (255, 0, 255),
        "Pink": (147, 20, 255)
    }

    # Function Definitions (same as before)
    def get_hand_info(img):
        hands = detector.findHands(img, draw=False, flipType=True)
        if hands and len(hands) > 0:
            hand = hands[0]
            return {
                'fingers': detector.fingersUp(hand),
                'landmarks': hand["lmList"],
                'type': hand.get("type", "Unknown")
            }
        return None

    def draw_tracing_template(canvas, char):
        height, width = canvas.shape[:2]
        canvas = np.zeros_like(canvas)
        x, y = width // 2, height // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10 if mode == "Letter Tracing" else 15
        thickness = 8
        text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(canvas, char, (text_x, text_y), font, font_scale, 
                   (100, 100, 100), thickness + 5)
        cv2.putText(canvas, char, (text_x, text_y), font, font_scale, 
                   (200, 200, 200), thickness)
        return canvas

    def process_drawing(info, img, display_frame):
        if info is None:
            st.session_state.is_drawing = False
            st.session_state.prev_pos = None
            return
        
        fingers = info['fingers']
        landmarks = info['landmarks']
        
        if fingers == [0, 1, 0, 0, 0] and len(landmarks) > 8:
            current_pos = landmarks[8][0:2]
            
            if st.session_state.is_drawing and st.session_state.prev_pos is not None:
                cv2.line(
                    st.session_state.canvas, 
                    st.session_state.prev_pos, 
                    current_pos, 
                    st.session_state.brush_color,
                    brush_size
                )
                
                if mode in ["Letter Tracing", "Number Tracing"] and test_mode and 'test_started' in st.session_state:
                    if st.session_state.test_canvas is None:
                        st.session_state.test_canvas = np.zeros_like(img)
                    cv2.line(
                        st.session_state.test_canvas,
                        st.session_state.prev_pos,
                        current_pos,
                        (255, 255, 255),
                        brush_size + 5
                    )
                
                if mode == "Math Quiz" and 'math_question' in st.session_state:
                    if st.session_state.math_canvas is None:
                        st.session_state.math_canvas = np.zeros_like(img)
                    cv2.line(
                        st.session_state.math_canvas,
                        st.session_state.prev_pos,
                        current_pos,
                        (255, 255, 255),
                        brush_size + 5
                    )
            
            st.session_state.prev_pos = current_pos
            st.session_state.is_drawing = True
        
        elif fingers == [1, 0, 0, 0, 0]:
            st.session_state.canvas = np.zeros_like(img)
            st.session_state.test_canvas = None
            st.session_state.math_canvas = None
            if mode == "Math Quiz" and 'math_feedback' in st.session_state:
                st.session_state.math_feedback = ""
        
        else:
            st.session_state.is_drawing = False
            st.session_state.prev_pos = None

    # Left Column - Just the camera view
    with col1:
        run = st.checkbox('Enable Camera', value=True)
        FRAME_WINDOW = st.image([], channels="BGR")

    # Right Column - All controls and information
    with col2:
        st.header("🎨 Drawing Controls")
        
        # Mode selection at the top
        mode = st.radio("Select Mode:", 
                    ["Free Drawing", "Letter Tracing", "Number Tracing", "Math Quiz"],
                    index=0)
        
        # Color selection
        st.markdown("**🌈 Brush Color:**")
        color_cols = st.columns(6)
        for i, (color_name, color_val) in enumerate(colors.items()):
            with color_cols[i]:
                if st.button(color_name, key=f"color_{color_name}"):
                    st.session_state.brush_color = color_val
        
        brush_size = st.slider("🖌️ Brush Size", 5, 30, 15)
        
        # Mode-specific controls
        if mode in ["Letter Tracing", "Number Tracing"]:
            st.markdown("### 🔤 Tracing Practice")
            if mode == "Letter Tracing":
                selected_char = st.selectbox("Select Letter", 
                                          [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)])
            else:
                selected_char = st.selectbox("Select Number", [str(i) for i in range(10)])
            
            test_mode = st.checkbox("Test Mode (AI will check your work)", False)
            if test_mode:
                if st.button("Give me a random letter/number to draw"):
                    if mode == "Letter Tracing":
                        st.session_state.test_char = random.choice([chr(i) for i in range(65, 91)])
                    else:
                        st.session_state.test_char = random.choice([str(i) for i in range(10)])
                    st.session_state.test_started = True
                    st.session_state.test_canvas = None
                
                if 'test_char' in st.session_state:
                    st.markdown(f"### ✏️ Draw: {st.session_state.test_char}")
                    if st.button("Check My Drawing"):
                        if st.session_state.test_canvas is not None:
                            pil_img = Image.fromarray(cv2.cvtColor(st.session_state.test_canvas, cv2.COLOR_BGR2RGB))
                            try:
                                response = model.generate_content([
                                    f"Is this a correctly drawn {st.session_state.test_char}? Answer only 'YES' or 'NO'.",
                                    pil_img
                                ])
                                result = response.text.strip().upper()
                                if "YES" in result:
                                    st.success("✅ Great job! You drew it perfectly!")
                                    st.balloons()
                                else:
                                    st.warning("❌ Almost there! Try again!")
                            except:
                                st.error("Couldn't check your drawing. Try again!")
        
        elif mode == "Math Quiz":
            st.markdown("### ➕ Math Quiz")
            difficulty = st.selectbox("Difficulty", 
                                   ["Easy (1-10)", "Medium (1-50)", "Hard (1-100)"])
            if st.button("Generate New Question"):
                if difficulty == "Easy (1-10)":
                    num1 = random.randint(1, 10)
                    num2 = random.randint(1, 10)
                elif difficulty == "Medium (1-50)":
                    num1 = random.randint(1, 50)
                    num2 = random.randint(1, 50)
                else:
                    num1 = random.randint(1, 100)
                    num2 = random.randint(1, 100)
                
                st.session_state.math_question = f"{num1} + {num2}"
                st.session_state.math_answer = num1 + num2
                st.session_state.math_canvas = None
                st.session_state.math_feedback = ""
            
            if 'math_question' in st.session_state:
                st.markdown(f"**Question:** {st.session_state.math_question}")
                if st.button("Check Answer"):
                    if st.session_state.math_canvas is not None:
                        pil_img = Image.fromarray(cv2.cvtColor(st.session_state.math_canvas, cv2.COLOR_BGR2RGB))
                        try:
                            response = model.generate_content([
                                "What number is written in this image? Return only the number with no additional text.",
                                pil_img
                            ])
                            user_answer = response.text.strip()
                            try:
                                user_num = int(user_answer)
                                if user_num == st.session_state.math_answer:
                                    st.session_state.math_feedback = f"✅ Correct! {st.session_state.math_question} = {user_num}"
                                    st.balloons()
                                else:
                                    st.session_state.math_feedback = f"❌ Oops! The correct answer is {st.session_state.math_answer}"
                            except:
                                st.session_state.math_feedback = "❌ I couldn't read your number. Try writing more clearly!"
                        except:
                            st.session_state.math_feedback = "❌ Error checking answer. Try again!"
                
                if 'math_feedback' in st.session_state and st.session_state.math_feedback:
                    if "✅" in st.session_state.math_feedback:
                        st.success(st.session_state.math_feedback)
                    else:
                        st.warning(st.session_state.math_feedback)
        
        # Hand Gestures section at the bottom
        st.markdown("---")
        st.markdown("### ✋ Hand Gestures Guide")
        gesture_cols = st.columns(2)
        with gesture_cols[0]:
            st.markdown("**👆 Index finger up**  \nDraw mode")
            st.markdown("**👍 Thumb up**  \nClear canvas")
        with gesture_cols[1]:
            st.markdown("**✌️ Two fingers up**  \nComplete shape")
            st.markdown("**🤟 All fingers up**  \nAsk for help")

        if st.button("🔄 Reset Application", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Initialize session state variables
    if 'canvas' not in st.session_state:
        st.session_state.canvas = None
    if 'prev_pos' not in st.session_state:
        st.session_state.prev_pos = None
    if 'is_drawing' not in st.session_state:
        st.session_state.is_drawing = False
    if 'brush_color' not in st.session_state:
        st.session_state.brush_color = colors["Red"]
    if 'test_canvas' not in st.session_state:
        st.session_state.test_canvas = None
    if 'math_canvas' not in st.session_state:
        st.session_state.math_canvas = None

    # Main application loop
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera error, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = initialize_camera()
            if not cap:
                st.error("Camera failed")
                run = False
                break
            continue
        
        frame = cv2.flip(frame, 1)
        
        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(frame)
        
        display_frame = frame.copy()
        
        if mode in ["Letter Tracing", "Number Tracing"] and not test_mode:
            template_frame = draw_tracing_template(np.zeros_like(frame), selected_char)
            display_frame = cv2.addWeighted(display_frame, 0.7, template_frame, 0.3, 0)
        elif mode in ["Letter Tracing", "Number Tracing"] and test_mode and 'test_char' in st.session_state:
            cv2.putText(display_frame, f"Draw: {st.session_state.test_char}", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        if mode == "Math Quiz" and 'math_question' in st.session_state:
            cv2.putText(display_frame, f"{st.session_state.math_question} = ?", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        display_frame = cv2.addWeighted(display_frame, 0.7, st.session_state.canvas, 0.3, 0)
        
        hand_info = get_hand_info(frame)
        process_drawing(hand_info, frame, display_frame)
        
        frame_count += 1
        curr_time = time.time()
        if curr_time - prev_time >= 1:
            fps = frame_count
            frame_count = 0
            prev_time = curr_time
        
        FRAME_WINDOW.image(display_frame, channels="BGR")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()