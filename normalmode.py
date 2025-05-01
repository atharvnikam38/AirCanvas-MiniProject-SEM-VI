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

        # Streamlit UI Setup
def main():
        st.set_page_config(layout="wide")
        st.title("🌟 Enhanced Air Canvas with Gemini AI OCR & Visual Effects ✨")

        # Create columns for layout
        col1, col2 = st.columns([3, 2])

        with col1:
            run = st.checkbox('Enable Camera', value=True)
            FRAME_WINDOW = st.image([], channels="BGR")
            
            # Drawing Tools
            st.markdown("### 🎨 Brush Settings")
            
            # Color selection buttons with better visual feedback
            st.markdown("**Select Color:**")
            color_cols = st.columns(4)
            with color_cols[0]: 
                red = st.button("🔴 Red", use_container_width=True, help="Red color")
            with color_cols[1]: 
                blue = st.button("🔵 Blue", use_container_width=True, help="Blue color")
            with color_cols[2]: 
                green = st.button("🟢 Green", use_container_width=True, help="Green color")
        
            brush_size = st.slider("🖌️ Brush Size", 5, 30, 10)
            eraser_size = st.slider("🧽 Eraser Size", 10, 50, 20)
                
            # Visual Effects
            st.markdown("### ✨ Visual Effects")
            effect = st.selectbox("Choose Effect", ["None", "Disappearing Text", "Shape Assist"])
            
            # Templates
            st.markdown("### 📝 Templates")
            template = st.selectbox("Choose Template", ["None", "Graph Paper", "Music Staff", "Text Lines"])
            
            # Save/Load
            st.markdown("### 💾 Save/Load")
            save_name = st.text_input("Filename:")
            save_cols = st.columns(2)
            with save_cols[0]: save_btn = st.button("💾 Save Drawing", use_container_width=True)
            with save_cols[1]: load_btn = st.button("📂 Load Last", use_container_width=True)

        with col2:
            st.header("🔍 Gemini AI OCR & Solutions")
            output_text_area = st.empty()
            
            # OCR Settings (using Gemini AI)
            st.markdown("### 🔤 OCR with Gemini AI")
            ocr_threshold = st.slider("Image Contrast Threshold", 0, 255, 180, 
                                    help="Adjust contrast for better text recognition")
            ocr_cols = st.columns(2)
            with ocr_cols[0]: ocr_btn = st.button("🔍 Run OCR", use_container_width=True)
            with ocr_cols[1]: live_ocr = st.checkbox("Live OCR", value=False, 
                                                help="Periodically scan canvas for text")
            
            # Add option to ask Gemini about detected text
            if "last_recognized_text" not in st.session_state:
                st.session_state.last_recognized_text = ""
            
            analyze_text = st.button("🧠 Analyze Last Recognized Text", 
                                    help="Ask Gemini AI to analyze/explain the recognized text")
            
            st.markdown("### 🛠️ Canvas Controls")
            clear_cols = st.columns(2)
            with clear_cols[0]: clear_button = st.button("🧹 Clear Canvas", use_container_width=True)
            with clear_cols[1]: undo_btn = st.button("↩️ Undo", use_container_width=True)
            
            # Visual Feedback Panel
            st.markdown("### 📊 Status")
            status = st.empty()
            fps_display = st.empty()

           
            if st.button("👶 Switch to Kids Mode", use_container_width=True):
                st.session_state.current_mode = "kids"
                st.rerun()

        # Color presets (BGR format)
        COLORS = {
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
        }

        # Initialize selected color and effects
        if 'brush_color' not in st.session_state:
            st.session_state.brush_color = COLORS["Red"]
        if 'effect' not in st.session_state:
            st.session_state.effect = "None"
        if 'disappearing_elements' not in st.session_state:
            st.session_state.disappearing_elements = []
        if 'drawing_path' not in st.session_state:
            st.session_state.drawing_path = []
        if 'permanent_drawing' not in st.session_state:
            st.session_state.permanent_drawing = None
        if 'temp_canvas' not in st.session_state:
            st.session_state.temp_canvas = None
        if 'template' not in st.session_state:
            st.session_state.template = "None"

        # Handle color selection
        if red: st.session_state.brush_color = COLORS["Red"]
        if blue: st.session_state.brush_color = COLORS["Blue"]
        if green: st.session_state.brush_color = COLORS["Green"]

        if effect: st.session_state.effect = effect
        if template: st.session_state.template = template

        # Gemini AI Setup
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

        # Canvas State Management
        if 'canvas' not in st.session_state:
            st.session_state.canvas = None
        if 'prev_pos' not in st.session_state:
            st.session_state.prev_pos = None
        if 'is_drawing' not in st.session_state:
            st.session_state.is_drawing = False
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'last_drawing_time' not in st.session_state:
            st.session_state.last_drawing_time = time.time()

        def save_canvas():
            if st.session_state.canvas is not None:
                filename = save_name if save_name else f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cv2.imwrite(f"{filename}.png", st.session_state.canvas)
                st.success(f"Saved as {filename}.png")

        def draw_eraser_preview(frame, position, size):
            """Draws a translucent red circle to show eraser area"""
            overlay = frame.copy()
            cv2.circle(overlay, position, size // 2, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.circle(frame, position, size // 2, (0, 0, 255), 2)

        def load_canvas():
            try:
                files = [f for f in os.listdir() if f.endswith('.png')]
                if files:
                    latest = max(files, key=os.path.getctime)
                    st.session_state.canvas = cv2.imread(latest)
                    st.success(f"Loaded {latest}")
            except Exception as e:
                st.error(f"Error loading: {str(e)}")

        def update_history():
            if st.session_state.canvas is not None:
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)
                st.session_state.history.append(st.session_state.canvas.copy())

        def undo_action():
            if st.session_state.history:
                st.session_state.canvas = st.session_state.history.pop()

        def apply_disappearing_effect():
            """Make drawn elements disappear after 6 seconds"""
            current_time = time.time()
            new_elements = []
            
            # Create a clean canvas
            if st.session_state.canvas is not None:
                st.session_state.canvas = np.zeros_like(st.session_state.canvas)
            
            for element in st.session_state.disappearing_elements:
                (start_time, points, color, size) = element
                elapsed = current_time - start_time
                
                if elapsed < 6:
                    opacity = max(0, 1.0 - (elapsed / 6.0))
                    blended_color = (
                        int(color[0] * opacity),
                        int(color[1] * opacity),
                        int(color[2] * opacity))
                    
                    for i in range(len(points) - 1):
                        cv2.line(
                            st.session_state.canvas,
                            points[i],
                            points[i+1],
                            blended_color,
                            size
                        )
                    
                    new_elements.append(element)
            
            st.session_state.disappearing_elements = new_elements

        def apply_template(canvas, template_type):
            """Apply the selected template to the canvas"""
            height, width = canvas.shape[:2]
            
            if template_type == "Graph Paper":
                # Draw graph paper with axes and grid
                color = (150, 150, 150)  # Light gray for grid
                
                # Draw X and Y axes
                cv2.line(canvas, (width//2, 0), (width//2, height), (0, 0, 255), 3)  # Y-axis (red)
                cv2.line(canvas, (0, height//2), (width, height//2), (255, 0, 0), 3)  # X-axis (blue)
                
                # Draw grid lines (every 50 pixels)
                grid_size = 50
                for x in range(0, width, grid_size):
                    cv2.line(canvas, (x, 0), (x, height), color, 2)
                for y in range(0, height, grid_size):
                    cv2.line(canvas, (0, y), (width, y), color, 2)
                
                # Add axis labels
                cv2.putText(canvas, "Y", (width//2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.putText(canvas, "X", (width - 30, height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
            
            elif template_type == "Music Staff":
                staff_color = (255, 255, 255)  # Black for staff lines
                staff_top = height // 3
                line_spacing = 20
                staff_width = width - 100
                
                # Draw the 5 lines of the staff - make them thicker
                for i in range(5):
                    y = staff_top + i * line_spacing
                    cv2.line(canvas, (50, y), (50 + staff_width, y), staff_color, 3)
                
                # Draw treble clef symbol (simplified)
                clef_points = [
                    (70, staff_top + 4 * line_spacing), (80, staff_top + 2 * line_spacing),
                    (90, staff_top + 4 * line_spacing), (80, staff_top + 1 * line_spacing)
                ]
                for i in range(len(clef_points) - 1):
                    cv2.line(canvas, clef_points[i], clef_points[i+1], staff_color, 2)
                
                # Draw vertical bar at end
                cv2.line(canvas, 
                        (50 + staff_width, staff_top), 
                        (50 + staff_width, staff_top + 4 * line_spacing), 
                        staff_color, 2)
            
            elif template_type == "Text Lines":
                # Add horizontal lines for writing practice
                line_color = (200, 200, 200)
                line_spacing = 40
                start_y = height // 4
                
                for i in range(8):
                    y = start_y + i * line_spacing
                    cv2.line(canvas, (50, y), (width - 50, y), line_color, 2)
            
            return canvas

        def detect_and_complete_shapes(points):
            """Enhanced shape detection with better accuracy"""
            if len(points) < 15:  # Increased minimum points for better accuracy
                return None
            
            points_np = np.array(points, dtype=np.int32)
            
            # Calculate convex hull to simplify shape
            hull = cv2.convexHull(points_np)
            
            # Calculate area and perimeter for shape analysis
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            
            # Skip too small shapes
            if area < 500 or perimeter < 50:
                return None
            
            # Shape approximation with dynamic epsilon
            epsilon = 0.03 * perimeter  # More precise approximation
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Shape detection with improved thresholds
            if len(approx) == 3:
                # Triangle detection
                return ("triangle", approx)
            
            elif len(approx) == 4:
                # Rectangle/Square detection with improved logic
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Calculate side lengths
                side1 = np.linalg.norm(box[0] - box[1])
                side2 = np.linalg.norm(box[1] - box[2])
                
                # Determine if square or rectangle
                aspect_ratio = max(side1, side2) / min(side1, side2)
                if 0.85 <= aspect_ratio <= 1.15:  # More forgiving square detection
                    return ("square", box)
                else:
                    return ("rectangle", box)
            
            else:
                # Circle detection with improved accuracy
                (x, y), radius = cv2.minEnclosingCircle(hull)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Calculate circularity
                circularity = 4 * math.pi * (area / (perimeter ** 2))
                
                if circularity > 0.7:  # Higher threshold for better circle detection
                    return ("circle", (center, radius))
            
            return None

        def draw_shape(canvas, shape_type, shape_data, color, thickness):
            """Draw the detected shape with proper visualization"""
            if shape_type == "circle":
                center, radius = shape_data
                cv2.circle(canvas, center, radius, color, thickness)
            elif shape_type in ["triangle", "square", "rectangle"]:
                cv2.drawContours(canvas, [shape_data], 0, color, thickness)
            
            # Add shape label
            if shape_type == "triangle":
                label_pos = shape_data[0][0]
            elif shape_type in ["square", "rectangle"]:
                label_pos = shape_data[0]
            elif shape_type == "circle":
                label_pos = (shape_data[0][0], shape_data[0][1] - shape_data[1] - 10)
            
            cv2.putText(canvas, shape_type.upper(), label_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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

        def process_drawing(info, img, display_frame):
            if info is None:
                st.session_state.is_drawing = False
                st.session_state.prev_pos = None
                return
            
            fingers = info['fingers']
            landmarks = info['landmarks']
            
            # Eraser Mode (pinky finger up)
            if fingers == [0, 0, 0, 0, 1] and len(landmarks) > 20:
                current_pos = landmarks[20][0:2]
                
                if st.session_state.prev_pos is not None:
                    cv2.line(
                        st.session_state.canvas, 
                        st.session_state.prev_pos, 
                        current_pos, 
                        (0, 0, 0),
                        eraser_size
                    )
                
                st.session_state.prev_pos = current_pos
            
            # Drawing Mode (index finger up)
            elif fingers == [0, 1, 0, 0, 0] and len(landmarks) > 8:
                current_pos = landmarks[8][0:2]
                
                if st.session_state.is_drawing and st.session_state.prev_pos is not None:
                    # For disappearing effect, we need to track the drawing path
                    if st.session_state.effect == "Disappearing Text":
                        if not hasattr(st.session_state, 'current_path'):
                            st.session_state.current_path = []
                            st.session_state.current_path_start = time.time()
                        st.session_state.current_path.append((st.session_state.prev_pos, current_pos))
                    
                    cv2.line(
                        st.session_state.canvas, 
                        st.session_state.prev_pos, 
                        current_pos, 
                        st.session_state.brush_color,
                        brush_size
                    )
                    
                    # Track drawing path for shape recognition
                    if st.session_state.effect == "Shape Assist":
                        st.session_state.drawing_path.append(current_pos)
                
                st.session_state.prev_pos = current_pos
                st.session_state.is_drawing = True
            
            # Shape Completion (index + middle fingers up)
            elif fingers == [0, 1, 1, 0, 0] and st.session_state.effect == "Shape Assist":
                if len(st.session_state.drawing_path) > 15:  # Minimum points increased
                    shape_data = detect_and_complete_shapes(st.session_state.drawing_path)
                    if shape_data:
                        shape_type, shape = shape_data
                        
                        # Save current state to history before modifications
                        update_history()
                        
                        # Create a clean canvas with all permanent drawings
                        if st.session_state.permanent_drawing is None:
                            st.session_state.permanent_drawing = np.zeros_like(img)
                        
                        # Draw the new perfect shape onto the permanent canvas
                        draw_shape(
                            st.session_state.permanent_drawing, 
                            shape_type, 
                            shape, 
                            st.session_state.brush_color, 
                            brush_size
                        )
                        
                        # Clear the rough drawing from the temp canvas
                        st.session_state.canvas = np.zeros_like(img)
                        
                        # Clear the drawing path
                        st.session_state.drawing_path = []
                
                st.session_state.drawing_path = []
            
            else:
                # When drawing stops, save the current path for disappearing effect
                if st.session_state.effect == "Disappearing Text" and hasattr(st.session_state, 'current_path') and st.session_state.current_path:
                    st.session_state.disappearing_elements.append((
                        st.session_state.current_path_start,
                        [point for segment in st.session_state.current_path for point in segment],
                        st.session_state.brush_color,
                        brush_size
                    ))
                    del st.session_state.current_path
                    del st.session_state.current_path_start
                
                st.session_state.is_drawing = False
                st.session_state.prev_pos = None
            
            # Clear canvas (thumb up)
            if fingers == [1, 0, 0, 0, 0]:
                update_history()
                st.session_state.canvas = np.zeros_like(img)
                st.session_state.permanent_drawing = None
                st.session_state.drawing_path = []
                st.session_state.disappearing_elements = []

        def process_ai_request(info):
            """Send drawing to Gemini AI when triggered (thumb + index + middle + ring fingers up)"""
            if info and info['fingers'] == [1, 1, 1, 1, 0]:
                try:
                    if st.session_state.canvas is not None:
                        pil_img = Image.fromarray(cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2RGB))
                        response = model.generate_content(["Solve this math problem and show your work:", pil_img])
                        return response.text
                except Exception as e:
                    st.error(f"AI error: {str(e)}")
            return None

        def perform_ocr_with_gemini(canvas_img):
            """Use Gemini AI for OCR instead of Tesseract"""
            try:
                # Create a copy of the canvas with better contrast for OCR
                ocr_image = canvas_img.copy()
                
                # Convert to grayscale
                gray = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
                
                # Apply thresholding to make text more visible
                _, thresh = cv2.threshold(gray, ocr_threshold, 255, cv2.THRESH_BINARY_INV)
                
                # Convert thresholded image back to RGB (white background, black text)
                inverted = cv2.bitwise_not(thresh)
                ocr_ready = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(ocr_ready)
                
                # Call Gemini AI with a prompt for OCR
                st.info("Sending to Gemini AI for text recognition...")
                response = model.generate_content([
                    "Please perform OCR (Optical Character Recognition) on this image and extract all text you can see. Return only the extracted text without any explanations or additional comments.",
                    pil_img
                ])
                
                recognized_text = response.text.strip()
                return recognized_text if recognized_text else "No text recognized"
                
            except Exception as e:
                st.error(f" OCR Error: {str(e)}")
                return " OCR processing failed"

        # Main application loop
        prev_time = time.time()
        frame_count = 0
        fps = 0
        last_ocr_time = 0
        ocr_interval = 2  # seconds between live OCR scans

        if analyze_text and st.session_state.last_recognized_text:
            with st.spinner("Analyzing text..."):
                analysis = model.generate_content([
                    f"Analyze the following text that was detected from a drawing:\n\n{st.session_state.last_recognized_text}\n\nProvide insights, corrections if it appears to contain errors, and explain what this text might represent or mean."
                ])
                st.markdown(f"**Analysis:**\n\n{analysis.text}")

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
            
            # Initialize canvases if needed
            if st.session_state.canvas is None:
                st.session_state.canvas = np.zeros_like(frame)
            if st.session_state.permanent_drawing is None:
                st.session_state.permanent_drawing = np.zeros_like(frame)
            
            if clear_button:
                update_history()
                st.session_state.canvas = np.zeros_like(frame)
                st.session_state.permanent_drawing = np.zeros_like(frame)
                st.session_state.disappearing_elements = []
                st.session_state.drawing_path = []
                clear_button = False
            
            if save_btn:
                save_canvas()
                save_btn = False
            
            if load_btn:
                load_canvas()
                load_btn = False
            
            if undo_btn:
                undo_action()
                undo_btn = False
            
            # Create the display frame by combining:
            # 1. Webcam background
            # 2. Template (if any)
            # 3. Permanent drawings (completed shapes)
            # 4. Current temporary drawing (rough sketches)
            display_frame = frame.copy()
            
            # Apply template if selected
            if st.session_state.template != "None":
                template_canvas = np.zeros_like(frame)
                template_canvas = apply_template(template_canvas, st.session_state.template)
                display_frame = cv2.addWeighted(display_frame, 0.8, template_canvas, 0.2, 0)
            
            # First add permanent drawings
            if st.session_state.permanent_drawing is not None:
                display_frame = cv2.addWeighted(display_frame, 0.7, st.session_state.permanent_drawing, 0.3, 0)
            
            # Apply disappearing effect if enabled
            if st.session_state.effect == "Disappearing Text":
                apply_disappearing_effect()
            
            # Then add current temporary drawing
            display_frame = cv2.addWeighted(display_frame, 0.7, st.session_state.canvas, 0.3, 0)
            
            hand_info = get_hand_info(frame)
            current_mode = "Ready"
            
            if hand_info:
                fingers = hand_info['fingers']
                landmarks = hand_info['landmarks']
                
                # Eraser preview
                if fingers == [0, 0, 0, 0, 1] and len(landmarks) > 20:
                    current_pos = landmarks[20][0:2]
                    current_mode = "Eraser"
                    draw_eraser_preview(display_frame, current_pos, eraser_size)
                
                # AI request (thumb + index + middle + ring fingers up)
                if fingers == [1, 1, 1, 1, 0]:
                    solution = process_ai_request(hand_info)
                    if solution:
                        output_text_area.markdown(f"**AI Solution:**\n\n{solution}")
            
            process_drawing(hand_info, frame, display_frame)
            
            # Handle OCR requests
            current_time = time.time()
            if (ocr_btn or (live_ocr and current_time - last_ocr_time > ocr_interval)) and st.session_state.canvas is not None:
                # Combine permanent and temporary drawings for OCR
                ocr_canvas = np.zeros_like(frame)
                if st.session_state.permanent_drawing is not None:
                    ocr_canvas = cv2.add(ocr_canvas, st.session_state.permanent_drawing)
                ocr_canvas = cv2.add(ocr_canvas, st.session_state.canvas)
                
                # Show processing indicator
                with st.spinner("Processing with Gemini AI..."):
                    
                    recognized_text = perform_ocr_with_gemini(ocr_canvas)
                    if recognized_text:
                        st.session_state.last_recognized_text = recognized_text
                        output_text_area.markdown(f"**Recognized Text (Gemini AI):**\n\n{recognized_text}")
                        last_ocr_time = current_time
                        ocr_btn = False
            
            # Update status
            status_text = f"Mode: {current_mode} | "
            status_text += f"Brush: {brush_size}px | Eraser: {eraser_size}px | "
            status_text += f"Color: {list(COLORS.keys())[list(COLORS.values()).index(st.session_state.brush_color)]} | "
            status_text += f"Effect: {st.session_state.effect} | "
            status_text += f"Template: {st.session_state.template}"
            status.markdown(f"**{status_text}**")
            
            # Calculate FPS
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1:
                fps = frame_count
                frame_count = 0
                prev_time = curr_time
            fps_display.markdown(f"**FPS:** {fps}")
            
            FRAME_WINDOW.image(display_frame, channels="BGR")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()