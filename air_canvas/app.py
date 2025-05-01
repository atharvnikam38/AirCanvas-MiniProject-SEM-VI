# app.py
import cv2
import numpy as np
import streamlit as st
import time
from PIL import Image
from config import COLORS
from canvas_manager import CanvasManager
from hand_gesture import HandGestureDetector
from ai_integration import AIIntegration
from effects import VisualEffects
from utils import initialize_camera, calculate_fps

# Streamlit UI Setup
def setup_ui():
    st.set_page_config(layout="wide")
    st.title("🌟 Enhanced Air Canvas with Gemini AI OCR & Visual Effects ✨")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        run = st.checkbox('Enable Camera', value=True)
        frame_window = st.image([], channels="BGR")
        
        st.markdown("### 🎨 Brush Settings")
        st.markdown("**Select Color:**")
        color_cols = st.columns(4)
        with color_cols[0]: red = st.button("🔴 Red", use_container_width=True)
        with color_cols[1]: blue = st.button("🔵 Blue", use_container_width=True)
        with color_cols[2]: green = st.button("🟢 Green", use_container_width=True)
        
        brush_size = st.slider("🖌️ Brush Size", 5, 30, 10)
        eraser_size = st.slider("🧽 Eraser Size", 10, 50, 20)
        
        st.markdown("### ✨ Visual Effects")
        effect = st.selectbox("Choose Effect", ["None", "Disappearing Text", "Shape Assist"])
        
        st.markdown("### 📝 Templates")
        template = st.selectbox("Choose Template", ["None", "Graph Paper", "Music Staff", "Text Lines"])
        
        st.markdown("### 💾 Save/Load")
        save_name = st.text_input("Filename:")
        save_cols = st.columns(2)
        with save_cols[0]: save_btn = st.button("💾 Save Drawing", use_container_width=True)
        with save_cols[1]: load_btn = st.button("📂 Load Last", use_container_width=True)
    
    with col2:
        st.header("🔍 Gemini AI OCR & Solutions")
        output_text_area = st.empty()
        
        st.markdown("### 🔤 OCR with Gemini AI")
        ocr_threshold = st.slider("Image Contrast Threshold", 0, 255, 180)
        ocr_cols = st.columns(2)
        with ocr_cols[0]: ocr_btn = st.button("🔍 Run OCR", use_container_width=True)
        with ocr_cols[1]: live_ocr = st.checkbox("Live OCR", value=False)
        
        if "last_recognized_text" not in st.session_state:
            st.session_state.last_recognized_text = ""
        
        analyze_text = st.button("🧠 Analyze Last Recognized Text")
        
        st.markdown("### 🛠️ Canvas Controls")
        clear_cols = st.columns(2)
        with clear_cols[0]: clear_button = st.button("🧹 Clear Canvas", use_container_width=True)
        with clear_cols[1]: undo_btn = st.button("↩️ Undo", use_container_width=True)
        
        st.markdown("### 📊 Status")
        status = st.empty()
        fps_display = st.empty()
    
    return {
        "run": run,
        "frame_window": frame_window,
        "red": red,
        "blue": blue,
        "green": green,
        "brush_size": brush_size,
        "eraser_size": eraser_size,
        "effect": effect,
        "template": template,
        "save_name": save_name,
        "save_btn": save_btn,
        "load_btn": load_btn,
        "output_text_area": output_text_area,
        "ocr_threshold": ocr_threshold,
        "ocr_btn": ocr_btn,
        "live_ocr": live_ocr,
        "analyze_text": analyze_text,
        "clear_button": clear_button,
        "undo_btn": undo_btn,
        "status": status,
        "fps_display": fps_display
    }

def process_drawing(hand_info, canvas_manager, brush_color, brush_size, eraser_size, effect):
    if hand_info is None:
        canvas_manager.is_drawing = False
        canvas_manager.prev_pos = None
        return
    
    fingers = hand_info['fingers']
    landmarks = hand_info['landmarks']
    
    # Eraser Mode
    if fingers == [0, 0, 0, 0, 1] and len(landmarks) > 20:
        current_pos = landmarks[20][0:2]
        
        if canvas_manager.prev_pos is not None:
            cv2.line(
                canvas_manager.canvas, 
                canvas_manager.prev_pos, 
                current_pos, 
                (0, 0, 0),
                eraser_size
            )
        
        canvas_manager.prev_pos = current_pos
    
    # Drawing Mode
    elif fingers == [0, 1, 0, 0, 0] and len(landmarks) > 8:
        current_pos = landmarks[8][0:2]
        
        if canvas_manager.is_drawing and canvas_manager.prev_pos is not None:
            if effect == "Disappearing Text":
                if canvas_manager.current_path is None:
                    canvas_manager.current_path = []
                    canvas_manager.current_path_start = time.time()
                canvas_manager.current_path.append((canvas_manager.prev_pos, current_pos))
            
            cv2.line(
                canvas_manager.canvas, 
                canvas_manager.prev_pos, 
                current_pos, 
                brush_color,
                brush_size
            )
            
            if effect == "Shape Assist":
                canvas_manager.drawing_path.append(current_pos)
        
        canvas_manager.prev_pos = current_pos
        canvas_manager.is_drawing = True
    
    # Shape Completion
    elif fingers == [0, 1, 1, 0, 0] and effect == "Shape Assist":
        if len(canvas_manager.drawing_path) > 15:
            shape_data = VisualEffects.detect_and_complete_shapes(canvas_manager.drawing_path)
            if shape_data:
                shape_type, shape = shape_data
                canvas_manager.update_history()
                
                if canvas_manager.permanent_drawing is None:
                    canvas_manager.permanent_drawing = np.zeros_like(canvas_manager.canvas)
                
                VisualEffects.draw_shape(
                    canvas_manager.permanent_drawing, 
                    shape_type, 
                    shape, 
                    brush_color, 
                    brush_size
                )
                
                canvas_manager.canvas.fill(0)
                canvas_manager.drawing_path = []
        
        canvas_manager.drawing_path = []
    
    else:
        if effect == "Disappearing Text" and canvas_manager.current_path and canvas_manager.current_path_start:
            canvas_manager.disappearing_elements.append((
                canvas_manager.current_path_start,
                [point for segment in canvas_manager.current_path for point in segment],
                brush_color,
                brush_size
            ))
            canvas_manager.current_path = None
            canvas_manager.current_path_start = None
        
        canvas_manager.is_drawing = False
        canvas_manager.prev_pos = None
    
    # Clear canvas (thumb up)
    if fingers == [1, 0, 0, 0, 0]:
        canvas_manager.clear()

def main():
    ui = setup_ui()
    
    # Initialize components
    cap = initialize_camera()
    if not cap:
        st.error("Could not initialize webcam. Please check your camera connection.")
        return
    
    hand_detector = HandGestureDetector()
    ai_integration = AIIntegration()
    
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from camera.")
        return
    
    canvas_manager = CanvasManager(frame.shape)
    visual_effects = VisualEffects()
    
    # Initialize session state for brush color
    if 'brush_color' not in st.session_state:
        st.session_state.brush_color = COLORS["Red"]
    
    # Handle color selection
    if ui["red"]: st.session_state.brush_color = COLORS["Red"]
    if ui["blue"]: st.session_state.brush_color = COLORS["Blue"]
    if ui["green"]: st.session_state.brush_color = COLORS["Green"]
    
    # Handle text analysis
    if ui["analyze_text"] and st.session_state.last_recognized_text:
        with st.spinner("Analyzing text..."):
            analysis = ai_integration.analyze_text(st.session_state.last_recognized_text)
            ui["output_text_area"].markdown(f"**Analysis:**\n\n{analysis}")
    
    # Main loop
    prev_time = time.time()
    frame_count = 0
    fps = 0
    last_ocr_time = 0
    ocr_interval = 2
    
    while ui["run"]:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera error, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = initialize_camera()
            if not cap:
                st.error("Camera failed")
                ui["run"] = False
                break
            continue
        
        frame = cv2.flip(frame, 1)
        
        if ui["clear_button"]:
            canvas_manager.clear()
            ui["clear_button"] = False
        
        if ui["save_btn"]:
            result = canvas_manager.save_canvas(ui["save_name"])
            st.success(result)
            ui["save_btn"] = False
        
        if ui["load_btn"]:
            result = canvas_manager.load_canvas()
            st.success(result)
            ui["load_btn"] = False
        
        if ui["undo_btn"]:
            if canvas_manager.undo():
                st.success("Undo successful")
            else:
                st.warning("Nothing to undo")
            ui["undo_btn"] = False
        
        # Create display frame
        display_frame = frame.copy()
        
        # Apply template if selected
        if ui["template"] != "None":
            template_canvas = np.zeros_like(frame)
            visual_effects.apply_template(template_canvas, ui["template"])
            display_frame = cv2.addWeighted(display_frame, 0.8, template_canvas, 0.2, 0)
        
        # Apply disappearing effect if enabled
        if ui["effect"] == "Disappearing Text":
            canvas_manager.disappearing_elements = visual_effects.apply_disappearing_effect(
                canvas_manager.canvas,
                canvas_manager.disappearing_elements
            )
        
        # First add permanent drawings
        if canvas_manager.permanent_drawing is not None:
            display_frame = cv2.addWeighted(display_frame, 0.7, canvas_manager.permanent_drawing, 0.3, 0)
        
        # Then add current temporary drawing
        display_frame = cv2.addWeighted(display_frame, 0.7, canvas_manager.canvas, 0.3, 0)
        
        hand_info = hand_detector.get_hand_info(frame)
        current_mode = "Ready"
        
        if hand_info:
            fingers = hand_info['fingers']
            landmarks = hand_info['landmarks']
            
            # Eraser preview
            if fingers == [0, 0, 0, 0, 1] and len(landmarks) > 20:
                current_pos = landmarks[20][0:2]
                current_mode = "Eraser"
                display_frame = hand_detector.draw_eraser_preview(display_frame, current_pos, ui["eraser_size"])
            
            # AI request (thumb + index + middle + ring fingers up)
            if fingers == [1, 1, 1, 1, 0]:
                solution = ai_integration.process_ai_request(canvas_manager.canvas)
                if solution:
                    ui["output_text_area"].markdown(f"**AI Solution:**\n\n{solution}")
        
        process_drawing(
            hand_info,
            canvas_manager,
            st.session_state.brush_color,
            ui["brush_size"],
            ui["eraser_size"],
            ui["effect"]
        )
        
        # Handle OCR requests
        current_time = time.time()
        if (ui["ocr_btn"] or (ui["live_ocr"] and current_time - last_ocr_time > ocr_interval)) and canvas_manager.canvas is not None:
            ocr_canvas = np.zeros_like(frame)
            if canvas_manager.permanent_drawing is not None:
                ocr_canvas = cv2.add(ocr_canvas, canvas_manager.permanent_drawing)
            ocr_canvas = cv2.add(ocr_canvas, canvas_manager.canvas)
            
            with st.spinner("Processing with Gemini AI..."):
                recognized_text = ai_integration.perform_ocr(ocr_canvas, ui["ocr_threshold"])
                if recognized_text:
                    st.session_state.last_recognized_text = recognized_text
                    ui["output_text_area"].markdown(f"**Recognized Text (Gemini AI):**\n\n{recognized_text}")
                    last_ocr_time = current_time
                    ui["ocr_btn"] = False
        
        # Update status
        status_text = f"Mode: {current_mode} | "
        status_text += f"Brush: {ui['brush_size']}px | Eraser: {ui['eraser_size']}px | "
        status_text += f"Color: {list(COLORS.keys())[list(COLORS.values()).index(st.session_state.brush_color)]} | "
        status_text += f"Effect: {ui['effect']} | "
        status_text += f"Template: {ui['template']}"
        ui["status"].markdown(f"**{status_text}**")
        
        # Calculate FPS
        fps, prev_time, frame_count = calculate_fps(prev_time, frame_count)
        if fps is not None:
            ui["fps_display"].markdown(f"**FPS:** {fps}")
        
        ui["frame_window"].image(display_frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()