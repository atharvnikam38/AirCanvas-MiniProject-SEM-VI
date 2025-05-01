# ai_integration.py
import google.generativeai as genai
from PIL import Image
import cv2
import streamlit as st

class AIIntegration:
    def __init__(self):
        try:
            genai.configure(api_key=["AIzaSyAuW8PdguB7eF1vwD2cM8mvWtO1EKmicrk"])
            self.model = genai.GenerativeModel(["gemini-2.0-flash"])
        except Exception as e:
            st.error(f"Failed to initialize Gemini AI: {str(e)}")
            raise

    def perform_ocr(self, canvas_img, threshold=180):
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            inverted = cv2.bitwise_not(thresh)
            ocr_ready = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(ocr_ready)
            
            response = self.model.generate_content([
                "Perform OCR on this image and extract all text. Return only the extracted text.",
                pil_img
            ])
            return response.text.strip() or "No text recognized"
        except Exception as e:
            return f"OCR processing failed: {str(e)}"

    def analyze_text(self, text):
        try:
            response = self.model.generate_content([
                f"Analyze this text:\n\n{text}\n\nProvide insights and corrections."
            ])
            return response.text
        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def process_ai_request(self, canvas_img, prompt="Solve this math problem and show your work:"):
        try:
            pil_img = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
            response = self.model.generate_content([prompt, pil_img])
            return response.text
        except Exception as e:
            return f"AI error: {str(e)}"