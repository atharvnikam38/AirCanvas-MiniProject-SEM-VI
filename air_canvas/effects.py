# effects.py
import cv2
import numpy as np
import time
import math

class VisualEffects:
    @staticmethod
    def apply_disappearing_effect(canvas, elements):
        current_time = time.time()
        new_elements = []
        canvas.fill(0)
        
        for element in elements:
            start_time, points, color, size = element
            elapsed = current_time - start_time
            
            if elapsed < 6:
                opacity = max(0, 1.0 - (elapsed / 6.0))
                blended_color = (
                    int(color[0] * opacity),
                    int(color[1] * opacity),
                    int(color[2] * opacity))
                
                for i in range(len(points) - 1):
                    cv2.line(
                        canvas,
                        points[i],
                        points[i+1],
                        blended_color,
                        size
                    )
                new_elements.append(element)
        
        return new_elements

    @staticmethod
    def apply_template(canvas, template_type):
        height, width = canvas.shape[:2]
        
        if template_type == "Graph Paper":
            color = (150, 150, 150)
            cv2.line(canvas, (width//2, 0), (width//2, height), (0, 0, 255), 3)
            cv2.line(canvas, (0, height//2), (width, height//2), (255, 0, 0), 3)
            
            grid_size = 50
            for x in range(0, width, grid_size):
                cv2.line(canvas, (x, 0), (x, height), color, 2)
            for y in range(0, height, grid_size):
                cv2.line(canvas, (0, y), (width, y), color, 2)
            
            cv2.putText(canvas, "Y", (width//2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(canvas, "X", (width - 30, height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        
        elif template_type == "Music Staff":
            staff_color = (255, 255, 255)
            staff_top = height // 3
            line_spacing = 20
            staff_width = width - 100
            
            for i in range(5):
                y = staff_top + i * line_spacing
                cv2.line(canvas, (50, y), (50 + staff_width, y), staff_color, 3)
            
            clef_points = [
                (70, staff_top + 4 * line_spacing), (80, staff_top + 2 * line_spacing),
                (90, staff_top + 4 * line_spacing), (80, staff_top + 1 * line_spacing)
            ]
            for i in range(len(clef_points) - 1):
                cv2.line(canvas, clef_points[i], clef_points[i+1], staff_color, 2)
            
            cv2.line(canvas, 
                    (50 + staff_width, staff_top), 
                    (50 + staff_width, staff_top + 4 * line_spacing), 
                    staff_color, 2)
        
        elif template_type == "Text Lines":
            line_color = (200, 200, 200)
            line_spacing = 40
            start_y = height // 4
            
            for i in range(8):
                y = start_y + i * line_spacing
                cv2.line(canvas, (50, y), (width - 50, y), line_color, 2)
        
        return canvas

    @staticmethod
    def detect_and_complete_shapes(points):
        if len(points) < 15:
            return None
        
        points_np = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_np)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        
        if area < 500 or perimeter < 50:
            return None
        
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 3:
            return ("triangle", approx)
        
        elif len(approx) == 4:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            side1 = np.linalg.norm(box[0] - box[1])
            side2 = np.linalg.norm(box[1] - box[2])
            aspect_ratio = max(side1, side2) / min(side1, side2)
            if 0.85 <= aspect_ratio <= 1.15:
                return ("square", box)
            else:
                return ("rectangle", box)
        
        else:
            (x, y), radius = cv2.minEnclosingCircle(hull)
            center = (int(x), int(y))
            radius = int(radius)
            circularity = 4 * math.pi * (area / (perimeter ** 2))
            if circularity > 0.7:
                return ("circle", (center, radius))
        
        return None

    @staticmethod
    def draw_shape(canvas, shape_type, shape_data, color, thickness):
        if shape_type == "circle":
            center, radius = shape_data
            cv2.circle(canvas, center, radius, color, thickness)
        elif shape_type in ["triangle", "square", "rectangle"]:
            cv2.drawContours(canvas, [shape_data], 0, color, thickness)
        
        if shape_type == "triangle":
            label_pos = shape_data[0][0]
        elif shape_type in ["square", "rectangle"]:
            label_pos = shape_data[0]
        elif shape_type == "circle":
            label_pos = (shape_data[0][0], shape_data[0][1] - shape_data[1] - 10)
        
        cv2.putText(canvas, shape_type.upper(), label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)