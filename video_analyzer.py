import cv2
import mediapipe as mp
import numpy as np
import math

class VideoAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def analyze_ad_video(self, video_path: str, ml_rating: float, ml_success_prob: float, ml_money_pred: str) -> str:
        """
        Reads a video file and uses OpenCV and MediaPipe to generate
        a structured marketing insight report without any cloud APIs.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return "Error: Could not open video file for analysis."

        # Video Properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        # Metrics to track
        total_brightness = 0
        cut_count = 0
        total_faces_detected = 0
        
        # Frame processing vars
        prev_hist = None
        frames_processed = 0
        
        # We don't need to process every single frame for pacing and brightness.
        # Analyzing every Nth frame is much faster and yields same marketing heuristics.
        frame_skip = int(fps / 5) if fps > 5 else 1  # Process ~5 frames per second
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Skip frames to optimize speed
            if current_frame_pos % frame_skip != 0:
                continue
                
            frames_processed += 1
            
            # 1. Brightness / Vibrancy
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = hsv[..., 2].mean()
            total_brightness += brightness
            
            # 2. Scene Cuts / Pacing (using Histogram differences)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                # Compare current frame histogram with previous
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > 0.35: # Threshold for a scene cut
                    cut_count += 1
            prev_hist = hist
            
            # 3. Face Detection (Analyze 1 frame per second to estimate actors)
            if frames_processed % 5 == 0:
                # Convert the BGR image to RGB before processing with MediaPipe.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(image_rgb)
                
                if results.detections:
                    total_faces_detected += len(results.detections)

        cap.release()

        # Compile Averages
        avg_brightness = total_brightness / frames_processed if frames_processed > 0 else 0
        
        # Heuristic Logic
        vibrancy_label = "High (Vibrant/Energetic)" if avg_brightness > 130 else "Moderate/Standard" if avg_brightness > 90 else "Low (Dark/Moody)"
        
        pacing_label = "Fast-Paced" if cut_count > (duration_seconds / 2) else "Moderate" if cut_count > (duration_seconds / 5) else "Slow/Continuous"
        
        face_density = total_faces_detected / (duration_seconds if duration_seconds > 0 else 1)
        human_focus = "High Human Presence (Actor/Spokesperson driven)" if face_density > 0.5 else "Low/Product-Focused"

        # Generate Marketing Report
        report = f"""
        ### 🤖 Computer Vision Insights
        *Analysis generated entirely locally using OpenCV and MediaPipe.*
        
        **Visual Profile:**
        - **Lighting & Vibrancy:** {vibrancy_label} (Avg Brightness score: {avg_brightness:.1f}/255)
        - **Pacing & Editing:** {pacing_label} ({cut_count} scene cuts detected over {duration_seconds:.1f} seconds)
        - **Subject Focus:** {human_focus}
        
        **Predicted Performance Synergy:**
        - Incorporating the **{ml_success_prob:.1f}% Success Probability** and **{ml_rating:.2f}/5 Rating**, this ad's {pacing_label.lower()} visual style is 
          {"likely contributing positively" if ml_success_prob > 60 else "potentially holding back"} the predicted outcomes.
        - **Guarantee Status:** The model predicts a "{ml_money_pred}" for a money-back guarantee. {"This strong offer pairs well with the visual style." if ml_money_pred == "Yes" else "Adding a guarantee might boost the metrics further."}
        
        **Key Recommendations:**
        - {"Brighten the footage to make the product stand out more." if avg_brightness < 90 else "Lighting is well-optimized."}
        - {"Increase the pacing/cuts to retain viewer attention." if pacing_label == "Slow/Continuous" else "Editing tempo is engaging."}
        """
        
        return report.strip().replace("        ", "")
