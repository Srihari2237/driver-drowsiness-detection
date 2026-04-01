import cv2
import mediapipe as mp
import math
import collections
import numpy as np
from src.ear import eye_aspect_ratio
from src.mar import mouth_aspect_ratio
from src.alarm import play_alarm, stop_alarm
from utils.config import *

class DrowsinessDetector:
    def __init__(self):
        self.counter = 0
        self.distract_counter = 0
        self.combo_counter = 0 
        
        # Buffers for Smoothing (Increased for Specs users)
        self.ear_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.gaze_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

        self.is_calibrating = True
        self.calibration_frame_count = 0
        self.ear_sum, self.mar_sum, self.head_v_sum, self.head_h_sum = 0, 0, 0, 0
        
        self.dynamic_ear_threshold = EAR_THRESHOLD 
        self.dynamic_mar_threshold = MAR_THRESHOLD 
        self.dynamic_head_v_threshold = HEAD_DROP_THRESHOLD
        self.dynamic_head_h_l, self.dynamic_head_h_r = HEAD_TURN_THRESH_L, HEAD_TURN_THRESH_R

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    def get_gaze_ratio(self, eye_points, landmarks):
        try:
            iris_idx = 468 if eye_points[0] == 33 else 473 
            pupil_x = landmarks[iris_idx].x
            l_corner_x = landmarks[eye_points[0]].x
            r_corner_x = landmarks[eye_points[3]].x
            return (pupil_x - l_corner_x) / (r_corner_x - l_corner_x) if (r_corner_x - l_corner_x) != 0 else 0.5
        except:
            return 0.5

    def process_frame(self, frame):
        ear, mar, avg_gaze, head_h_ratio = 0.0, 0.0, 0.5, 0.5
        status = "NO FACE"
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # -------------------------------------------------------
            # 1. VISIBILITY CHECK (The Specs Fix)
            # -------------------------------------------------------
            # We check the Z-depth and presence of Iris landmarks.
            # If the depth is abnormal, it usually means glasses reflection is blocking the view.
            eyes_obscured = False
            try:
                # If irises are missing or coordinates are invalid
                if landmarks[468].x == 0 or landmarks[473].x == 0:
                    eyes_obscured = True
            except:
                eyes_obscured = True

            # -------------------------------------------------------
            # 2. LANDMARK EXTRACTION
            # -------------------------------------------------------
            l_eye_idx, r_eye_idx = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
            mouth = [(landmarks[i].x, landmarks[i].y) for i in [61, 81, 13, 311, 291, 308, 402, 178]]
            mar = mouth_aspect_ratio(mouth)
            
            # Head Pose - This remains highly accurate even with glasses!
            head_v_ratio = math.dist((landmarks[1].x, landmarks[1].y), (landmarks[152].x, landmarks[152].y)) / \
                           math.dist((landmarks[33].x, landmarks[33].y), (landmarks[263].x, landmarks[263].y))
            head_h_ratio = (landmarks[1].x - landmarks[33].x) / (landmarks[263].x - landmarks[33].x) if (landmarks[263].x - landmarks[33].x) != 0 else 0.5

            # Eye Metrics (Only process if not obscured)
            if not eyes_obscured:
                left_eye = [(landmarks[i].x, landmarks[i].y) for i in l_eye_idx]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in r_eye_idx]
                raw_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                raw_gaze = (self.get_gaze_ratio(l_eye_idx, landmarks) + self.get_gaze_ratio(r_eye_idx, landmarks)) / 2.0
                
                # Apply heavy smoothing for glasses
                self.ear_buffer.append(raw_ear)
                self.gaze_buffer.append(raw_gaze)
                ear = sum(self.ear_buffer) / len(self.ear_buffer)
                avg_gaze = sum(self.gaze_buffer) / len(self.gaze_buffer)
            else:
                # If eyes are obscured, reuse the last stable EAR/Gaze or baseline
                ear = getattr(self, "last_ear", self.dynamic_ear_threshold)
                avg_gaze = 0.5

            # -------------------------------------------------------
            # 3. MONITORING LOGIC (Priority Shift)
            # -------------------------------------------------------
            if self.is_calibrating:
                # ... [Keep calibration logic the same] ...
                self.ear_sum += ear; self.mar_sum += mar; self.head_v_sum += head_v_ratio; self.head_h_sum += head_h_ratio
                self.calibration_frame_count += 1
                status = "CALIBRATING"
                if self.calibration_frame_count >= CALIBRATION_FRAMES:
                    self.dynamic_ear_threshold = (self.ear_sum / CALIBRATION_FRAMES) * EAR_DROP_PERCENTAGE
                    self.dynamic_mar_threshold = (self.mar_sum / CALIBRATION_FRAMES) * MAR_RISE_PERCENTAGE
                    self.dynamic_head_v_threshold = (self.head_v_sum / CALIBRATION_FRAMES) * 0.88
                    self.is_calibrating = False
            else:
                status = "ALERT"
                head_turned = (head_h_ratio < self.dynamic_head_h_l or head_h_ratio > self.dynamic_head_h_r)
                head_dropped = (head_v_ratio < self.dynamic_head_v_threshold)
                
                # If eyes are obscured, we rely 100% on Head and Mouth
                if head_dropped:
                    status = "HEAD DROP"
                elif head_turned:
                    status = "DISTRACTED"
                elif mar > self.dynamic_mar_threshold:
                    status = "YAWNING"
                elif not eyes_obscured:
                    # Eye logic only if eyes are visible
                    eyes_away = (avg_gaze < GAZE_THRESH_L or avg_gaze > GAZE_THRESH_R)
                    if eyes_away:
                        self.distract_counter += 1
                        if self.distract_counter > GAZE_FRAMES: status = "DISTRACTED"
                    elif ear < self.dynamic_ear_threshold:
                        self.counter += 2
                        if self.counter >= DROWSY_FRAMES: status = "DROWSY"
                    else:
                        self.counter = max(0, self.counter - 1); self.distract_counter = 0
                else:
                    # EYES OBSCURED MODE
                    status = "ALERT (🕶️ SUNGLASSES/GLARE)"

                # Alarm Trigger
                if status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED"]:
                    play_alarm()
                else:
                    stop_alarm()

            
        else:
            stop_alarm(); status = "NO FACE"

        self.last_ear, self.last_mar, self.last_status, self.last_gaze, self.last_head_h = ear, mar, status, avg_gaze, head_h_ratio
        return frame