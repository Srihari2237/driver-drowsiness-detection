import cv2
import mediapipe as mp
import math
import collections
from src.ear import eye_aspect_ratio
from src.mar import mouth_aspect_ratio
from src.alarm import play_alarm, stop_alarm
from utils.config import *
 
class DrowsinessDetector:
    def __init__(self):
        self.counter = 0
        self.distract_counter = 0
        
        # Buffers for smoothing (Handles glasses glare/flicker)
        self.ear_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.gaze_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
 
        self.is_calibrating = True
        self.calibration_frame_count = 0
        self.ear_sum, self.mar_sum, self.head_sum = 0, 0, 0
        
        self.dynamic_ear_threshold = EAR_THRESHOLD 
        self.dynamic_mar_threshold = MAR_THRESHOLD 
        self.dynamic_head_threshold = HEAD_DROP_THRESHOLD
 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True  # Required for Iris tracking
        )
 
    def get_gaze_ratio(self, eye_points, landmarks, is_right_eye=False):
        """
        Returns gaze ratio (0.0 = looking far left, 1.0 = looking far right).
        For the right eye, the iris landmark is 473 and x-axis is mirrored,
        so we invert the ratio to stay consistent with the left eye.
        """
        iris_idx = 473 if is_right_eye else 468
        pupil = landmarks[iris_idx].x
        l_corner = landmarks[eye_points[0]].x
        r_corner = landmarks[eye_points[3]].x
 
        if (r_corner - l_corner) == 0:
            return 0.5
 
        ratio = (pupil - l_corner) / (r_corner - l_corner)
 
        # FIX: Right eye x-coords are mirrored in MediaPipe — invert to match left eye direction
        if is_right_eye:
            ratio = 1.0 - ratio
 
        return ratio
 
    def process_frame(self, frame):
        ear = getattr(self, "last_ear", 0.0)
        mar = getattr(self, "last_mar", 0.0)
        status = "NO FACE"
 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
 
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Landmark indices
            l_eye_idx = [33, 160, 158, 133, 153, 144]
            r_eye_idx = [362, 385, 387, 263, 373, 380]
 
            left_eye  = [(landmarks[i].x, landmarks[i].y) for i in l_eye_idx]
            right_eye = [(landmarks[i].x, landmarks[i].y) for i in r_eye_idx]
            mouth     = [(landmarks[i].x, landmarks[i].y) for i in [61, 81, 13, 311, 291, 308, 402, 178]]
 
            raw_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Head Drop: nose-to-chin / face width ratio
            head_ratio = math.dist(
                (landmarks[1].x,  landmarks[1].y),
                (landmarks[152].x, landmarks[152].y)
            ) / math.dist(
                (landmarks[33].x, landmarks[33].y),
                (landmarks[263].x, landmarks[263].y)
            )
 
            # FIX: Pass is_right_eye flag so right eye gaze is correctly un-mirrored
            left_gaze  = self.get_gaze_ratio(l_eye_idx, landmarks, is_right_eye=False)
            right_gaze = self.get_gaze_ratio(r_eye_idx, landmarks, is_right_eye=True)
            raw_gaze   = (left_gaze + right_gaze) / 2.0
 
            # 2. Smoothing
            self.ear_buffer.append(raw_ear)
            self.gaze_buffer.append(raw_gaze)
            ear      = sum(self.ear_buffer)   / len(self.ear_buffer)
            avg_gaze = sum(self.gaze_buffer)  / len(self.gaze_buffer)
 
            # 3. Calibration
            if self.is_calibrating:
                self.ear_sum  += ear
                self.mar_sum  += mar
                self.head_sum += head_ratio
                self.calibration_frame_count += 1
                status = "CALIBRATING"
                if self.calibration_frame_count >= CALIBRATION_FRAMES:
                    self.dynamic_ear_threshold  = (self.ear_sum  / CALIBRATION_FRAMES) * EAR_DROP_PERCENTAGE
                    self.dynamic_mar_threshold  = (self.mar_sum  / CALIBRATION_FRAMES) * MAR_RISE_PERCENTAGE
                    self.dynamic_head_threshold = (self.head_sum / CALIBRATION_FRAMES) * 0.88
                    self.is_calibrating = False
 
            # 4. Detection (post-calibration)
            else:
                status = "ALERT"
 
                if head_ratio < self.dynamic_head_threshold:
                    # Head has dropped — reset unrelated counters
                    status = "HEAD DROP"
                    self.counter = 0
                    self.distract_counter = 0
 
                elif avg_gaze < GAZE_THRESH_L or avg_gaze > GAZE_THRESH_R:
                    # Gaze is off-center
                    self.distract_counter += 1
                    self.counter = 0  # FIX: reset drowsy counter while distracted
 
                    if self.distract_counter >= GAZE_FRAMES:
                        status = "DISTRACTED"
                        # FIX: cap counter so it doesn't grow unboundedly
                        self.distract_counter = GAZE_FRAMES
 
                elif ear < self.dynamic_ear_threshold:
                    self.counter += 1
                    self.distract_counter = 0  # FIX: reset distract counter while drowsy
                    if self.counter >= DROWSY_FRAMES:
                        status = "DROWSY"
                        self.counter = DROWSY_FRAMES  # cap counter
 
                elif mar > self.dynamic_mar_threshold:
                    status = "YAWNING"
                    self.counter = 0
                    self.distract_counter = 0
 
                else:
                    # Fully ALERT — reset all counters
                    self.counter = 0
                    self.distract_counter = 0
 
                # Alarm for all danger states
                if status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED"]:
                    play_alarm()
                else:
                    stop_alarm()
        else:
            stop_alarm()
 
        self.last_ear, self.last_mar, self.last_status = ear, mar, status
        return frame