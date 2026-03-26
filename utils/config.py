EAR_THRESHOLD = 0.20
EAR_SEMI_THRESHOLD = 0.27

MAR_THRESHOLD = 0.60

DROWSY_FRAMES = 20

ALARM_SOUND = "assets/sounds/alarm.wav"
# --- CALIBRATION SETTINGS ---
# Number of frames to analyze to calculate the user's normal baseline
CALIBRATION_FRAMES = 60 

# Multipliers for dynamic thresholds
EAR_DROP_PERCENTAGE = 0.75  # Alarm triggers if eyes close by 25% of baseline
MAR_RISE_PERCENTAGE = 1.5   # Alarm triggers if mouth opens 50% wider than baseline
# --- HEAD DROP SETTINGS ---
# Threshold for vertical head compression (Pitch)
# If (Nose-to-Chin distance / Face-Width) drops below this, head is down.
HEAD_DROP_THRESHOLD = 0.35
# --- GAZE / DISTRACTION SETTINGS ---
# Deviation threshold (How far from center the pupil moves)
# 0.5 is center, < 0.35 is looking left/up, > 0.65 is looking right/down
GAZE_THRESH_L = 0.40
GAZE_THRESH_R = 0.60
GAZE_FRAMES = 30 # Number of frames to trigger "DISTRACTED"
# --- SPECS OPTIMIZATION ---
# Number of frames to average for smoothing (Prevents flicker from glasses glare)
SMOOTHING_WINDOW = 3