import math

def eye_aspect_ratio(eye):
    # Calculate euclidean distances using Python's ultra-fast built-in math module
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])

    # Safety check to prevent division by zero
    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear