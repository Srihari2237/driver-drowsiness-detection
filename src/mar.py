import math

def mouth_aspect_ratio(mouth):
    # Calculate euclidean distances
    A = math.dist(mouth[1], mouth[7])
    B = math.dist(mouth[2], mouth[6])
    C = math.dist(mouth[3], mouth[5])
    D = math.dist(mouth[0], mouth[4])

    # Safety check to prevent division by zero
    if D == 0:
        return 0.0

    mar = (A + B + C) / (2.0 * D)
    return mar