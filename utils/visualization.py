import cv2


def draw_eye_landmarks(frame, eye):

    for (x, y) in eye:
        cv2.circle(frame, (x, y), 2, (0,255,0), -1)


def draw_mouth_landmarks(frame, mouth):

    for (x, y) in mouth:
        cv2.circle(frame, (x, y), 2, (255,0,0), -1)


def display_metrics(frame, ear, mar, status):

    cv2.putText(frame,
                f"EAR: {ear:.2f}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"MAR: {mar:.2f}",
                (30,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"STATUS: {status}",
                (30,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)