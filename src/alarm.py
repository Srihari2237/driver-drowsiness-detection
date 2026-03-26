import pygame
import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import ALARM_SOUND

# Initialize the pygame mixer for background audio
pygame.mixer.init()
alarm_sound = None
alarm_active = False

def play_alarm():
    global alarm_active, alarm_sound
    
    if not alarm_active:
        # Load the sound only once
        if alarm_sound is None:
            if os.path.exists(ALARM_SOUND):
                alarm_sound = pygame.mixer.Sound(ALARM_SOUND)
            else:
                print(f"⚠️ Error: Sound file not found at {ALARM_SOUND}")
                return
        
        # Play asynchronously on a loop (-1 means infinite loop)
        alarm_sound.play(loops=-1)
        alarm_active = True


def stop_alarm():
    global alarm_active, alarm_sound
    
    if alarm_active and alarm_sound:
        alarm_sound.stop()
        alarm_active = False