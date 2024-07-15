import os
import sys
import threading
import time
import numpy as np
import sounddevice as sd

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

class Alarm():
    def __init__(self):
        self.sound_lock = threading.Lock()
    
    def fire_alarm(self):
        print("Alarm: Safety Violation detected \n")

    def fire_sound_alarm(self):
        if self.sound_lock.locked() == True:
            return
        else:
            self.sound_lock.acquire()
            frequency = 1000  # Set the frequency in Hz
            duration = 1.5  # Set the duration in seconds (1 second in this example)
            sample_rate = 44100  # Set the sample rate

            t = 1.0 / sample_rate  # Time between samples
            samples = int(sample_rate * duration)
            time_x = np.arange(0, duration, t)
            waveform = 0.5 * (1 + np.cos(2 * np.pi * frequency * time_x))
            sd.play(waveform, sample_rate)
            sd.wait()
            time.sleep(0.5)
            self.sound_lock.release()


if __name__ == '__main__':

    frequency = 1000  # Set the frequency in Hz
    duration = 1.5  # Set the duration in seconds (1 second in this example)
    sample_rate = 44100  # Set the sample rate

    t = 1.0 / sample_rate  # Time between samples
    samples = int(sample_rate * duration)
    time_x = np.arange(0, duration, t)
    waveform = 0.5 * (1 + np.cos(2 * np.pi * frequency * time_x))

    # Save the waveform to a WAV file
    filename = 'alarm_sound.wav'
    wav.write(filename, sample_rate, waveform)

    # Optionally, you can play the sound as well
    sd.play(waveform, sample_rate)
    sd.wait()
