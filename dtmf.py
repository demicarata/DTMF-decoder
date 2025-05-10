import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

dtmf_frequencies_low = [697, 770, 852, 941]
dtmf_frequencies_high = [1209, 1336, 1477, 1633]
dtmf_digits = {
    (low, high): digit 
    for low, digit in zip(dtmf_frequencies_low, ['1', '2', '3', 'A'])
    for high, digit in zip(dtmf_frequencies_high, ['1', '4', '7', '*', '2', '5', '8', '0', '3', '6', '9', '#', 'A', 'B', 'C', 'D'])
}

audio_path = '911_sound.mp3'
audio_file, sample_rate = librosa.load(audio_path, sr=None)

D = librosa.stft(audio_file)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(9, 8))
librosa.display.specshow(D_db, x_axis='time' ,y_axis='log', sr=sample_rate)
plt.colorbar(format="%+2.0f dB")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
# plt.show()

def get_dtmf_tones(stft, frequencies, threshold=0.5):
    detected_digits = []

    for i in range(stft.shape[1]):
        frame = np.abs(stft[:, i])
        
        detected_low = None
        detected_high = None

        # Find low frequency
        for low in dtmf_frequencies_low:
            low_freq_bin = np.argmin(np.abs(frequencies - low))
            if (frame[low_freq_bin] > threshold):
                detected_low = low
                break
        
        # Find high frequency
        for high in dtmf_frequencies_high:
            high_freq_bin = np.argmin(np.abs(frequencies - high))
            if (frame[high_freq_bin] > threshold):
                detected_high = high
                break

        if (detected_low and detected_high):
            detected_digits.append(dtmf_digits.get((detected_low, detected_high), "Unknown"))

    return detected_digits

frequencies = librosa.fft_frequencies(sr=sample_rate)
detected_digits = get_dtmf_tones(D, frequencies)

print("Detected DTMF digits:", detected_digits)
