import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram
from moviepy import VideoFileClip

# Bandpass filter setup
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

def generate_wav(video_file, audio_file):
    if not os.path.exists(audio_file):
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_file)
        audio.close()
        video.close()

def plot_specgram(data, beep_interval=[]):
    plt.figure(figsize=(12, 6))
    plt.specgram(data, Fs=fs, NFFT=2048, noverlap=1024, cmap="viridis")
    plt.title("Audio Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")

    # Highlight specific frequency ranges
    plt.axhline(500, color="red", linestyle="--", label="500 Hz")
    plt.axhline(1000, color="blue", linestyle="--", label="1000 Hz")

    for start, end in beep_interval:
        plt.axvspan(start, end, color="purple", alpha=0.3, label="Highlighted Interval")
    
    plt.ylim(0, 2000)  # Focus on frequencies up to 2000 Hz
    plt.show()

def find_beeps(data, fs):
    # Create the spectrogram
    f, t, Sxx = spectrogram(data, fs, nperseg=1024)

    # Find the index of the 1000Hz frequency band
    freq_index = np.argmin(np.abs(f - 1010))

    # Extract the power at 1000Hz over time
    power_1000Hz = Sxx[freq_index, :]

    # Normalize power for easier thresholding
    power_1000Hz = power_1000Hz / np.max(power_1000Hz)

    # Define a threshold for detecting the beep (adjust as needed)
    threshold = 0.01

    # Find regions where power exceeds the threshold
    above_threshold = power_1000Hz > threshold

    # Detect start and end indices
    starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
    ends = np.where(np.diff(above_threshold.astype(int)) == -1)[0]

    # If the last beep doesn't end before the recording ends
    if len(ends) < len(starts):
        ends = np.append(ends, len(power_1000Hz) - 1)

    # Convert indices to times
    start_times = t[starts]
    end_times = t[ends]

    # Print results
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        print(f"Beep {i+1}: Start = {start:.3f}s, End = {end:.3f}s")

    # Optional: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, power_1000Hz, label="1000Hz Power")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    
    intervals = list(zip(start_times, end_times))
    intervals = [i for i in intervals if i[0] > 5]
    while True:
        intervals_length = [v[1]-v[0] for v in intervals]
        min_interval_length = min(intervals_length)
        if min_interval_length > 0.9:
            break
        removed_index = intervals_length.index(min_interval_length)
        if removed_index == 0:
            merged_index = 1
        elif removed_index == len(intervals) - 1:
            merged_index = removed_index - 1
        else:
            distance_next = intervals[removed_index+1][1] - intervals[removed_index][0]
            distance_prev = intervals[removed_index][1] - intervals[removed_index-1][0]
            if distance_next < distance_prev:
                merged_index = removed_index + 1
            else:
                merged_index = removed_index - 1
        ir = intervals[removed_index]
        im = intervals[merged_index]
        intervals[merged_index] = (min(ir[0], im[0]), max(ir[1], im[1]))
        intervals.pop(removed_index)
        
        
    for start, end in intervals:
        plt.axvspan(start, end, color='green', alpha=0.3, label="Detected Beep")
    plt.title("1000Hz Beep Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Power")
    # plt.legend()
    # plt.show()
    return intervals

def detect_beeps():
    video_file = "data\\28.11.24\\robot_footage.mp4"
    audio_file = "audio.wav"

    generate_wav(video_file, audio_file)

    fs, data = wavfile.read(audio_file)  # fs: sampling rate, data: audio signal
    if data.ndim > 1:  # If stereo, convert to mono
        data = data.mean(axis=1)

    beeps = find_beeps(data, fs)

    return beeps

#plot_specgram(data, beeps)