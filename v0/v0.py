
import scipy.io.wavfile
#import soundfile as sf
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft
import librosa.core as librosa
import librosa.display as librosa_display


# files                                                                         
src = "audio.mp3"
dst = "test.wav"

# convert wav to mp3                                                            
mp3 = AudioSegment.from_mp3(src)
mp3.export(dst, format="wav")

rate, audData = scipy.io.wavfile.read(dst)


#wav number of channels mono/stereo
#print(np.shape(audData))
# here = 2, so it is stereo

#if stereo grab both channels
channel1 = audData[:,0] #left
channel2 = audData[:,1] #right



#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate 

"""
#plot amplitude (or loudness) over time
plt.figure()
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.02, alpha=0.9, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, linewidth=0.02, alpha=0.9, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
"""

# FFT
fourier = fft.fft(channel1)
#print(fourier)

"""plt.figure()
plt.plot(fourier, alpha=0.9, color='blue')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.show()
"""

# CQT
# On charge le fichier wav avec librosa
x, sr = librosa.load("test.wav", sr=44100, mono=True) # mono=True transforme l'audio en mono (à faire)
cqt = librosa.cqt(x, sr=sr, bins_per_octave=36)
log_cqt = librosa.amplitude_to_db(np.abs(cqt))


# Spectrogram FFT
"""
plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('plasma'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('plasma'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()
"""


# Spectrogram CQT
"""
plt.figure(2, figsize=(12,6))
plt.subplot(211)
librosa_display.specshow(librosa.amplitude_to_db(cqt), sr=sr, x_axis='time', y_axis='hz', cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.title("CQT")
plt.subplot(212)
librosa_display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='hz', cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.title("log CQT")
plt.show()
"""


"""
print("########### channel1 ###########")
print(type(channel1))
print(np.shape(channel1))

print("########### Pxx ###########")
print(Pxx)
print(np.shape(Pxx))

print("########### freqs ###########")
print(freqs)

print("########### bins ###########")
print(bins)

print("########### im ###########")
print(im)
"""


print(fourier)
print(np.shape(fourier))
print(type(fourier))

print(channel1)
print(type(channel1))
print(np.shape(channel1))

print(cqt)
print(type(cqt))
print(np.shape(cqt))
print(cqt.ndim)
sss

# 2D Peaks (peaks in spectrogram)

# Import
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                          iterate_structure, binary_erosion)

# variables:
PEAK_NEIGHBORHOOD_SIZE = 20
axes = librosa_display.specshow(librosa.amplitude_to_db(cqt), sr=sr, x_axis='time', y_axis='hz', cmap='plasma')

struct = generate_binary_structure(2, 1)
neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
# find local maxima using our filter shape
local_max = maximum_filter(axes, footprint=neighborhood) == axes
