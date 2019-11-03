import urllib
import scipy.io.wavfile
import soundfile as sf
from os import path
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft

# files                                                                         
src = "audio.mp3"
dst = "test.wav"

# convert wav to mp3                                                            
mp3 = AudioSegment.from_mp3(src)
mp3.export(dst, format="wav")

rate, audData = scipy.io.wavfile.read(dst)


#wav number of channels mono/stereo
print(np.shape(audData))
# here = 2, so it is stereo

#if stereo grab both channels
channel1 = audData[:,0] #left
channel2 = audData[:,1] #right



#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate 

#plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.02, alpha=0.9, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, linewidth=0.02, alpha=0.9, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.show()


# FFT
fourier = fft.fft(channel1)
print(fourier)

plt.figure()
plt.plot(fourier, alpha=0.9, color='blue')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.show()



# Spectrogram
plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()
