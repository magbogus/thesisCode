import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks
import numpy as np
import wave
import math

click_rec = wave.open('recordings\\MaBo_2.wav', 'r')

#Extract Raw Audio from Wav File
signal = click_rec.readframes(-1)
signal = np.frombuffer(signal, dtype='int16')
fs = click_rec.getframerate()


# num = len(signal)
# Time=np.linspace(0, num/fs, num)


plt.figure(1)
plt.title('Signal Wave of the tone')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
#plt.plot(abs(signal))
# plt.show()

# Calculate energy of the signal:  https://stackoverflow.com/questions/29429733/cant-find-the-right-energy-using-scipy-signal-welch
signalabs=abs(signal[15000:120000])

i = 0
sum_x = 0

while i < len(signalabs)-100:
    sum_x = 0
    j = 0
    while j < 100:
        sum_x = sum_x + signalabs[i+j]
        j += 1
    
    signalabs[i] = sum_x/100
    i=i+1 


# Find peaks on the graph
x_peaks, properties = find_peaks(signalabs, distance = 2000, prominence=(2000, None))
properties["prominences"].max()

y_peaks = signalabs[x_peaks]

print(x_peaks)         # x values of peaks
print(y_peaks)         # y values of peaks

# Obtain 30% of the peak values
y_start = y_peaks*0.3
print(y_start)


# Get x values for y_start to obtain beggining of the window
k = 0
while k < len(signalabs):
    if signalabs[k] == y_start[0]:
        print(k)
    k+=1
   




# Plot the graph with signal energy and peaks
plt.figure(1)
plt.title('Energy')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
plt.plot(signalabs)
plt.plot(x_peaks, signalabs[x_peaks], "x")
plt.plot()
plt.show()





# #find peak
# signal_values = [abs(x.real) for x in signalabs]
# x_start = 0 
# iterator_y = 0 
# for x in signal_values: 
#     y = signal_values[x] 
#     if y > iterator_y: 
#         x_start = abs(x)
#         iterator_y = y
# print(x_start)



#FFT

# fourier_values = [abs(x.real) for x in fft(signal)]
# x_max = 0 
# iterator_y = 0 
# for x in signal: 
#     y = fourier_values[x] 
#     if y > iterator_y: 
#         x_max = abs(x)
#         iterator_y = y
# print ('Frequency obtained in the Fast Fourier Transform is equal to ', x_max, 'Hz')
# plt.figure(2)
# plt.title('Fast Fourier Transform')
# plt.plot(fft(signal))
# plt.show()


"# thesis" 
