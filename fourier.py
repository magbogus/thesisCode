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

# print (fs)

# num = len(signal)
# Time=np.linspace(0, num/fs, num)
# print(num, Time)


# plt.figure(1)
# plt.title('Signal Wave of the tone')
# plt.xlabel('Number of sample')
# plt.ylabel('Amplitude')
# plt.plot(abs(signal))
# plt.show()

# Calculate signal energy
signalabs=abs(signal[:500000]) #120000])

i = 0
sum_x = 0

while i < len(signalabs)-200:
    sum_x = 0
    j = 0
    while j < 200:
        sum_x = sum_x + signalabs[i+j]
        j += 1
    
    signalabs[i] = sum_x/200
    i=i+1 

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

# Find peaks on the graph (besides echoes)
x_peaks, properties = find_peaks(signalabs, distance = 2000, prominence=(2000, None))
properties["prominences"].max() 

y_peaks = signalabs[x_peaks]

# print("X values of peaks:", x_peaks)         # x values of peaks (skipping echo)
# print("Y values of peaks:", y_peaks)         # y values of peaks (skipping echo)

# Obtain 10% of the peaks' heights
y_range = y_peaks*0.1
# print("10% of peaks' height:", y_range)


starts = [ ]
a = 0
while a < len(y_range):
    # Set range on the left side of a peak where the window beggining will be found
    start_range = signalabs[x_peaks[a]-500:x_peaks[a]+1]

    # Find the closest value in array to y_range
    closest_start = closest(start_range, y_range[a])

    # Get x values for y_range to obtain beggining of the window
    k = 0
    while k < len(start_range):
        if start_range[k] == closest_start:
            # To achieve the x value for the whole signal (not only for the cut piece):
            s_value = k + x_peaks[a]-500
            # print('Beggining of the window (x): ', s_value)
            starts.append(s_value.copy())
            # print("s:", starting)
        k+=1 
    a+=1

print("Windows' begginings:", starts)


# Find all peaks (echoes included)
x_peak_echo, properties = find_peaks(signalabs, prominence=(1500, None))
properties["prominences"].max()

y_peak_echo = signalabs[x_peak_echo]


# Look for windows' ends - finding minimal value between sound and its echo
ends = []
d=0
while d < len(x_peak_echo): 
    # Define piece of signal between each sound and its echo
    piece = signalabs[x_peak_echo[d]:x_peak_echo[d+1]]
    # Find minimal value (y)
    y_dip = min(piece)
    l=0
    # Find corresponding x value and add it to an array
    while l < len(piece):
        if piece[l] == y_dip:
            x_d = l + x_peak_echo[d]
            ends.append(x_d)
        l+=1
    d = d + 2

print("Windows' ends:", ends)


# # Find the greatest amplitude
# max_amplitude = max(y_peaks)
# # print("Maximal amplitude:", max_amplitude)

# method needed to round up received values
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# # Show other peaks as percentage value regarding max_amplitude
# t = 0
# percentage_arr = [ ]
# while t < len(y_peaks):
#     per_value = round_half_up(y_peaks[t]/max_amplitude*100.0, 2)
#     percentage_arr.append(per_value)
#     t+=1

# print("Percentage values of amplitudes:", percentage_arr)


# # while w < len(x_peaks):
# #     fft(signalabs[s_list[w]:e_list[w]])


sig = signal[starts[0]:ends[0]]
fourier = fft(sig)

# n = sig.size
# freq = np.fft.fftfreq(n, 1/fs)


# Plot the graph with signal energy and peaks
plt.figure(1)
plt.title('Signal Energy')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
plt.plot(signalabs)
plt.plot(x_peak_echo, y_peak_echo, "x")
plt.plot(starts, signalabs[starts], "x")
plt.plot(ends, signalabs[ends], "x")
plt.show()


# Plot the graph with FFT
plt.figure(2)
plt.title('FFT')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
plt.plot(fourier)
plt.plot()
plt.show()




# ___________________________________________________________________________________
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
