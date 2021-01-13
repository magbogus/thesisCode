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
# plt.plot(abs(signal))
# plt.show()

# Calculate signal energy
signalabs=abs(signal[:500000]) #120000])

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

print("X values of peaks:", x_peaks)         # x values of peaks
print("Y values of peaks:", y_peaks)         # y values of peaks

# Obtain 10% of the peak values
y_range = y_peaks*0.1
print("10% of peaks' height:", y_range)


# Look for the value in array which is the closest to y_range

# start_range = signalabs[x_peaks[0]-500:x_peaks[0]+1]
# end_range = signalabs[x_peaks[0]:x_peaks[0]+501]
# print (start_range)
# print (x_peaks[0])

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
      
a = 0
while a < len(y_range):
    start_range = signalabs[x_peaks[a]-500:x_peaks[a]+1]
    closest_start = closest(start_range, y_range[a])

    end_range = signalabs[x_peaks[a]:x_peaks[a]+501]
    closest_end = closest(end_range, y_range[a])


    # Get x values for y_range to obtain beggining and ending of the window
    k = 0
    starting = [None]*30
    ending = [None]*30
    r = 0
    m = 0
    p = 0
    while k < len(start_range):
        if start_range[k] == closest_start:
            s_value = k + x_peaks[a]-500
            print('Beggining of the window (x): ', s_value)
            starting[r] = s_value
            r+=1
            break
        k+=1

    while m < len(end_range):
        if end_range[m] == closest_end:
            e_value = m + x_peaks[a]
            print('Ending of the window (x): ', e_value)
            ending[p] = e_value
            p+=1
            break
        m+=1
    
    a+=1

print(starting)
print(ending)


# closest_start = closest(start_range, y_range[0])
# closest_end = closest(end_range, y_range[0])

# print('closest start =', closest_start) 
# print('closest end =', closest_end) 


# # Get x values for y_range to obtain beggining of the window
# k = 0
# while k < len(signalabs):
#     if signalabs[k] == closest_start: #y_range[0]:
#         # print('Beggining of the window (x): ', k)
#         break
#     k+=1
   
# print('Beggining of the window (x):', k)



# Plot the graph with signal energy and peaks
plt.figure(1)
plt.title('Signal Energy')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
plt.plot(signalabs)
plt.plot(x_peaks, signalabs[x_peaks], "x")
# plt.plot(k, closest_start, "x")
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
