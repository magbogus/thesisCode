import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np
import wave
import math

click_rec = wave.open('recordings\\BaOr_2.wav', 'r')

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
signalabs=abs(signal) #[:530000]) #120000])

i = 0
sum_x = 0

while i < len(signalabs)-300:
    sum_x = 0
    j = 0
    while j < 300:
        sum_x = sum_x + signalabs[i+j]
        j += 1
    
    signalabs[i] = sum_x/300
    i=i+1 

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

# Find peaks on the graph (besides echoes)
x_peaks, properties = find_peaks(signalabs, distance = 10000, prominence=(200, None))
properties["prominences"].max() 

y_peaks = signalabs[x_peaks]

# print("X values of peaks:", x_peaks)         # x values of peaks (skipping echo)
# print("Y values of peaks:", y_peaks)         # y values of peaks (skipping echo)

# Obtain 10% of the peaks' heights
y_range = y_peaks*0.1
# print("10% of peaks' height:", y_range)


starts = []
ends = []
a = 0
while a < len(y_range):
    # Set range on the left side of a peak where the window beggining will be found
    start_range = signalabs[x_peaks[a]-500:x_peaks[a]+1]

    # Find the closest value in array to y_range
    closest_start = closest(start_range, y_range[a])

    # Set range on the right side of a peak where the window ending will be found
    end_range = signalabs[x_peaks[a]:x_peaks[a]+1500]

    # Find the closest value in array to y_range
    closest_end = closest(end_range, y_range[a])

    # Get x values for y_range to obtain beggining of the window
    k = 0
    while k < len(start_range):
        if start_range[k] == closest_start:
            # To achieve the x value for the whole signal (not only for the cut piece):
            s_value = k + x_peaks[a]-500
            # print('Beggining of the window (x): ', s_value)
            starts.append(s_value)
            break
            # print("s:", starting)
        k+=1

    m = 0
    while m < len(end_range):
        if end_range[m] == closest_end:
            # To achieve the x value for the whole signal (not only for the cut piece):
            e_value = m + x_peaks[a]
            # print('Ending of the window (x): ', e_value)
            ends.append(e_value)
            break
            # print("s:", starting)
        m+=1

    a+=1

# print("Windows' begginings:", starts)
# print("Windows' begginings:", ends)


# # Find all peaks (echoes included)
# x_peak_echo, properties = find_peaks(signalabs, distance = 50, prominence=(800, None))
# properties["prominences"].max()

# y_peak_echo = signalabs[x_peak_echo]


# # Look for windows' ends - finding minimal value between sound and its echo
# ends = []
# d=0
# while d < len(x_peak_echo): 
#     # Define piece of signal between each sound and its echo
#     piece = signalabs[x_peak_echo[d]:x_peak_echo[d+1]]
#     # Find minimal value (y)
#     y_dip = min(piece)
#     l=0
#     # Find corresponding x value and add it to an array
#     while l < len(piece):
#         if piece[l] == y_dip:
#             x_d = l + x_peak_echo[d]
#             ends.append(x_d)
#             break
#         l+=1
#     d = d + 2

# print("Windows' ends:", ends)


# Find the greatest amplitude
max_amplitude = max(y_peaks)
# print("Maximal amplitude:", max_amplitude)

# method needed to round up received values
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# Show other peaks as percentage value regarding max_amplitude
t = 0
percentage_arr = [ ]
while t < len(y_peaks):
    per_value = round_half_up(y_peaks[t]/max_amplitude*100.0, 2)
    percentage_arr.append(per_value)
    t+=1

print("Percentage values of amplitudes:", percentage_arr)



w=0

allfreqs = []
allpowers = []
allpeaks = []
while w < len(x_peaks):
    sig = signal[starts[w]:ends[w]]
    fourier = fft(sig)
    fourier = fourier[:math.ceil(len(fourier)/2)]
    power = np.abs(fourier)**2

    sample_freq = fftfreq(sig.size, d=1/fs)
    pos_mask = (np.where(sample_freq >= 0)) #and (np.where(sample_freq <= 800 ))

    # s=0
    # pos_mask = []
    # while (sample_freq >= 0 and sample_freq <= 8000):
    #     pos_mask.append(sample_freq)
    #     s+=1

    # print('pos_mask length:', len(pos_mask))
    # print('pos_mask', pos_mask)
    # print('sample_freq length:', len(sample_freq))
    # print('sample_freq', sample_freq)

    # pos_mask = np.where((sample_freq >= 0) and (sample_freq <= 8000))
    # pos_mask = np.where(pos_mask <= 8000)
    
    freqs = sample_freq[pos_mask]/1000
    # print('freqs length:', len(freqs))
    # freqs = sample_freq/1000
    freqs = freqs[:math.ceil(len(freqs)/5)]
    print('freqs length:', len(freqs))
    print(freqs[len(freqs)-1])
    power = power[:len(freqs)]
    # Find peaks in the graph - frequencies in Hz
    peak_freq = round_half_up(freqs[power.argmax()], 2)
    # print('Frequency of', w+1, 'mouth click:', peak_freq, 'kHz')

    
    allpowers.append(power)
    allfreqs.append(freqs)
    allpeaks.append(peak_freq)
    
    w+=1

print('Frequencies [kHz] obtained in the FFT:', allpeaks)



# Plot the graph with signal energy and peaks
plt.figure(1)
plt.title('Signal Energy')
plt.xlabel('Number of sample')
plt.ylabel('Amplitude')
plt.plot(signalabs)
plt.plot(x_peaks, y_peaks, "x")
plt.plot(starts, signalabs[starts], "x")
plt.plot(ends, signalabs[ends], "x")
plt.savefig('output\\signalenergy.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot the graph with FFT
plt.figure(2)
plt.title('Fast Fourier Transform')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Power')
plt.plot(allfreqs[0], allpowers[0]) #
plt.plot(allfreqs[1], allpowers[1])
plt.plot(allfreqs[2], allpowers[2])
plt.plot(allfreqs[3], allpowers[3]) #
plt.plot(allfreqs[4], allpowers[4])
plt.plot(allfreqs[5], allpowers[5])
plt.plot(allfreqs[6], allpowers[6])
plt.plot(allfreqs[7], allpowers[7])
plt.plot(allfreqs[8], allpowers[8])
plt.plot(allfreqs[9], allpowers[9])
plt.plot(allfreqs[10], allpowers[10])
plt.plot(allfreqs[11], allpowers[11])
plt.plot(allfreqs[12], allpowers[12])
plt.plot(allfreqs[13], allpowers[13])
plt.plot(allfreqs[14], allpowers[14])
plt.plot(allfreqs[15], allpowers[15])
plt.plot(allfreqs[16], allpowers[16])
plt.plot(allfreqs[17], allpowers[17])
plt.plot(allfreqs[18], allpowers[18])
plt.plot(allfreqs[19], allpowers[19])
plt.plot(allfreqs[20], allpowers[20]) #
plt.plot(allfreqs[21], allpowers[21]) #
plt.plot(allfreqs[22], allpowers[22])
plt.plot(allfreqs[23], allpowers[23])
plt.plot(allfreqs[24], allpowers[24]) #
plt.plot(allfreqs[25], allpowers[25])
plt.plot(allfreqs[26], allpowers[26])
plt.plot(allfreqs[27], allpowers[27])
plt.plot(allfreqs[28], allpowers[28])
plt.plot(allfreqs[29], allpowers[29])

# plt.plot(x_freq, y_freq, "x")
plt.plot()
plt.savefig('output\\fourier.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot frequencies for each mouth click
plt.figure(3)
plt.title('Frequency per mouth click')
plt.xlabel('Number of mouth click')
plt.ylabel('Frequency [kHz]')
plt.plot(allpeaks)
plt.grid()
plt.savefig('output\\frequencies.png', dpi=300, bbox_inches='tight')
plt.show()

#_________________________________________________________________________________________________

# # Boxplots presenting amplitudes
# percentage_arr = [percentage_arr[:9], percentage_arr[10:19], percentage_arr[20:]]
# fig1, ax1 = plt.subplots()
# # plt.figure(4)
# ax1.set_title('Amplitudes divided into series')
# ax1.boxplot(percentage_arr)
# plt.savefig('output\\amplitude_boxplot.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Boxplots presenting frequencies
# allpeaks = [allpeaks[:9], allpeaks[10:19], allpeaks[20:]]
# fig2, ax2 = plt.subplots()
# # plt.figure(4)
# ax2.set_title('Frequencies divided into series')
# ax2.boxplot(allpeaks)
# plt.savefig('output\\freqs_boxplot.png', dpi=300, bbox_inches='tight')
# plt.show()

"# thesis" 
