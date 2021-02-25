import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np
import wave
import math

# Function needed to find value from the graph which is the closest to 10% of the peak's height,
# source: https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

# Method needed to round up received values,
# source: https://realpython.com/python-rounding/
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


click_rec = wave.open('recordings\\HaMa_2.wav', 'r')

#Extract Raw Audio from Wav File, source: 
signal = click_rec.readframes(-1)
signal = np.frombuffer(signal, dtype='int16')
fs = click_rec.getframerate()

# Calculate signal energy
signalabs=abs(signal)

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


# Find peaks on the graph on x axis,
# source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
x_peaks, properties = find_peaks(signalabs, distance = 10000, prominence=(700, None))
properties["prominences"].max() 

# Determine height of the found peak
y_peaks = signalabs[x_peaks]

# Obtain 10% of the peaks' heights
y_range = y_peaks*0.1


starts = []
ends = []
a = 0
while a < len(y_range):
    # Set range on the left side of a peak where the window beggining will be found
    start_range = signalabs[x_peaks[a]-1000:x_peaks[a]+1]

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
            s_value = k + x_peaks[a]-1000
            starts.append(s_value)
            break
        k+=1

    m = 0
    while m < len(end_range):
        if end_range[m] == closest_end:
            # To achieve the x value for the whole signal (not only for the cut piece):
            e_value = m + x_peaks[a]
            ends.append(e_value)
            break
        m+=1

    a+=1


# Find the greatest amplitude
max_amplitude = max(y_peaks)

# Show other peaks as percentage value regarding max_amplitude
t = 0
percentage_arr = [ ]
while t < len(y_peaks):
    per_value = round_half_up(y_peaks[t]/max_amplitude*100.0, 2)
    percentage_arr.append(per_value)
    t+=1

print("Percentage values of amplitudes:", percentage_arr)


# Calculate power and Fast Fourier Transform
# based on source: http://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
w=0
allfreqs = []
allpowers = []
allpeaks = []

while w < len(x_peaks):
    sig = signal[starts[w]:ends[w]]
    fourier = fft(sig)
    
    # Limit FFT to show only half values
    fourier = fourier[:math.ceil(len(fourier)/2)]
    power = np.abs(fourier)**2
    
    sample_freq = fftfreq(sig.size, d=1/fs)
    
    # Limit frequency only to positive values presented in kHz
    pos_mask = (np.where(sample_freq >= 0)) 
    freqs = sample_freq[pos_mask]/1000
  
    # Limit frequency up to about 4 kHz
    freqs = freqs[:math.ceil(len(freqs)/5)]
    power = power[:len(freqs)]
    
    # Find peaks in the graph - frequencies in kHz
    peak_freq = round_half_up(freqs[power.argmax()], 2)

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
plt.plot(allfreqs[0], allpowers[0]) 
plt.plot(allfreqs[1], allpowers[1])
plt.plot(allfreqs[2], allpowers[2])
plt.plot(allfreqs[3], allpowers[3]) 
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
plt.plot(allfreqs[20], allpowers[20]) 
plt.plot(allfreqs[21], allpowers[21]) 
plt.plot(allfreqs[22], allpowers[22])
plt.plot(allfreqs[23], allpowers[23])
plt.plot(allfreqs[24], allpowers[24]) 
plt.plot(allfreqs[25], allpowers[25])
plt.plot(allfreqs[26], allpowers[26])
plt.plot(allfreqs[27], allpowers[27])
plt.plot(allfreqs[28], allpowers[28])
plt.plot(allfreqs[29], allpowers[29])
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


"# thesis" 
