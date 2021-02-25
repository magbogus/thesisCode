import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# amplitudes in % collected from fourier.py
agpo_a = [29.3, 85.57, 96.93, 52.2, 54.36, 61.72, 29.12, 38.23, 22.21, 32.97, 66.76, 69.34, 43.27, 83.39, 65.04, 100.0, 42.56, 76.97, 85.23, 46.65, 8.47, 26.64, 19.15, 45.38, 15.67, 13.37, 35.74, 27.08, 43.92, 43.41]
ango_a = [70.17, 68.14, 64.87, 61.53, 66.03, 67.4, 44.77, 54.51, 65.66, 33.47, 60.52, 23.06, 32.84, 100.0, 31.94, 49.39, 67.1, 42.44, 39.92, 45.39, 24.08, 29.69, 10.71, 37.73, 76.76, 62.67, 57.8, 69.86, 52.69, 48.98]
anku_a = [63.15, 59.54, 84.0, 54.75, 100.0, 91.34, 87.22, 56.35, 58.45, 66.68, 43.71, 12.65, 84.29, 53.19, 43.27, 13.35, 89.15, 24.81, 72.29, 66.61, 20.68, 28.86, 17.28, 99.86, 10.91, 15.8, 21.02, 36.52, 7.25, 91.35]
bepa_a = [31.47, 28.46, 23.07, 12.88, 83.82, 62.87, 69.69, 51.97, 84.86, 84.23, 66.71, 27.57, 39.57, 62.39, 42.43, 42.16, 39.18, 54.45, 36.68, 52.74, 46.08, 38.19, 50.18, 44.08, 22.38, 20.85, 27.98, 100.0, 55.0, 99.02]
kabo_a = [100.0, 46.05, 47.59, 26.85, 34.48, 28.78, 33.43, 38.85, 38.3, 35.65, 24.92, 29.6, 19.09, 16.52, 18.7, 14.14, 18.01, 31.53, 19.26, 14.14, 47.65, 28.42, 44.93, 18.4, 30.61, 36.38, 26.82, 31.12, 41.18, 30.61]
kasz_a = [41.59, 53.0, 56.28, 50.1, 54.37, 45.78, 49.13, 50.98, 57.89, 51.57, 64.45, 100.0, 37.14, 71.18, 69.8, 44.94, 38.21, 21.72, 35.36, 82.04, 35.88, 42.27, 47.57, 50.3, 75.91, 44.56, 57.6, 41.62, 66.93, 35.88]
lera_a = [91.5, 61.88, 71.47, 100.0, 38.97, 93.4, 44.38, 66.77, 47.35, 89.29, 34.59, 25.84, 20.64, 34.2, 33.62, 34.59, 37.2, 25.84, 19.2, 33.62, 21.17, 18.92, 19.21, 21.93, 25.52, 24.14, 15.08, 13.84, 41.61, 21.93]
llga_a = [92.25, 100.0, 85.76, 77.47, 76.18, 75.73, 74.69, 73.24, 77.47, 82.61, 57.85, 37.52, 48.93, 50.47, 62.47, 33.22, 49.6, 42.49, 52.03, 52.14, 8.01, 35.21, 17.18, 82.64, 39.01, 39.64, 58.57, 71.98, 27.93, 36.18]
matu_a = [57.74, 48.91, 42.23, 39.54, 37.01, 59.09, 48.91, 65.23, 67.17, 38.57, 27.08, 50.15, 41.27, 36.04, 50.15, 60.02, 44.55, 27.08, 41.25, 100.0, 30.12, 70.05, 37.1, 33.89, 33.25, 31.91, 37.1, 34.61, 61.95, 56.81]
mika_a = [64.93, 100.0, 65.9, 39.13, 42.35, 43.7, 53.57, 38.45, 76.09, 60.17, 47.54, 36.76, 75.66, 44.79, 35.8, 61.86, 42.69, 34.45, 21.99, 23.22, 11.56, 51.06, 45.77, 17.27, 22.16, 14.09, 7.98, 14.21, 20.94, 34.67]
nagl_a = [39.73, 60.68, 60.03, 55.88, 81.92, 83.18, 78.89, 71.19, 70.93, 73.65, 64.21, 36.37, 30.84, 40.92, 42.25, 40.18, 46.2, 48.61, 57.24, 21.69, 78.36, 28.09, 82.51, 88.3, 32.61, 90.66, 74.99, 88.22, 100.0, 90.52]
nami_a = [69.28, 69.28, 100.0, 83.6, 69.28, 39.14, 60.63, 38.65, 41.92, 19.92, 59.39, 35.23, 53.87, 36.58, 22.27, 18.04, 35.23, 35.6, 46.85, 39.14, 51.33, 38.34, 27.79, 68.71, 28.73, 37.85, 16.32, 34.32, 34.17, 35.68]
norz_a = [56.45, 62.91, 45.24, 64.24, 91.95, 55.14, 42.31, 32.65, 57.44, 43.37, 32.37, 63.44, 65.16, 59.16, 66.09, 56.2, 40.36, 44.96, 57.08, 67.55, 100.0, 93.02, 93.08, 71.15, 78.17, 72.44, 43.77, 41.25, 93.66, 68.59]
mabo_a = [72.95, 76.35, 67.82, 89.21, 47.7, 66.56, 72.95, 71.54, 90.01, 74.22, 83.09, 51.09, 67.15, 68.64, 64.17, 66.55, 57.91, 85.81, 70.73, 44.46, 62.02, 88.79, 75.56, 93.36, 84.79, 76.07, 88.79, 100.0, 95.5, 84.79]
pamy_a = [56.94, 53.58, 31.01, 91.28, 74.99, 100.0, 40.71, 67.78, 43.78, 71.84, 76.39, 62.25, 89.5, 29.35, 91.73, 48.33, 82.81, 22.18, 50.53, 55.09, 22.23, 45.37, 51.19, 64.24, 31.64, 38.75, 21.74, 51.49, 39.95, 84.49]
ergd_a = [29.13, 28.84, 28.96, 28.09, 21.16, 93.41, 24.92, 79.48, 88.05, 22.13, 49.56, 73.76, 87.67, 100.0, 56.4, 63.17, 51.96, 46.27, 31.87, 41.7, 26.4, 20.5, 20.41, 27.4, 19.08, 22.12, 20.49, 26.4, 24.09, 15.08]
para_a = [100.0, 38.7, 68.78, 67.71, 59.22, 47.37, 15.11, 68.42, 57.38, 18.92, 95.41, 37.75, 31.33, 45.04, 37.8, 38.74, 31.33, 26.66, 36.9, 37.8, 30.26, 62.91, 52.03, 47.27, 60.81, 63.96, 69.29, 36.73, 85.15, 68.76]
elma_a = [100.0, 47.71, 26.19, 17.15, 28.11, 25.47, 33.38, 41.89, 21.6, 9.55, 95.88, 28.87, 12.42, 10.93, 51.39, 59.52, 52.94, 15.19, 19.52, 30.6, 13.9, 21.9, 26.17, 10.4, 20.22, 33.53, 13.9, 9.75, 21.9, 25.51]
xxyy_a = [41.25, 27.09, 100.0, 49.45, 66.28, 27.09, 41.25, 48.14, 49.45, 23.69, 48.37, 36.84, 47.6, 71.87, 35.57, 35.47, 47.6, 73.39, 44.41, 37.22, 21.99, 13.77, 43.79, 26.69, 21.41, 16.87, 23.4, 16.04, 22.09, 29.85]
powa_a = [9.14, 13.49, 47.94, 35.54, 69.67, 100.0, 17.53, 13.47, 24.93, 58.74, 13.12, 24.03, 31.8, 13.12, 29.01, 20.84, 48.08, 24.03, 33.69, 25.95, 19.43, 11.15, 26.71, 80.21, 9.64, 10.65, 15.32, 23.23, 7.57, 9.66]
olwo_a = [98.17, 100.0, 28.65, 79.68, 32.36, 45.89, 31.21, 31.15, 13.47, 15.39, 11.71, 14.17, 5.26, 8.74, 10.98, 7.29, 3.55, 20.95, 5.6, 20.95, 12.87, 22.81, 18.49, 26.26, 24.45, 13.86, 21.1, 16.1, 12.95, 12.79]
juka_a = [57.48, 71.2, 29.62, 58.15, 78.07, 84.37, 61.04, 67.19, 56.82, 31.4, 75.19, 44.86, 29.14, 19.19, 20.95, 43.24, 47.26, 31.43, 38.66, 37.82, 16.77, 28.14, 27.39, 32.06, 99.5, 100.0, 52.65, 69.62, 27.25, 28.92]
kapr_a = [94.82, 44.96, 48.49, 46.19, 51.36, 67.32, 54.11, 51.53, 34.49, 47.42, 80.07, 45.11, 47.55, 44.66, 43.44, 60.43, 87.6, 93.56, 100.0, 86.46, 33.98, 46.53, 46.38, 26.31, 25.82, 23.54, 47.42, 39.9, 30.64, 45.25]
ewbo_a = [34.51, 100.0, 55.1, 49.52, 33.03, 89.33, 34.72, 34.16, 66.96, 90.73, 29.65, 29.6, 22.23, 23.65, 37.25, 31.59, 15.87, 34.02, 25.4, 30.58, 29.12, 21.62, 25.95, 25.97, 35.36, 57.56, 36.27, 29.53, 29.48, 18.5]
jobo_a = [37.38, 33.37, 39.89, 49.82, 100.0, 35.56, 47.57, 34.88, 37.58, 27.59, 58.72, 23.26, 32.58, 20.65, 36.44, 18.3, 34.97, 39.41, 28.68, 49.82, 36.74, 36.57, 30.28, 14.74, 14.39, 16.74, 21.21, 34.14, 29.51, 28.41] 
joma_a = [94.3, 51.65, 85.88, 65.17, 76.35, 42.29, 38.78, 65.0, 100.0, 87.42, 35.26, 42.17, 41.72, 38.69, 45.15, 48.87, 40.18, 88.32, 28.08, 23.61, 5.5, 7.77, 8.98, 16.58, 14.18, 13.21, 11.45, 16.93, 18.81, 13.89]
mama_a = [34.25, 42.39, 37.2, 29.13, 52.3, 32.98, 33.73, 13.64, 23.21, 30.33, 74.46, 100.0, 82.67, 63.73, 37.46, 15.64, 29.84, 39.05, 52.78, 52.2, 29.55, 23.2, 26.5, 10.53, 31.0, 14.28, 25.21, 23.65, 13.35, 12.48]
mb_a = [29.82, 52.11, 100.0, 68.02, 25.07, 97.43, 53.98, 65.42, 34.57, 18.12, 8.22, 15.52, 27.61, 18.08, 34.95, 15.48, 26.57, 16.12, 20.02, 19.48, 15.55, 21.8, 17.22, 14.46, 16.15, 7.5, 16.48, 15.76, 12.54, 7.37]
pc_a = [69.11, 74.77, 48.74, 79.61, 40.55, 64.01, 43.48, 39.76, 70.1, 39.38, 51.08, 55.71, 62.99, 72.56, 28.73, 81.17, 27.73, 44.67, 61.09, 59.65, 8.03, 6.6, 100.0, 8.96, 12.93, 18.38, 10.21, 19.38, 20.82, 9.51]
hama_a = [100.0, 51.44, 92.22, 75.67, 25.16, 64.4, 89.41, 41.65, 75.0, 23.07, 30.8, 53.27, 23.74, 63.07, 30.8, 76.16, 43.01, 24.48, 30.07, 93.73, 15.45, 9.74, 9.38, 20.82, 14.72, 40.5, 28.45, 14.72, 9.38, 8.59]

# frequencies collected from fourier.py
agpo_f = [0.78, 0.75, 0.82, 0.83, 0.73, 0.66, 0.74, 0.75, 1.06, 1.01, 2.12, 1.96, 1.9, 1.97, 1.95, 1.98, 1.91, 1.99, 1.85, 1.74, 0.6, 0.68, 0.73, 0.73, 0.73, 0.72, 0.72, 0.83, 0.6, 0.69]
ango_f = [0.52, 1.49, 0.52, 0.52, 0.52, 0.52, 0.63, 1.42, 1.03, 1.2, 1.19, 1.97, 1.82, 1.2, 1.21, 1.72, 1.53, 2.17, 1.53, 1.74, 0.65, 0.65, 3.6, 0.64, 0.64, 0.64, 0.64, 0.64, 0.65, 0.53]
anku_f = [1.16, 1.15, 1.16, 1.33, 1.06, 1.15, 0.99, 1.06, 1.33, 1.16, 1.33, 1.15, 1.16, 1.23, 1.33, 1.54, 1.16, 1.33, 1.15, 1.15, 0.63, 0.63, 0.73, 0.66, 3.94, 0.71, 0.73, 0.8, 0.8, 0.78]
bepa_f = [1.12, 0.97, 1.06, 1.05, 1.06, 1.08, 1.03, 1.05, 1.07, 1.05, 1.67, 1.56, 1.34, 1.63, 1.68, 1.73, 1.34, 1.23, 1.71, 1.15, 0.63, 0.59, 0.63, 0.74, 0.53, 0.56, 0.64, 0.68, 0.75, 0.8]
kabo_f = [0.87, 1.02, 0.86, 0.82, 1.04, 0.83, 0.97, 1.01, 1.04, 0.84, 0.66, 0.84, 0.87, 0.92, 0.86, 0.82, 0.87, 0.82, 0.85, 0.82, 0.27, 0.27, 0.29, 0.22, 0.28, 0.25, 0.27, 0.29, 0.31, 0.28]
kasz_f = [1.38, 0.68, 1.19, 1.1, 1.19, 1.11, 1.11, 1.25, 1.28, 1.28, 0.95, 0.66, 0.67, 0.68, 1.11, 1.6, 0.66, 1.59, 1.6, 1.6, 0.68, 0.7, 0.78, 1.0, 1.0, 1.0, 1.12, 1.12, 0.93, 0.68]
lera_f = [1.11, 0.9, 1.02, 0.99, 1.17, 1.05, 0.87, 1.0, 1.04, 1.12, 0.87, 0.77, 1.13, 0.93, 0.88, 0.87, 0.86, 0.77, 1.58, 0.88, 0.75, 0.66, 0.71, 0.71, 0.7, 0.79, 0.63, 0.73, 0.68, 0.71]
llga_f = [0.82, 0.8, 0.83, 1.05, 0.81, 0.81, 0.99, 1.02, 1.02, 1.04, 0.74, 1.47, 1.35, 1.46, 0.79, 1.36, 0.79, 1.47, 1.45, 0.79, 0.55, 0.56, 0.55, 0.57, 0.57, 0.57, 0.67, 0.58, 0.57, 0.7]
matu_f = [1.21, 1.18, 1.22, 1.21, 1.33, 1.21, 1.18, 1.25, 1.26, 1.21, 2.08, 1.73, 1.74, 1.75, 1.73, 1.73, 1.74, 2.08, 1.74, 1.73, 0.66, 0.82, 0.75, 0.66, 0.77, 0.8, 0.75, 0.62, 0.77, 0.85]
mika_f = [1.43, 1.39, 1.38, 1.39, 1.68, 1.38, 0.69, 1.45, 1.39, 1.45, 1.82, 1.72, 1.82, 1.72, 1.72, 1.76, 1.75, 1.6, 1.44, 1.58, 0.88, 0.91, 1.43, 1.19, 0.73, 0.91, 0.91, 0.65, 1.02, 1.21]
nagl_f = [0.55, 1.08, 0.99, 0.89, 0.81, 0.88, 0.82, 0.78, 0.76, 0.8, 1.97, 1.96, 1.25, 0.87, 1.3, 2.02, 1.12, 1.24, 1.13, 2.16, 0.56, 0.54, 0.61, 0.68, 0.56, 0.7, 0.7, 0.69, 0.73, 0.79]
nami_f = [1.51, 1.51, 1.76, 1.68, 1.51, 1.07, 1.32, 1.36, 1.69, 1.69, 1.24, 1.32, 1.36, 1.27, 2.24, 2.13, 1.32, 1.19, 1.76, 1.81, 0.8, 0.89, 0.72, 0.8, 1.08, 0.82, 0.97, 0.82, 0.96, 0.87]
norz_f = [0.46, 0.8, 0.43, 1.0, 0.93, 0.41, 0.46, 0.97, 0.43, 0.46, 0.46, 0.42, 0.41, 0.43, 0.43, 0.46, 0.44, 0.45, 0.46, 0.45, 0.45, 0.44, 0.5, 0.54, 0.67, 0.51, 0.58, 0.55, 0.53, 0.55]
mabo_f = [1.26, 3.15, 1.24, 1.14, 1.24, 1.12, 1.26, 1.24, 1.11, 1.27, 3.13, 1.07, 1.26, 1.25, 1.23, 1.26, 1.25, 1.24, 1.27, 1.26, 4.27, 0.93, 0.95, 0.95, 0.98, 1.09, 0.93, 0.91, 0.92, 0.98]
pamy_f = [1.19, 1.11, 1.13, 1.08, 1.06, 1.11, 1.06, 0.98, 1.0, 1.03, 1.3, 1.31, 1.32, 1.37, 1.28, 1.28, 1.3, 1.38, 1.39, 1.21, 0.58, 0.56, 0.56, 0.6, 0.71, 0.58, 0.77, 0.69, 0.63, 0.65]
ergd_f = [1.12, 1.03, 0.93, 1.04, 0.87, 0.89, 1.05, 0.92, 0.89, 0.87, 1.31, 1.25, 1.18, 1.19, 1.28, 1.35, 1.36, 1.36, 1.95, 1.3, 0.89, 0.86, 0.79, 0.76, 0.74, 0.67, 0.7, 0.89, 0.7, 0.63]
para_f = [1.01, 1.61, 1.56, 1.52, 1.62, 1.46, 1.45, 1.63, 1.68, 1.65, 3.34, 2.05, 1.65, 2.12, 1.64, 1.62, 1.65, 1.55, 1.68, 1.64, 0.71, 1.07, 0.75, 0.73, 0.81, 0.83, 0.87, 0.89, 0.86, 0.9]
elma_f = [1.33, 1.41, 1.38, 1.34, 1.0, 1.27, 1.28, 1.28, 1.29, 1.18, 1.6, 1.49, 1.76, 1.75, 1.44, 1.5, 1.57, 1.38, 0.02, 1.37, 0.77, 0.79, 0.87, 0.69, 0.87, 0.86, 0.78, 0.81, 0.79, 0.88]
xxyy_f = [1.71, 1.6, 1.85, 1.68, 1.7, 1.6, 1.71, 1.66, 1.68, 1.72, 1.98, 1.94, 1.69, 1.69, 1.56, 1.64, 1.69, 1.74, 1.68, 1.71, 0.64, 0.78, 0.8, 0.76, 0.81, 0.81, 1.02, 0.78, 0.79, 0.82]
powa_f = [2.04, 1.32, 1.21, 1.15, 1.13, 1.0, 1.11, 1.12, 1.02, 0.98, 1.76, 1.38, 1.43, 1.76, 1.38, 0.05, 1.25, 1.38, 1.43, 1.37, 0.87, 1.0, 1.02, 1.02, 0.99, 0.9, 0.98, 1.02, 1.01, 1.01]
olwo_f = [0.97, 0.86, 0.69, 0.75, 0.71, 0.76, 0.8, 0.87, 0.78, 0.71, 1.74, 0.6, 1.56, 0.55, 1.61, 0.61, 1.86, 1.42, 1.75, 1.42, 0.61, 0.63, 0.67, 0.7, 0.69, 0.76, 0.91, 0.84, 0.68, 0.79]
juka_f = [1.27, 0.65, 1.75, 1.68, 1.25, 1.38, 1.53, 0.89, 1.11, 1.54, 0.56, 2.04, 1.45, 3.32, 1.76, 1.46, 0.87, 1.17, 2.04, 1.36, 0.55, 0.64, 0.73, 0.89, 0.89, 0.91, 1.03, 1.11, 1.06, 0.99]
kapr_f = [0.71, 0.66, 0.73, 0.72, 0.78, 1.0, 0.89, 2.89, 0.82, 0.9, 1.58, 1.58, 1.7, 1.59, 1.59, 1.56, 1.56, 1.57, 1.61, 1.6, 0.69, 0.62, 3.9, 0.76, 0.84, 0.93, 0.67, 0.64, 0.76, 0.85]
ewbo_f = [1.18, 1.08, 1.07, 1.17, 1.24, 1.08, 1.09, 1.09, 1.1, 1.18, 4.04, 1.87, 1.9, 3.21, 1.84, 1.86, 1.38, 1.83, 2.06, 1.88, 0.58, 0.54, 0.6, 0.61, 0.77, 0.77, 0.85, 0.83, 0.91, 0.9]
jobo_f = [1.1, 1.32, 1.11, 1.07, 1.31, 1.15, 1.07, 1.12, 1.07, 1.29, 1.09, 1.88, 1.87, 1.82, 1.84, 1.79, 1.85, 1.98, 1.84, 1.88, 0.68, 0.65, 0.71, 0.68, 0.74, 0.88, 0.83, 1.05, 1.0, 0.92]
joma_f = [1.44, 1.22, 1.21, 1.13, 1.14, 1.13, 1.07, 1.07, 1.14, 1.12, 1.72, 1.73, 1.92, 1.75, 2.05, 1.45, 1.43, 1.43, 1.73, 3.24, 3.19, 0.48, 0.44, 3.42, 0.47, 0.49, 0.46, 0.61, 3.48, 0.61]
mama_f = [3.61, 1.87, 1.73, 1.67, 1.63, 1.66, 1.66, 1.49, 1.68, 1.79, 1.97, 1.97, 1.97, 1.89, 1.78, 1.36, 1.87, 2.07, 1.98, 1.78, 0.68, 0.62, 0.73, 0.72, 0.69, 0.72, 0.86, 0.78, 0.86, 0.78]
mb_f = [1.11, 1.11, 1.47, 1.56, 1.21, 1.46, 1.58, 1.47, 1.69, 1.11, 1.81, 2.66, 1.86, 1.73, 1.54, 1.64, 1.73, 2.1, 1.75, 1.73, 0.95, 0.84, 1.1, 1.05, 0.85, 0.79, 0.91, 0.95, 0.95, 0.93]
pc_f = [1.63, 1.79, 1.51, 1.85, 1.72, 2.15, 1.87, 1.63, 1.74, 1.87, 2.46, 1.68, 2.4, 1.87, 1.68, 1.87, 2.39, 2.4, 1.85, 1.84, 0.6, 0.72, 0.82, 0.75, 0.92, 0.99, 0.82, 0.84, 0.83, 0.93]
hama_f = [1.34, 1.26, 1.27, 1.25, 1.59, 1.3, 1.26, 1.25, 1.25, 1.6, 1.49, 1.46, 1.61, 1.48, 1.51, 1.5, 1.49, 0.97, 1.49, 1.46, 0.52, 0.64, 3.4, 0.54, 0.65, 0.63, 0.58, 0.65, 3.4, 3.41]

# Amplitudes grouped regarding series
amp_loudest_array = [mb_a[:10], agpo_a[:10], ango_a[:10], anku_a[:10], bepa_a[:10], kabo_a[:10], kasz_a[:10], lera_a[:10], llga_a[:10], matu_a[:10], mika_a[:10], nagl_a[:10], nami_a[:10], norz_a[:10], pc_a[:10], mabo_a[:10], pamy_a[:10], ergd_a[:10], para_a[:10], elma_a[:10], hama_a[:10], xxyy_a[:10], powa_a[:10], olwo_a[:10], juka_a[:10], kapr_a[:10], ewbo_a[:10], jobo_a[:10], joma_a[:10], mama_a[:10]]
amp_highest_array = [mb_a[10:20], agpo_a[10:20], ango_a[10:20], anku_a[10:20], bepa_a[10:20], kabo_a[10:20], kasz_a[10:20], lera_a[10:20], llga_a[10:20], matu_a[10:20], mika_a[10:20], nagl_a[10:20], nami_a[10:20], norz_a[10:20], pc_a[10:20], mabo_a[10:20], pamy_a[10:20], ergd_a[10:20], para_a[10:20], elma_a[10:20], hama_a[10:20], xxyy_a[10:20], powa_a[10:20], olwo_a[10:20], juka_a[10:20], kapr_a[10:20], ewbo_a[10:20], jobo_a[10:20], joma_a[10:20], mama_a[10:20]]
amp_lowest_array = [mb_a[20:], agpo_a[20:], ango_a[20:], anku_a[20:], bepa_a[20:], kabo_a[20:], kasz_a[20:], lera_a[20:], llga_a[20:], matu_a[20:], mika_a[20:], nagl_a[20:], nami_a[20:], norz_a[20:], pc_a[20:], mabo_a[20:], pamy_a[20:], ergd_a[20:], para_a[20:], elma_a[20:], hama_a[20:], xxyy_a[20:], powa_a[20:], olwo_a[20:], juka_a[20:], kapr_a[20:], ewbo_a[20:], jobo_a[20:], joma_a[20:], mama_a[20:]]

# Frequencies grouped regarding series
freq_loudest_array = [mb_f[:10], agpo_f[:10], ango_f[:10], anku_f[:10], bepa_f[:10], kabo_f[:10], kasz_f[:10], lera_f[:10], llga_f[:10], matu_f[:10], mika_f[:10], nagl_f[:10], nami_f[:10], norz_f[:10], pc_f[:10], mabo_f[:10], pamy_f[:10], ergd_f[:10], para_f[:10], elma_f[:10], hama_f[:10], xxyy_f[:10], powa_f[:10], olwo_f[:10], juka_f[:10], kapr_f[:10], ewbo_f[:10], joma_f[:10], jobo_f[:10], mama_f[:10]]
freq_highest_array = [mb_f[10:20], agpo_f[10:20], ango_f[10:20], anku_f[10:20], bepa_f[10:20], kabo_f[10:20], kasz_f[10:20], lera_f[10:20], llga_f[10:20], matu_f[10:20], mika_f[10:20], nagl_f[10:20], nami_f[10:20], norz_f[10:20], pc_f[10:20], mabo_f[10:20], pamy_f[10:20], ergd_f[10:20], para_f[10:20], elma_f[10:20], hama_f[10:20], xxyy_f[10:20], powa_f[10:20], olwo_f[10:20], juka_f[10:20], kapr_f[10:20], ewbo_f[10:20], jobo_f[10:20], joma_f[10:20], mama_f[10:20]]
freq_lowest_array = [mb_f[20:], agpo_f[20:], ango_f[20:], anku_f[20:], bepa_f[20:], kabo_f[20:], kasz_f[20:], lera_f[20:], llga_f[20:], matu_f[20:], mika_f[20:], nagl_f[20:], nami_f[20:], norz_f[20:], pc_f[20:], mabo_f[20:], pamy_f[20:], ergd_f[20:], para_f[20:], elma_f[20:], hama_f[20:], xxyy_f[20:], powa_f[20:], olwo_f[20:], juka_f[20:], kapr_f[20:], ewbo_f[20:], jobo_f[20:], joma_f[20:], mama_f[20:]]

# Function needed to calculate mean and standard deviation added into one array for violin plots
def count_mean_std(init_array):
    j=0
    mean_std = []
    while (j < len(init_array)):
        mean = np.mean(init_array[j])
        std = np.std(init_array[j])
        std_plus = mean + std
        mean_std.append(mean)
        mean_std.append(std_plus)
        if (std > mean):
            pass
        else:
            std_minus = mean - std
            mean_std.append(std_minus)
        j+=1
    return (mean_std)

# Function needed to fix labels in violin plots,
# source: https://matplotlib.org/3.3.4/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


# Function needed to calculate mean and std to create a table
def count_mean_std_table(init_array):
    j=0
    mean_participants = []
    while (j < len(init_array)):
        # Calculate means for each participant separately and add into array
        means =  [' '.join([str(round(np.mean(init_array[j]), 2)), u"\u00B1", str(round(np.std(init_array[j]), 2))])]
        mean_participants.append([''.join(means)])
        j+=1
    # Calculate mean of all results of one series and add it as a last position in mean_participant[]
    general_mean =  [' '.join([str(round(np.mean(init_array), 2)), u"\u00B1", str(round(np.std(init_array), 2))])]
    mean_participants.append(general_mean)
    mean_std = np.concatenate(mean_participants)
    return (mean_std)


# Function needed to convert amplitudes from percent to decibels
def percent_to_db(init_array):
    q = 0
    b = []
    while (q < len(init_array)):
        counter = round(20*(np.log10(init_array[q]/100)), 2)
        b.append(counter)
        q = q + 1
    return b


# Calculate mean and standard deviation for all amplitude and frequency series
mean_std_aloud = count_mean_std(amp_loudest_array)
mean_std_ahigh = count_mean_std(amp_highest_array)
mean_std_alow = count_mean_std(amp_lowest_array)

mean_std_floud = count_mean_std(freq_loudest_array)
mean_std_fhigh = count_mean_std(freq_highest_array)
mean_std_flow = count_mean_std(freq_lowest_array)


# # # Violin plots presenting amplitudes
series = ['The loudest\nmouth clicks', 'The highest\nmouth clicks', 'The lowest\nmouth clicks']
db_arr = [percent_to_db(mean_std_aloud), percent_to_db(mean_std_ahigh), percent_to_db(mean_std_alow)]
fig1, ax1 = plt.subplots(figsize=(8,5))
# ax1.set_title('Amplitudes per series')
ax1.set_ylabel('Amplitude [dB]')
set_axis_style(ax1, series)
plt.grid(axis='y', alpha=0.7)
ax1.violinplot(db_arr, showmeans=True)
plt.savefig('output\\stats\\amplitudes_meanstd_db_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()


# Violin plots presenting frequencies
allpeaks = [mean_std_floud, mean_std_fhigh, mean_std_flow]
fig2, ax2 = plt.subplots(figsize=(8,5))
# ax2.set_title('Frequencies per series')
ax2.set_ylabel('Frequency [kHz]')
set_axis_style(ax2, series)
plt.grid(axis='y', alpha=0.7)
ax2.violinplot(allpeaks, showmeans=True)
plt.savefig('output\\stats\\freqs_meanstd_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()


# Scatter plot freq vs amp per series
fig3, ax3 = plt.subplots(figsize=(8,5))
# ax3.set_title('Frequency vs amplitude per series')
ax3.set_xlabel('Frequency [kHz]')
ax3.set_ylabel('Amplitude [%]')
ax3.scatter(freq_loudest_array, amp_loudest_array, c='#252627', s=60, edgecolors='#231123')
ax3.scatter(freq_highest_array, amp_highest_array, c='#FF6B35', s=60, edgecolors='#231123')
ax3.scatter(freq_lowest_array, amp_lowest_array, c='#2274A5', s=60, edgecolors='#231123') 
ax3.legend(series)
plt.savefig('output\\stats\\scatterplot_series.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot freq vs amp per people - the loudest motuh clicks
plt.figure(figsize=(8,5))
# plt.title('Frequency vs amplitude per participant for the loudest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Amplitude [%]')
plt.scatter(mb_f[:10], mb_a[:10], c ='#212529', s=60, edgecolors='#393E41')
plt.scatter(agpo_f[:10], agpo_a[:10], c ='#495057', s=60, edgecolors='#393E41')
plt.scatter(ango_f[:10], ango_a[:10], c ='#adb5bd', s=60, edgecolors='#393E41')
plt.scatter(anku_f[:10], anku_a[:10], c ='#8a5a44', s=60, edgecolors='#393E41')
plt.scatter(bepa_f[:10], bepa_a[:10], c ='#cd9777', s=60, edgecolors='#393E41')
plt.scatter(kabo_f[:10], kabo_a[:10], c ='#588157', s=60, edgecolors='#393E41')
plt.scatter(kasz_f[:10], kasz_a[:10], c ='#a3b18a', s=60, edgecolors='#393E41')
plt.scatter(lera_f[:10], lera_a[:10], c ='#ff477e', s=60, edgecolors='#393E41')
plt.scatter(llga_f[:10], llga_a[:10], c ='#ff99ac', s=60, edgecolors='#393E41')
plt.scatter(matu_f[:10], matu_a[:10], c ='#f9bec7', s=60, edgecolors='#393E41')
plt.scatter(mika_f[:10], mika_a[:10], c ='#a9d6e5', s=60, edgecolors='#393E41')
plt.scatter(nagl_f[:10], nagl_a[:10], c ='#61a5c2', s=60, edgecolors='#393E41') 
plt.scatter(nami_f[:10], nami_a[:10], c ='#014f86', s=60, edgecolors='#393E41')
plt.scatter(norz_f[:10], norz_a[:10], c ='#4cc9f0', s=60, edgecolors='#393E41')
plt.scatter(pc_f[:10], pc_a[:10], c ='#ba181b', s=60, edgecolors='#393E41')
plt.scatter(mabo_f[:10], mabo_a[:10], c ='#ffea00', s=60, edgecolors='#393E41')
plt.scatter(pamy_f[:10], pamy_a[:10], c ='#ffb700', s=60, edgecolors='#393E41')
plt.scatter(ergd_f[:10], ergd_a[:10], c ='#ff7700', s=60, edgecolors='#393E41')
plt.scatter(para_f[:10], para_a[:10], c ='#9ef01a', s=60, edgecolors='#393E41')
plt.scatter(elma_f[:10], elma_a[:10], c ='#38b000', s=60, edgecolors='#393E41')
plt.scatter(hama_f[:10], hama_a[:10], c ='#6247aa', s=60, edgecolors='#393E41')
plt.scatter(xxyy_f[:10], xxyy_a[:10], c ='#a06cd5', s=60, edgecolors='#393E41')
plt.scatter(powa_f[:10], powa_a[:10], c ='#ecfee8', s=60, edgecolors='#393E41')
plt.scatter(olwo_f[:10], olwo_a[:10], c ='#7c6a0a', s=60, edgecolors='#393E41')
plt.scatter(juka_f[:10], juka_a[:10], c ='#b7ce63', s=60, edgecolors='#393E41')
plt.scatter(kapr_f[:10], kapr_a[:10], c ='#41ead4', s=60, edgecolors='#393E41')
plt.scatter(ewbo_f[:10], ewbo_a[:10], c ='#FFFFC7', s=60, edgecolors='#393E41')
plt.scatter(jobo_f[:10], jobo_a[:10], c ='#D62828', s=60, edgecolors='#393E41')
plt.scatter(joma_f[:10], joma_a[:10], c ='#F8F4E3', s=60, edgecolors='#393E41')
plt.scatter(mama_f[:10], mama_a[:10], c ='#C3A29E', s=60, edgecolors='#393E41')
plt.savefig('output\\stats\\scatterplot_loud_participants.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot freq vs amp per people - the highest motuh clicks
plt.figure(figsize=(8,5))
# plt.title('Frequency vs amplitude per participant for the highest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Amplitude [%]')
plt.scatter(mb_f[10:20], mb_a[10:20], c ='#212529', s=60, edgecolors='#393E41')
plt.scatter(agpo_f[10:20], agpo_a[10:20], c ='#495057', s=60, edgecolors='#393E41')
plt.scatter(ango_f[10:20], ango_a[10:20],  c ='#adb5bd', s=60, edgecolors='#393E41')
plt.scatter(anku_f[10:20], anku_a[10:20],  c ='#8a5a44', s=60, edgecolors='#393E41')
plt.scatter(bepa_f[10:20], bepa_a[10:20], c ='#cd9777', s=60, edgecolors='#393E41')
plt.scatter(kabo_f[10:20], kabo_a[10:20], c ='#588157', s=60, edgecolors='#393E41')
plt.scatter(kasz_f[10:20], kasz_a[10:20], c ='#a3b18a', s=60, edgecolors='#393E41')
plt.scatter(lera_f[10:20], lera_a[10:20],  c ='#ff477e', s=60, edgecolors='#393E41')
plt.scatter(llga_f[10:20], llga_a[10:20], c ='#ff99ac', s=60, edgecolors='#393E41')
plt.scatter(matu_f[10:20], matu_a[10:20], c ='#f9bec7', s=60, edgecolors='#393E41')
plt.scatter(mika_f[10:20], mika_a[10:20],  c ='#a9d6e5', s=60, edgecolors='#393E41')
plt.scatter(nagl_f[10:20], nagl_a[10:20], c ='#61a5c2', s=60, edgecolors='#393E41') 
plt.scatter(nami_f[10:20], nami_a[10:20], c ='#014f86', s=60, edgecolors='#393E41')
plt.scatter(norz_f[10:20], norz_a[10:20], c ='#4cc9f0', s=60, edgecolors='#393E41')
plt.scatter(pc_f[10:20], pc_a[10:20], c ='#ba181b', s=60, edgecolors='#393E41')
plt.scatter(mabo_f[10:20], mabo_a[10:20], c ='#ffea00', s=60, edgecolors='#393E41')
plt.scatter(pamy_f[10:20], pamy_a[10:20], c ='#ffb700', s=60, edgecolors='#393E41')
plt.scatter(ergd_f[10:20], ergd_a[10:20],  c ='#ff7700', s=60, edgecolors='#393E41')
plt.scatter(para_f[10:20], para_a[10:20],  c ='#9ef01a', s=60, edgecolors='#393E41')
plt.scatter(elma_f[10:20], elma_a[10:20], c ='#38b000', s=60, edgecolors='#393E41')
plt.scatter(hama_f[10:20], hama_a[10:20], c ='#6247aa', s=60, edgecolors='#393E41')
plt.scatter(xxyy_f[10:20], xxyy_a[10:20], c ='#a06cd5', s=60, edgecolors='#393E41')
plt.scatter(powa_f[10:20], powa_a[10:20], c ='#ecfee8', s=60, edgecolors='#393E41')
plt.scatter(olwo_f[10:20], olwo_a[10:20], c ='#7c6a0a', s=60, edgecolors='#393E41')
plt.scatter(juka_f[10:20], juka_a[10:20], c ='#b7ce63', s=60, edgecolors='#393E41')
plt.scatter(kapr_f[10:20], kapr_a[10:20], c ='#41ead4', s=60, edgecolors='#393E41')
plt.scatter(ewbo_f[10:20], ewbo_a[10:20], c ='#FFFFC7', s=60, edgecolors='#393E41')
plt.scatter(jobo_f[10:20], jobo_a[10:20], c ='#D62828', s=60, edgecolors='#393E41')
plt.scatter(joma_f[10:20], joma_a[10:20], c ='#F8F4E3', s=60, edgecolors='#393E41')
plt.scatter(mama_f[10:20], mama_a[10:20], c ='#C3A29E', s=60, edgecolors='#393E41')
plt.savefig('output\\stats\\scatterplot_high_participants.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot freq vs amp per people - the lowest mouth clicks
plt.figure(figsize=(8,5))
# plt.title('Frequency vs amplitude per participant for the lowest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Amplitude [%]')
plt.scatter(mb_f[20:], mb_a[20:], c ='#212529', s=60, edgecolors='#393E41')
plt.scatter(agpo_f[20:], agpo_a[20:], c ='#495057', s=60, edgecolors='#393E41')
plt.scatter(ango_f[20:], ango_a[20:], c ='#adb5bd', s=60, edgecolors='#393E41')
plt.scatter(anku_f[20:], anku_a[20:],  c ='#8a5a44', s=60, edgecolors='#393E41')
plt.scatter(bepa_f[20:], bepa_a[20:], c ='#cd9777', s=60, edgecolors='#393E41')
plt.scatter(kabo_f[20:], kabo_a[20:],  c ='#588157', s=60, edgecolors='#393E41')
plt.scatter(kasz_f[20:], kasz_a[20:], c ='#a3b18a', s=60, edgecolors='#393E41')
plt.scatter(lera_f[20:], lera_a[20:],  c ='#ff477e', s=60, edgecolors='#393E41')
plt.scatter(llga_f[20:], llga_a[20:], c ='#ff99ac', s=60, edgecolors='#393E41')
plt.scatter(matu_f[20:], matu_a[20:], c ='#f9bec7', s=60, edgecolors='#393E41')
plt.scatter(mika_f[20:], mika_a[20:],  c ='#a9d6e5', s=60, edgecolors='#393E41')
plt.scatter(nagl_f[20:], nagl_a[20:], c ='#61a5c2', s=60, edgecolors='#393E41') 
plt.scatter(nami_f[20:], nami_a[20:], c ='#014f86', s=60, edgecolors='#393E41')
plt.scatter(norz_f[20:], norz_a[20:], c ='#4cc9f0', s=60, edgecolors='#393E41')
plt.scatter(pc_f[20:], pc_a[20:], c ='#ba181b', s=60, edgecolors='#393E41')
plt.scatter(mabo_f[20:], mabo_a[20:], c ='#ffea00', s=60, edgecolors='#393E41')
plt.scatter(pamy_f[20:], pamy_a[20:], c ='#ffb700', s=60, edgecolors='#393E41')
plt.scatter(ergd_f[20:], ergd_a[20:],  c ='#ff7700', s=60, edgecolors='#393E41')
plt.scatter(para_f[20:], para_a[20:],  c ='#9ef01a', s=60, edgecolors='#393E41')
plt.scatter(elma_f[20:], elma_a[20:], c ='#38b000', s=60, edgecolors='#393E41')
plt.scatter(hama_f[20:], hama_a[20:], c ='#6247aa', s=60, edgecolors='#393E41')
plt.scatter(xxyy_f[20:], xxyy_a[20:], c ='#a06cd5', s=60, edgecolors='#393E41')
plt.scatter(powa_f[20:], powa_a[20:], c ='#ecfee8', s=60, edgecolors='#393E41')
plt.scatter(olwo_f[20:], olwo_a[20:], c ='#7c6a0a', s=60, edgecolors='#393E41')
plt.scatter(juka_f[20:], juka_a[20:], c ='#b7ce63', s=60, edgecolors='#393E41')
plt.scatter(kapr_f[20:], kapr_a[20:], c ='#41ead4', s=60, edgecolors='#393E41')
plt.scatter(ewbo_f[20:], ewbo_a[20:], c ='#FFFFC7', s=60, edgecolors='#393E41')
plt.scatter(jobo_f[20:], jobo_a[20:], c ='#D62828', s=60, edgecolors='#393E41')
plt.scatter(joma_f[20:], joma_a[20:], c ='#F8F4E3', s=60, edgecolors='#393E41')
plt.scatter(mama_f[20:], mama_a[20:], c ='#C3A29E', s=60, edgecolors='#393E41')
plt.savefig('output\\stats\\scatterplot_low_participants.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram for the loudest
plt.figure(figsize=(8,5))
# plt.title('Histogram for the loudest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Frequency of presence')
plt.grid(axis='y', alpha=0.7)

i=0
bins = []
while i <= 4.1:
    bins.append(round(i, 1))
    i = i + 0.1

plt.hist(np.concatenate(freq_loudest_array), color='#252627', rwidth=0.85, bins=bins)
plt.legend(series)
plt.savefig('output\\stats\\hist_loudest.png', dpi=300, bbox_inches='tight')
plt.show()


# Histogram for the highest
plt.figure(figsize=(8,5))
# plt.title('Histogram for the highest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Frequency of presence')
plt.grid(axis='y', alpha=0.7)
plt.hist(np.concatenate(freq_highest_array), color='#FF6B35', rwidth=0.85, bins=bins)
plt.legend(series[1:])
plt.savefig('output\\stats\\hist_highest.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram for the lowest
plt.figure(figsize=(8,5))
# plt.title('Histogram for the lowest mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Frequency of presence')
plt.grid(axis='y', alpha=0.7)
plt.hist(np.concatenate(freq_lowest_array), color='#2274A5', rwidth=0.85, bins=bins)
plt.legend(series[2:])
plt.savefig('output\\stats\\hist_lowest.png', dpi=300, bbox_inches='tight')
plt.show()

i=0
bins = []
while i <= 2.6:
    bins.append(round(i, 1))
    i = i + 0.1

# Histogram for all series
plt.figure(figsize=(8,5))
# plt.title('Histogram for the all series of mouth clicks')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Density')
plt.grid(axis='y', alpha=0.7)

plt.hist([np.concatenate(freq_loudest_array),np.concatenate(freq_highest_array),np.concatenate(freq_lowest_array)], 
        color=['#252627', '#FF6B35', '#2274A5'], histtype='bar',  rwidth=0.85, bins=bins, density=True, alpha=0.7)
kde1 = stats.gaussian_kde([np.concatenate(freq_loudest_array)])
kde2 = stats.gaussian_kde([np.concatenate(freq_highest_array)])
kde3 = stats.gaussian_kde([np.concatenate(freq_lowest_array)])
x_kde = np.linspace(0, 2.5, 1000)
plt.plot(x_kde, kde1(x_kde), color='#252627')
plt.plot(x_kde, kde2(x_kde), color='#FF6B35')
plt.plot(x_kde, kde3(x_kde), color='#2274A5')
plt.plot(np.mean(freq_loudest_array), kde1(np.mean(freq_loudest_array)), 'd', color='#252627')
plt.plot(np.mean(freq_highest_array), kde2(np.mean(freq_highest_array)), 'd', color='#FF6B35')
plt.plot(np.mean(freq_lowest_array), kde3(np.mean(freq_lowest_array)), 'd', color='#2274A5')
plt.legend(series)
plt.savefig('output\\stats\\hist_all.png', dpi=300, bbox_inches='tight')
plt.show()


# Calculate means of frequency and amplitude (general mean and mean per participant) for each series
freq_loud_table = count_mean_std_table(freq_loudest_array)
freq_high_table = count_mean_std_table(freq_highest_array)
freq_low_table = count_mean_std_table(freq_lowest_array)

amp_loud_table = count_mean_std_table(amp_loudest_array)
amp_high_table = count_mean_std_table(amp_highest_array)
amp_low_table = count_mean_std_table(amp_lowest_array)

print('THE LOUDEST MOUTH CLICKS:\nFrequency [kHz]:', freq_loud_table, '\nAmplitude [%]:', amp_loud_table, '\nAmplitude [dB]:', percent_to_db(mean_std_aloud))
print('THE HIGHEST MOUTH CLICKS:\nFrequency [kHz]:', freq_high_table, '\nAmplitude [%]:', amp_high_table, '\nAmplitude [dB]:', percent_to_db(mean_std_ahigh))
print('THE LOWEST MOUTH CLICKS:\nFrequency [kHz]:', freq_loud_table, '\nAmplitude [%]:', amp_low_table, '\nAmplitude [dB]:', percent_to_db(mean_std_alow))


