from tools import fwhm_burst_norm
import numpy as np

peak_value = 10
peak_loc = [50,50]


empty = np.zeros((100,100))
single = np.zeros((100,100))
single[50,50] = peak_value
square = np.zeros((100,100))
square[40:60,40:60] = peak_value


print(fwhm_burst_norm(empty, (peak_loc[0], peak_loc[1])))
print(fwhm_burst_norm(single, (peak_loc[0], peak_loc[1])))
print(fwhm_burst_norm(square, (peak_loc[0], peak_loc[1])))
print(fwhm_burst_norm(square, (0, 0)))


    