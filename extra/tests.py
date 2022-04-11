from tools import fwhm_burst_norm
import numpy as np
import unittest




class FWHM_testing(unittest.TestCase):
    def setUp(self):
        self.peak_value = 10
        self.peak_loc = [50,50]
        self.edge_loc = [50,99]
        self.corner_loc = [99,99]
        self.square_loc = [40,60]
        self.empty = np.zeros((100,100))
        self.single = np.zeros((100,100))
        self.single[self.peak_loc] = self.peak_value
        self.square = np.zeros((100,100))
        self.square[
            self.square_loc[0]:self.square_loc[1], 
            self.square_loc[0]:self.square_loc[1]
        ] = self.peak_value
        self.edge = np.zeros((100,100))
        self.edge[self.edge_loc] = self.peak_value
        self.corner = np.zeros((100,100))
        self.edge[self.edge_loc] = self.peak_value

    def test_empty(self):
        self.assertEqual(
            fwhm_burst_norm(self.empty, (self.peak_loc[0], self.peak_loc[1])),
            (0, 0, 0, 0)
        )

    def test_single_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.single, (self.peak_loc[0], self.peak_loc[1])),
            (0, 0, 0, 0)
        )
    
    def test_square_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.square, (self.peak_loc[0], self.peak_loc[1])),
            (10, 10, 10, 10)
        )
    
    def test_edge_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.edge, (self.edge_loc[0], self.edge_loc[1])),
            (0, 0, 0, 0)
        )
    
    def test_corner_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.corner, (self.corner_loc[0], self.corner_loc[1])),
            (0, 0, 0, 0)
        )

if __name__ == "__main__":
    unittest.main()


# peak_value = 10
# peak_loc = [50,50]
# edge_loc = [50,99]
# corner_loc = [99,99]

# empty = np.zeros((100,100))
# single = np.zeros((100,100))
# single[peak_loc] = peak_value
# square = np.zeros((100,100))
# square[40:60,40:60] = peak_value
# edge = np.zeros((100,100))
# edge[edge_loc] = peak_value
# corner = np.zeros((100,100))
# edge[edge_loc] = peak_value

# print(fwhm_burst_norm(empty, (peak_loc[0], peak_loc[1])))
# print(fwhm_burst_norm(single, (peak_loc[0], peak_loc[1])))
# print(fwhm_burst_norm(square, (peak_loc[0], peak_loc[1])))
# print(fwhm_burst_norm(edge, (edge_loc[0], edge_loc[1])))
# print(fwhm_burst_norm(corner, (corner_loc[0], corner_loc[1])))
