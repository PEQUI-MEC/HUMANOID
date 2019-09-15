import math
import numpy as np

def sigmoid_deslocada(x, periodo):
	return 1. / (1. + math.exp((-12. / periodo) * (x - (periodo / 2.))))

def rad_to_deg(angles):
  return np.array(angles) * 1800 / np.pi
