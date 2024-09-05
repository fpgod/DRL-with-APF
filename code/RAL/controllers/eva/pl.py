import matplotlib.pyplot as plt
import numpy as np
a = np.load("D:/fp/nav/10/col/all_obs/controllers/apf_sac/models/SAC_near_r_nav_number_10.npy")
plt.plot(a[0:100])
plt.show()