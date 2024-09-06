import numpy as np
import matplotlib.pyplot as plt
a_mean=np.load('C:/Users/fp/Downloads/all_obs/controllers/res/SAC_env_nav_number_mean.npy')
a_mean_f = np.load('C:/Users/fp/Downloads/all_obs/controllers/res/SAC_env_nav_number_mean_-0.05.npy')
a = np.load('C:/Users/fp/Downloads/all_obs/controllers/res/2SAC_env_nav_number.npy')
plt.plot(a_mean,label  = 'mean')
# plt.plot(a_z,label = 'z')
plt.plot(a_mean_f,label = 'f')
plt.legend()
plt.show()

import sys
import os
import numpy as np



