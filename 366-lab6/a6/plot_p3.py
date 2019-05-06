from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
filename = 'value.npy'
data = np.load(filename)
fig = plt.figure()
ax = Axes3D(fig)
steps = 50
x = np.arange(-1.2, 0.5, 1.7 / steps)
y = np.arange(-0.07, 0.07, 0.14 / steps)
x, y = np.meshgrid(x, y)

ax.set_xticks([-1.2, 0.5])
ax.set_yticks([-0.07, 0.07])
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.plot_surface(x, y, data,cmap='rainbow')

plt.show()