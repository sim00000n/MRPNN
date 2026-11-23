import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("QtAgg")

print(matplotlib.get_backend())

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
plt.show(block=False)

y = []
for i in range(200):
    y.append(np.exp(-i/50) + 0.01)
    x = np.arange(len(y))
    line.set_data(x, y)
    ax.set_xlim(0, max(10, len(x)))
    ax.set_ylim(1e-4, 2)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.05)