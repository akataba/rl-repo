import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class RealTimeScatterPlot:
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.suptitle('Real-Time Scatter Plot')
        self.axs[0].set_title('Fidelity')
        self.axs[1].set_title('1 - Fidelity (log scale)')
        self.axs[2].set_title('Reward')

        self.scatter1 = self.axs[0].scatter([], [], c='blue')
        self.scatter2 = self.axs[1].scatter([], [], c='red')
        self.scatter3 = self.axs[2].scatter([], [], c='green')

        for ax in self.axs:
            ax.set_xlim(0, 1)

        # self.axs[1].set_yscale('log')

    def update_data(self, fidelity, reward):
        x = np.arange(len(fidelity))

        self.scatter1.set_offsets(np.c_[x, fidelity])
        self.scatter2.set_offsets(np.c_[x, [1 - ii for ii in fidelity]])
        self.scatter3.set_offsets(np.c_[x, reward])

    def animate(self, i):
        self.update_data(self.fidelity, self.reward)

    def plot(self, fidelity, reward):
        self.fidelity = fidelity
        self.reward = reward

        self.fig.show()
        self.fig.canvas.draw()

        def update_plot():
            while True:
                self.update_data(self.fidelity, self.reward)
                self.fig.canvas.draw()
                plt.pause(0.2)

        def close_figure(event):
            if event.key == 'escape':
                plt.close()

        self.fig.canvas.mpl_connect('key_press_event', close_figure)

        update_plot()