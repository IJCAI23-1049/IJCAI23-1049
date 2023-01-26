import time
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class plot_h36m(object):

    def __init__(self, prediction_data, GT_data):
        self.joint_xyz = GT_data
        self.nframes = GT_data.shape[0]
        self.joint_xyz_f = prediction_data

        # set up the axes
        xmin = -350
        xmax = 350
        ymin = -350
        ymax = 350
        zmin = -350
        zmax = 350

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.chain = [np.array([0, 1, 2, 3]),
                      np.array([0, 4, 5, 6]),
                      np.array([0, 7, 8, 9, 10]),
                      np.array([8, 11, 12, 13]),
                      np.array([8, 14, 15, 16])]

        # self.chain = [np.array([0, 1, 2, 3]),
        #               np.array([0, 4, 5, 6])
        #
        #               ]

        print(type(self.chain))
        print(len(self.chain))
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])
        # print ('zdata',type(zdata))
        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e'))
            # red: prediction #f94e3e' #008000 green
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.nframes, interval=300, repeat=False)
        plt.axis('off')
        # fig = plt.gcf()
        # fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0, 0)
        ani.save('/home/lv/MM/H3.6M/Smoking/ours/input.gif', writer='pillow')

        plt.show()

if __name__ == '__main__':
    config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
    use_node = np.array([0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22])
    # load GT_data
    base_path = '/home/lv/MM/H3.6M/motion_prediction_code/code/data'
    test_save_path = os.path.join(base_path, 'test.npy')
    GT_data = np.load(test_save_path)

    start = 401
    # start = 3043enhance_eachjoint_training.py
    # start = 1438
    # GT_data = GT_data[0]
    GT_data = GT_data[start:start+25, use_node, 4:]
    # GT_data = GT_data[start + 25:start + 50, use_node, 4:]
    # GT_data = GT_data[15:, :,:]

    # load prediction_data
    prediction_data_path = os.path.join(base_path, 'vis.npy')
    prediction_data = np.load(prediction_data_path)
    print('prediction_data:\n', prediction_data.shape)
    # print('prediction_data:\n', prediction_data[:, 1, :])
    # print('prediction_data:\n', GT_data[:, 1, :])

    nframes = GT_data.shape[0]
    prediction_data = prediction_data[:, :, :]
    prediction_data = prediction_data.reshape(-1, len(use_node), 3)
    GT_data = GT_data[:, :, :]

    predict_plot = plot_h36m(prediction_data, GT_data)
    predict_plot.plot()
