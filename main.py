import torch
import matplotlib.pyplot as plt

from plot_pose import plot_pose
from keypoint_def import JOINTS, EDGES

"""
Run the following to plot a skeleton keypoint from the SLRTP Challenge.
"""

if __name__ == '__main__':
    # Load the skeleton keypoint
    pred = torch.load('./data/phix/dev.pt')

    # choose a frame
    key = list(pred.keys())[0]
    pose = pred[key]['poses_3d'][-1].unsqueeze(0)

    # plot the pose
    fig = plot_pose(pose, connections=EDGES, save_fname='./pose.png', azim=100, elev=90, is_blank=True, show_axes=False)
    fig.show()
