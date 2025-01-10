import torch
import numpy as np
import matplotlib.pyplot as plt

from colorsys import hls_to_rgb
from mpl_toolkits.mplot3d import Axes3D

from keypoint_def import JOINTS, EDGES


def get_palette(
    n: int, hue: float = 0.01, luminance: float = 0.6, saturation: float = 0.65
) -> np.array:
    hues = np.linspace(0, 1, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(int)
    palette = [hls_to_rgb(float(hue), luminance, saturation) for hue in hues]
    # palette = torch.tensor(palette)
    palette = np.array(palette)
    return palette


def getDimBox(points):
    """
    points: ... N x DIM
    output:	[[min_d1, max_d1], ..., [min_dN, max_dN]]
    """

    is_torch = torch.is_tensor(points[0])
    num_dim = points[0].shape[-1]
    if isinstance(points, list):
        assert is_torch or isinstance(points[0], np.ndarray)
        if is_torch:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min().detach().cpu() for pts in points]),
                        np.median([pts[..., k].max().detach().cpu() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min() for pts in points]),
                        np.median([pts[..., k].max() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
    else:
        if is_torch:
            points = points.cpu().detach()
        if isinstance(points, np.ndarray):
            return np.array(
                [
                    [
                        np.median(points[..., k].min(-1)[0]),
                        np.median(points[..., k].max(-1)[0]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            return np.array(
                [
                    [
                        points[..., k].min(-1)[0].median(),
                        points[..., k].max(-1)[0].median(),
                    ]
                    for k in range(num_dim)
                ]
            )


def plot_pose(
    poses: torch.Tensor,
    connections: list = None,
    azim: int = 0,
    elev: int = 0,
    save_fname: str = "",
    legend: list = [],
    dim_box=None,
    title: str = "",
    ax: Axes3D = None,
    is_blank: bool = False,
    s: int = 7,
    lw: int = 3,
    dpi: int = 300,
    show_axes: bool = True,
    legend_fontsize: int = 10,
    title_fontsize: int = 30,
    is_title_below: bool = False,
):
    """
    Plot 3D poses with connections.

    Parameters
    ----------
    poses : torch.Tensor
        BATCH x NUM_PTS x 3 tensor containing the poses.
        (NUM_PTS = 178)
    connections : list, optional
        BATCH x EDGES or EDGES list of connections between points.
    azim : int, default: 0
        Azimuth angle for the plot.
    elev : int, default: 0
        Elevation angle for the plot.
    save_fname : str, default: ""
        Filename to save the plot. If empty, the plot is not saved.
    legend : list, optional
        List of legend labels.
    dim_box : array-like, optional
        Dimension box for the plot.
    title : str, default: ""
        Title of the plot.
    ax : Axes3D, optional
        Axes object to plot on. If None, a new plot is created.
    is_blank : bool, default: False
        If True, creates a blank plot without axes.
    s : int, default: 7
        Size of the scatter points.
    lw : int, default: 3
        Line width for the connections.
    dpi : int, default: 300
        Dots per inch for the saved plot.
    show_axes : bool, default: True
        If True, shows the axes.
    legend_fontsize : int, default: 10
        Font size for the legend.
    title_fontsize : int, default: 30
        Font size for the title.
    is_title_below : bool, default: False
        If True, places the title below the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    # get colours
    colors_p = get_palette(n=poses.shape[-2] + 1)
    if isinstance(colors_p, np.ndarray) and (colors_p > 1).any():
        colors_p = [colors_p.copy() / 255]

    # create new plot
    if ax is None:
        plt.close("all")
        fig = plt.figure(figsize=(13.0, 20.0))
        # canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection="3d")
    else:
        if isinstance(ax, Axes3D):
            fig = ax.get_figure()
        else:
            fig = ax
            ax = fig.gca()

    # get bbox
    if dim_box is None:
        dim_box = getDimBox(poses)
    axes_line = [None for _ in range(len(poses))]

    for p, pose_ in enumerate(poses):
        s = [s for _ in range(len(pose_))]
        pose = pose_.cpu().detach().numpy() if torch.is_tensor(pose_) else pose_
        ax.scatter(
            pose[:, 0], pose[:, 1], pose[:, 2], color="black", marker="o", s=30
        )

        if connections is not None:
            connections_ = (
                connections[p] if isinstance(connections[0][0], tuple) else connections
            )
            for k, (x, y) in enumerate(connections_):
                # color = colors_l[p] if isinstance(colors_l, list) else colors_l[k]
                color = colors_p[k]
                # if k == black_edge:
                #     color = np.array((1.0, 1.0, 1.0))
                try:
                    axes_line[p] = ax.plot(
                        [pose[x, 0], pose[y, 0]],
                        [pose[x, 1], pose[y, 1]],
                        zs=[pose[x, 2], pose[y, 2]],
                        linewidth=lw,
                        color=color,
                    )[0]
                except:
                    print(f'wait, {k}')

    ax.set_xlim(dim_box[0])
    ax.set_ylim(dim_box[1])
    ax.set_zlim(dim_box[2])
    ax.set_box_aspect(
        (
            dim_box[0, 1] - dim_box[0, 0],
            dim_box[1, 1] - dim_box[1, 0],
            dim_box[2, 1] - dim_box[2, 0],
        )
    )
    if title is not None and title != "":
        if is_title_below:
            ax.set_title(title, fontsize=title_fontsize, y=-0.01)
        else:
            ax.set_title(title, fontsize=title_fontsize)

    ax.view_init(azim=azim, elev=elev)
    ax.legend(axes_line, legend, prop={"size": legend_fontsize})

    if is_blank:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        if show_axes:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.zaxis.set_ticks([])
        else:
            ax.axis("off")
        ax.grid(False)
    else:
        ax.set_xlabel("x", fontsize=25)
        ax.set_ylabel("y", fontsize=25)
        ax.set_zlabel("z", fontsize=25)

    if save_fname is not None and save_fname != "":
        plt.savefig(
            save_fname, bbox_inches="tight", pad_inches=0, dpi=dpi, transparent=True
        )
        print(f"Plot Saved: {save_fname}")

    return fig
