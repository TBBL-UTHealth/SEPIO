import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modules.disc_plotter as dpt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math

class ElectrodePlotter:
    """
    Class ElectrodePlotter

    Description: a helper class that plots DISC in various configurations and views.
    Currently, electrode positions are hard-coded to the values from ANSYS.
    In the future, this should be dynamically created from ANSYS metadata. 

    The 3D (isometric) view has a configurable angle, and the 2D viewer simply plots
    the DISC or ECOG electrodes as if they were rolled flat.
    """
    def __init__(self, electrode_ids, num_electrodes, dipoles=[]):
        self.electrode_ids = electrode_ids
        self.num_electrodes = num_electrodes
        self.dipoles = dipoles
        self.elec_pos = np.array([
            [0.2863782464, 0.2863782464, 2],
            [0.2863782464, 0.2863782464, 2.5],
            [0.2863782464, 0.2863782464, 3],
            [0.2863782464, 0.2863782464, 3.5],
            [0.2863782464, 0.2863782464, 4],
            [0.2863782464, 0.2863782464, 4.5],
            [0.2863782464, 0.2863782464, 5],
            [0.2863782464, 0.2863782464, 5.5],
            [0.2863782464, 0.2863782464, 6],
            [0.2863782464, 0.2863782464, 6.5],
            [0.2863782464, 0.2863782464, 7],
            [0.2863782464, 0.2863782464, 7.5],
            [0.2863782464, 0.2863782464, 8],
            [0.2863782464, 0.2863782464, 8.5],
            [0.2863782464, 0.2863782464, 9],
            [0.2863782464, 0.2863782464, 9.5],
            [-0.2863782464, 0.2863782464, 2],
            [-0.2863782464, 0.2863782464, 2.5],
            [-0.2863782464, 0.2863782464, 3],
            [-0.2863782464, 0.2863782464, 3.5],
            [-0.2863782464, 0.2863782464, 4],
            [-0.2863782464, 0.2863782464, 4.5],
            [-0.2863782464, 0.2863782464, 5],
            [-0.2863782464, 0.2863782464, 5.5],
            [-0.2863782464, 0.2863782464, 6],
            [-0.2863782464, 0.2863782464, 6.5],
            [-0.2863782464, 0.2863782464, 7],
            [-0.2863782464, 0.2863782464, 7.5],
            [-0.2863782464, 0.2863782464, 8],
            [-0.2863782464, 0.2863782464, 8.5],
            [-0.2863782464, 0.2863782464, 9],
            [-0.2863782464, 0.2863782464, 9.5],
            [-0.2863782464, -0.2863782464, 2],
            [-0.2863782464, -0.2863782464, 2.5],
            [-0.2863782464, -0.2863782464, 3],
            [-0.2863782464, -0.2863782464, 3.5],
            [-0.2863782464, -0.2863782464, 4],
            [-0.2863782464, -0.2863782464, 4.5],
            [-0.2863782464, -0.2863782464, 5],
            [-0.2863782464, -0.2863782464, 5.5],
            [-0.2863782464, -0.2863782464, 6],
            [-0.2863782464, -0.2863782464, 6.5],
            [-0.2863782464, -0.2863782464, 7],
            [-0.2863782464, -0.2863782464, 7.5],
            [-0.2863782464, -0.2863782464, 8],
            [-0.2863782464, -0.2863782464, 8.5],
            [-0.2863782464, -0.2863782464, 9],
            [-0.2863782464, -0.2863782464, 9.5],
            [0.2863782464, -0.2863782464, 2],
            [0.2863782464, -0.2863782464, 2.5],
            [0.2863782464, -0.2863782464, 3],
            [0.2863782464, -0.2863782464, 3.5],
            [0.2863782464, -0.2863782464, 4],
            [0.2863782464, -0.2863782464, 4.5],
            [0.2863782464, -0.2863782464, 5],
            [0.2863782464, -0.2863782464, 5.5],
            [0.2863782464, -0.2863782464, 6],
            [0.2863782464, -0.2863782464, 6.5],
            [0.2863782464, -0.2863782464, 7],
            [0.2863782464, -0.2863782464, 7.5],
            [0.2863782464, -0.2863782464, 8],
            [0.2863782464, -0.2863782464, 8.5],
            [0.2863782464, -0.2863782464, 9],
            [0.2863782464, -0.2863782464, 9.5],
            # duplicated and rotated
            [0.405, 0, 2],
            [0.405, 0, 2.5],
            [0.405, 0, 3],
            [0.405, 0, 3.5],
            [0.405, 0, 4],
            [0.405, 0, 4.5],
            [0.405, 0, 5],
            [0.405, 0, 5.5],
            [0.405, 0, 6],
            [0.405, 0, 6.5],
            [0.405, 0, 7],
            [0.405, 0, 7.5],
            [0.405, 0, 8],
            [0.405, 0, 8.5],
            [0.405, 0, 9],
            [0.405, 0, 9.5],
            [0, 0.405, 2],
            [0, 0.405, 2.5],
            [0, 0.405, 3],
            [0, 0.405, 3.5],
            [0, 0.405, 4],
            [0, 0.405, 4.5],
            [0, 0.405, 5],
            [0, 0.405, 5.5],
            [0, 0.405, 6],
            [0, 0.405, 6.5],
            [0, 0.405, 7],
            [0, 0.405, 7.5],
            [0, 0.405, 8],
            [0, 0.405, 8.5],
            [0, 0.405, 9],
            [0, 0.405, 9.5],
            [-0.405, 0, 2],
            [-0.405, 0, 2.5],
            [-0.405, 0, 3],
            [-0.405, 0, 3.5],
            [-0.405, 0, 4],
            [-0.405, 0, 4.5],
            [-0.405, 0, 5],
            [-0.405, 0, 5.5],
            [-0.405, 0, 6],
            [-0.405, 0, 6.5],
            [-0.405, 0, 7],
            [-0.405, 0, 7.5],
            [-0.405, 0, 8],
            [-0.405, 0, 8.5],
            [-0.405, 0, 9],
            [-0.405, 0, 9.5],
            [0, -0.405, 2],
            [0, -0.405, 2.5],
            [0, -0.405, 3],
            [0, -0.405, 3.5],
            [0, -0.405, 4],
            [0, -0.405, 4.5],
            [0, -0.405, 5],
            [0, -0.405, 5.5],
            [0, -0.405, 6],
            [0, -0.405, 6.5],
            [0, -0.405, 7],
            [0, -0.405, 7.5],
            [0, -0.405, 8],
            [0, -0.405, 8.5],
            [0, -0.405, 9],
            [0, -0.405, 9.5]
        ])

    def translate(self, x=0, y=0, z=0):
        """
        Translates the electrode positions to make the viewer have a more accurate view.
        Note that this is in no way connected to actual translations in the calculations.
        Care should be taken to do identical translations for accurate viewing.
        """
        self.elec_pos[:, 0] = self.elec_pos[:, 0] + x
        self.elec_pos[:, 1] = self.elec_pos[:, 1] + y
        self.elec_pos[:, 2] = self.elec_pos[:, 2] + z

    def add_dipole(self, dipole):
        """
        Adds a dipole to the list of dipoles to be plotted.
        Dipole must be a list of size 3.

        Example:
        plotter.add_dipole([0,10,0])
        """
        self.dipoles.append(dipole)

    def plot(self, view=(40,37)):
        """
        Plots the device and dipole on a 3D plot.
        Uses matplotlib to display the graph, so this is best run in a Jupyter notebook.
        """
        electrode_ids = self.electrode_ids[:self.num_electrodes]

        elec_on = self.elec_pos[electrode_ids]
        elec_off = np.setdiff1d(np.arange(self.elec_pos.shape[0]), electrode_ids)

        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d')

        #sns.set_style("whitegrid", {"grid.color": "0", "grid.alpha":"0.6"})
        sns.set_style("whitegrid")

        cmap = LinearSegmentedColormap.from_list("black2orange", [(1,0.2,0),(.2,0,.5)], N=self.num_electrodes)

        on_plot = ax.scatter3D(
                    elec_on[:,0], 
                    elec_on[:,1], 
                    elec_on[:,2], 
                    cmap=cmap,
                    c = self.num_electrodes - np.arange(self.num_electrodes),
                    s = np.arange(self.num_electrodes) / self.num_electrodes * 50 + 10, 
                    alpha = 1,
                    label='on electrodes')

        ax.scatter3D(
            self.elec_pos[elec_off,0], 
            self.elec_pos[elec_off,1], 
            self.elec_pos[elec_off,2], 
            alpha=0.5, 
            s=3,
            c="black",
            marker="X",
            label='off electrodes')

        for dipole in self.dipoles:
            a = _Arrow3D(dipole[0], dipole[1], 
                            dipole[2], mutation_scale=10, 
                            lw=3, arrowstyle="-|>", color="b")
            ax.add_artist(a)

        for line in ax.get_xgridlines() + ax.get_ygridlines() + ax.get_zgridlines():
             line.set_zorder(-10)

        cbar = fig.colorbar(on_plot, ax = ax, shrink = 0.5, aspect = 5, ticks=np.linspace(1, self.num_electrodes, 2)[::-1])
        cbar.ax.invert_yaxis()
        cbar.ax.set_title('Priority (1 = most important)', pad=12)

        ax.view_init(*view)

        plt.title(f'Enabled/Disabled DISC electrodes ({np.shape(electrode_ids)[0]} electrodes)')
        ax.set(xlim=(-.9,.9), ylim=(-.9, .9), zlim=(10,0))
        fig.show()

    def plot_3d(self, view=(40,37)):
        """
        Plots the device and dipole on a 3D plot.
        Uses matplotlib to display the graph, so this is best run in a Jupyter notebook.
        """
        electrode_ids = self.electrode_ids[:self.num_electrodes]

        elec_on = self.elec_pos[electrode_ids]
        elec_off = np.setdiff1d(np.arange(self.elec_pos.shape[0]), electrode_ids)

        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d')

        #sns.set_style("whitegrid", {"grid.color": "0", "grid.alpha":"0.6"})
        sns.set_style("whitegrid")

        cmap = LinearSegmentedColormap.from_list("black2orange", [(1,0.2,0),(.2,0,.5)], N=self.num_electrodes)

        on_plot = ax.scatter3D(
            elec_on[:,0], 
            elec_on[:,1], 
            elec_on[:,2], 
            cmap=cmap,
            c = self.num_electrodes - np.arange(self.num_electrodes),
            s = np.arange(self.num_electrodes) / self.num_electrodes * 50 + 10, 
            alpha = 1,
            label='on electrodes'
        )

        ax.scatter3D(
            self.elec_pos[elec_off,0], 
            self.elec_pos[elec_off,1], 
            self.elec_pos[elec_off,2], 
            alpha=0.5, 
            s=3,
            c="black",
            marker="X",
            label='off electrodes'
        )

        ax.scatter3D(
            self.dipoles[:,0], 
            self.dipoles[:,1], 
            self.dipoles[:,2],
            s=3,
            c="blue",
            marker="o"
        )

        for line in ax.get_xgridlines() + ax.get_ygridlines() + ax.get_zgridlines():
             line.set_zorder(-10)

        cbar = fig.colorbar(on_plot, ax = ax, shrink = 0.5, aspect = 5, ticks=np.linspace(1, self.num_electrodes, 2)[::-1])
        cbar.ax.invert_yaxis()
        cbar.ax.set_title('Priority (1 = most important)', pad=12)

        ax.view_init(*view)

        plt.title(f'Enabled/Disabled DISC electrodes ({np.shape(electrode_ids)[0]} electrodes)')
        ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(10, 0))
        fig.show()

    def plot_2d_side_combined(self):
        """
        Plots a combined view of all 128 electrodes for DISC. 

        Combines the left and right side electrodes into one grid.
        """
        electrode_ids = self.electrode_ids[:self.num_electrodes]

        elec_on = self.elec_pos[electrode_ids]
        elec_off = np.setdiff1d(np.arange(self.elec_pos.shape[0]), electrode_ids)

        fig, ax = plt.subplots(1, 1, figsize=(6,8))

        sns.set_style("whitegrid")

        cmap = LinearSegmentedColormap.from_list("black2orange", [(1,0.2,0),(.2,0,0)], N=self.num_electrodes)

        ax.scatter(
            45 * ((elec_off % 128) // 16), 
            self.elec_pos[elec_off, 2],
            alpha=0.2, 
            s=4*3,
            c="black",
            label='off electrodes'
        )

        sct = ax.scatter(
            45 * ((electrode_ids % 128) // 16), 
            elec_on[:,2], 
            s=(np.arange(self.num_electrodes) / self.num_electrodes * 50 + 10)*3,
            c = self.num_electrodes - np.arange(self.num_electrodes),
            cmap=cmap,
            label='on electrodes',
            vmin=1, vmax=self.num_electrodes
        )

        ax.set(xlim=(-15,330), ylim=(10,1.5), xticks=range(0,360,45), xlabel='Degrees', ylabel='Depth (mm)')
        ax.set_title("Electrodes")

        cbar = fig.colorbar(sct, ax = ax, location="bottom", shrink = 0.95, aspect = 10, ticks=np.linspace(1, self.num_electrodes, 2)[::-1])
        cbar.ax.invert_yaxis()
        cbar.ax.set_title('Priority (1 = most important)', pad=12)


class _Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)