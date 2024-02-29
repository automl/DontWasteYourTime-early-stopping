from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

# https://stackoverflow.com/a/47664533/5332072


class SeabornFig2Grid:
    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid | sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """Move PairGrid or Facetgrid."""
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            n,
            m,
            subplot_spec=self.subplot,
            hspace=0,
            wspace=0,
        )
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """Move Jointgrid."""
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))

        marg_y_deactivated = getattr(self.sg.ax_marg_y, "monkey_deactivated", False)
        r2 = r if marg_y_deactivated else r + 1

        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            r + 1,
            r2,
            subplot_spec=self.subplot,
            wspace=0.0,
            hspace=0.0,
        )

        if marg_y_deactivated:
            self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :])
            self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :])
        else:
            self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
            self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
            self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
