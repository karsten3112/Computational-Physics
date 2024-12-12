
class Atom_Collection_Plotter():
    def __init__(self, default_cols=("C0", "C1"), markersize=50.0, scale_with_mass=False, alpha=1.0) -> None:
        self.markersize = markersize
        self.default_cols = default_cols
        self.scale_with_mass = scale_with_mass
        self.alpha = alpha
    
    def plot_atom_col(self, atom_col):
        ax = atom_col.plot_axes
        plot_elems = []
        if atom_col.plot_elems["atoms"] is None:
            for position, frozen in zip(atom_col.positions, atom_col.frozens):
                p = ax.plot(position[0], position[1], "o", c=self.default_cols[frozen.astype(int)], ms=self.markersize, alpha=self.alpha, markeredgecolor="k")[0]
                plot_elems.append(p)
        else:
            for position, plot_elem in zip(atom_col.positions, atom_col.plot_elems["atoms"]):
                plot_elem.set_data(position[0], position[1])
            plot_elems = atom_col.plot_elems["atoms"]
        return plot_elems
            #print("Has not been implemented for updating yet")

    def plot_cell(self, atom_col):
        ax = atom_col.plot_axes
        plot_elems = []
        if atom_col.plot_elems["unit_cell"] is None:
            l1,l2 = self.unit_cell
            l_end = l1+l2
            for l in [l1, l2]:
                p1 = ax.plot([0.0, l[0]], [0.0, l[1]], "--", c='k')[0]
                p2 = ax.plot([l[0], l_end[0]],[l[1], l_end[1]], "--", c='k')[0]
                plot_elems.append([p1,p2])
            return plot_elems
        else:
            print("Has not been implemented yet")

    def __call__(self, atom_col, ax=None, plot_cell=False):
        if atom_col.plot_axes is None:
            atom_col.plot_axes = ax
            atom_col.plot_elems = {"atoms":None,
                                   "unit_cell":None,
                                   }
        if plot_cell == True:
            try:
                atom_col.plot_elems["unit_cell"] = self.plot_cell(atom_col)
            except:
                raise Exception("No proper unit_cell has been supplied and cannot be plotted")
        else:
            atom_col.plot_elems["atoms"] = self.plot_atom_col(atom_col)
        
        return atom_col.plot_elems
    
class Animator(Atom_Collection_Plotter):
    def __init__(self, default_cols=("C0", "C1"), markersize=50, scale_with_mass=False, alpha=1) -> None:
        super().__init__(default_cols, markersize, scale_with_mass, alpha)

    def animate(self):
        raise Exception("SHOULD BE IMPLEMENTED IN SUBCLASS")