
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

    def plot_several_cells(self, atom_col, ax=None, method="auto", add_cols="C1"):
        add_plot_elems = []
        ax = atom_col.plot_axes
        if atom_col.plot_elems["extra_cells"] is None:
            if method == "auto":
                l1,l2 = atom_col.unit_cell
                positions = atom_col.positions
                for x_step in [-2,-1, 0, 1, 2]:
                    p_disp_x = []
                    for y_step in [-2,-1, 0, 1, 2]:
                        p_disp_y = []
                        if x_step == 0 and y_step == 0:
                            pass
                        else:
                            disp_pos = x_step*l1 + y_step*l2
                            new_poses = positions + disp_pos
                            for new_pos in new_poses:
                                p = ax.plot(new_pos[0], new_pos[1], "o", c=add_cols, ms=self.markersize, alpha=self.alpha*0.6, markeredgecolor="k")[0]
                                p_disp_y.append(p)
                        p_disp_x.append(p_disp_y)
                    add_plot_elems.append(p_disp_x)
            atom_col.plot_elems["extra_cells"] = add_plot_elems
        else:
            l1,l2 = atom_col.unit_cell
            positions = atom_col.positions
            for plot_elems_disp_x, x_step in zip(atom_col.plot_elems["extra_cells"], [-2,-1, 0, 1, 2]):
                for plot_elems_disp_y, y_step in zip(plot_elems_disp_x, [-2,-1, 0, 1, 2]):
                    disp_pos = x_step*l1 + y_step*l2
                    new_poses = positions + disp_pos
                    for plot_elem, new_pos in zip(plot_elems_disp_y, new_poses):
                        plot_elem.set_data(new_pos[0], new_pos[1])
        return atom_col.plot_elems["extra_cells"]
    
    def plot_cell(self, atom_col):
        ax = atom_col.plot_axes
        plot_elems = []
        if atom_col.plot_elems["unit_cell"] is None:
            try:
                l1,l2 = atom_col.unit_cell
            except:
                raise Exception("No proper unitcell has been supplied for plotting")
            
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
                                   "extra_cells":None,
                                   }
        if plot_cell == True:
            atom_col.plot_elems["unit_cell"] = self.plot_cell(atom_col)
            atom_col.plot_elems["atoms"] = self.plot_atom_col(atom_col)
        else:
            atom_col.plot_elems["atoms"] = self.plot_atom_col(atom_col)
        
        return atom_col.plot_elems
    
class Animator(Atom_Collection_Plotter):
    def __init__(self, default_cols=("C0", "C1"), markersize=50, scale_with_mass=False, alpha=1) -> None:
        super().__init__(default_cols, markersize, scale_with_mass, alpha)

    def animate(self):
        raise Exception("SHOULD BE IMPLEMENTED IN SUBCLASS")