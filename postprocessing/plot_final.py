import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from diagnostics import plot_mac_diagnostics

class DummySolver:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.u = data['u']
        self.v = data['v']
        self.p = data['p']
        self.mask = data['mask']
        self.ny, self.nx = self.p.shape
        self.dx = float(data['dx']) if 'dx' in data else 8.0 / self.nx
        self.dy = float(data['dy']) if 'dy' in data else 2.0 / self.ny
        self.Lx = float(data['Lx']) if 'Lx' in data else self.nx * self.dx
        self.Ly = float(data['Ly']) if 'Ly' in data else self.ny * self.dy
        self.D = float(data['D']) if 'D' in data else 0.4
        self.xc = float(data['xc']) if 'xc' in data else 2.0
        self.yc = float(data['yc']) if 'yc' in data else 1.0
        self.Re = float(data['Re']) if 'Re' in data else 100.0
        self.slip_walls = bool(data['slip_walls']) if 'slip_walls' in data else False
        self.time = float(data['time_history'][-1])
        self.steps = len(data['time_history'])

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    npz_path = os.path.join(repo_root, 'data/final_state_direct.npz') if os.path.exists(os.path.join(repo_root, 'data/final_state_direct.npz')) else 'final_state_direct.npz'
    out_fig = os.path.join(repo_root, 'figures/vorticity_mac_direct.png') if os.path.exists(os.path.join(repo_root, 'figures')) else 'vorticity_mac_direct.png'
    solver = DummySolver(npz_path)
    plot_mac_diagnostics(solver, step=solver.steps, time_val=solver.time, save_path=out_fig)
    print(f"Saved {out_fig}")
