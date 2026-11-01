"""A sealed disconnected fluid region must fail the pressure uniqueness check."""
from types import SimpleNamespace
import numpy as np
from validation.pressure_graph_audit import inspect


def run():
    geom = SimpleNamespace(nx=4,ny=4,dx=.25,dy=.25,
        ax=np.ones((4,5)),ay=np.ones((5,4)),cell_type=np.ones((4,4),dtype=int))
    good = inspect(geom)
    assert good['coercivity_graph_conditions_met'] and good['components'] == 1
    assert good['checkerboard_rayleigh_quotient'] > 0
    geom.ax[:,2] = 0
    bad = inspect(geom)
    assert bad['components'] == 2 and not bad['every_component_grounded_at_outlet']
    assert not bad['coercivity_graph_conditions_met']
    print('PASS: grounded fluid graph; sealed pressure-null component rejected')


if __name__ == '__main__':
    run()
