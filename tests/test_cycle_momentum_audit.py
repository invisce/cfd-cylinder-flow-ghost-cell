"""Analytical accelerating control volume across force zero crossings."""
import numpy as np
from validation.cycle_momentum_audit import balance_statistics


def run():
    errors = []
    for dt in (.02,.01,.005):
        t = np.arange(0,4+dt/2,dt)
        momentum = np.column_stack((np.sin(np.pi*t),np.cos(np.pi*t)))
        derivative = np.column_stack((np.pi*np.cos(np.pi*t),-np.pi*np.sin(np.pi*t)))
        surface = np.column_stack((3+.1*np.cos(2*np.pi*t),np.sin(np.pi*t)))
        data = np.column_stack((t,momentum,momentum,surface+derivative,surface))
        result = balance_statistics(data,.5)
        errors.append(result['shared_volume']['rms_coefficient_difference'][1])
        assert np.isfinite(result['shared_volume']['lift_rms_error_relative_to_fluctuation'])
        assert result['shared_volume'] == result['outer_mass_moment']
    assert np.all((np.log2(np.array(errors[:-1])/errors[1:]) > 1.95))
    print('PASS: phase-wide centered storage order two and robust zero-crossing normalization')


if __name__ == '__main__':
    run()
