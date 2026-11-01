"""Check variance attribution without running the numerical solver."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.audit_ghost_stage0 import components


class ForceAttributionTests(unittest.TestCase):
    def test_correlated_components_and_closure(self):
        t=np.linspace(0,2*np.pi,1001)
        pressure=3+np.sin(t)
        viscous=-2+.5*np.sin(t)
        result=components(t,pressure+viscous,pressure,viscous,t[0],t[-1])
        self.assertAlmostEqual(result['correlation'],1)
        np.testing.assert_allclose(result['variance_shares_percent'],[400/9,100/9,400/9],atol=1e-10)
        self.assertLess(result['closure_linf'],1e-12)

    def test_inconsistent_or_constant_force_rejected(self):
        t=np.linspace(0,1,10)
        with self.assertRaisesRegex(ValueError,'sum to total'):
            components(t,t+1,t,t,t[0],t[-1])
        with self.assertRaisesRegex(ValueError,'nonconstant'):
            components(t,np.ones(10),np.ones(10),np.zeros(10),t[0],t[-1])


if __name__=='__main__':
    unittest.main()
