from __future__ import print_function


from stompy import harm_decomp
import numpy as np

def test_basic():
    # A sample problem:
    omegas = np.array([1.0,0.0])

    # the constructed data:
    amps = np.array([1,5.0])
    phis = np.array([1,0])

    t = np.linspace(0,10*np.pi,125)
    h = amps[0]*np.cos(omegas[0]*t - phis[0]) + amps[1]*np.cos(omegas[1]*t - phis[1])

    comps = harm_decomp.decompose(t,h,omegas)
    recon=harm_decomp.recompose(t,comps,omegas)

    assert np.allclose( recon, h)

    print("Components: ",comps)

