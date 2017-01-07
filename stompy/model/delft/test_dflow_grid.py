import nose
import nefis
import numpy as np
import dflow_grid as dg

def load_data():
    nef=nefis.Nefis('data/trim-tut_fti_waq.dat',
                    'data/trim-tut_fti_waq.def')
    g=dg.DflowGrid2D.from_nef_map(nef)

    u1=nef['map-series'].getelt('U1',[10])
    rho=nef['map-series'].getelt('RHO',[10])
    dp0=nef['map-const'].getelt('DP0',[0])
    dps0=nef['map-const'].getelt('DPS0',[0])

    nef.close()
    return g,u1,rho,dp0,dps0

def test_create():
    g,u1,rho,dp0,dps0=load_data()
    assert(g)

# Plotting
def test_wireframe():
    g,u1,rho,dp0,dps0 = load_data()
    g.wireframe()

def test_pcolor_face():
    g,u1,rho,dp0,dps0 = load_data()
    g.pcolor_face(rho[1])

def test_contourf_node():
    g,u1,rho,dp0,dps0 = load_data()
    g.contourf_node(dp0)

def test_pcolor_node():
    g,u1,rho,dp0,dps0 = load_data()
    g.pcolor_node(dp0)

if __name__ == '__main__':
    nose.main()
