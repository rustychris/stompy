import os

from stompy.grid import trigrid

def test_read_sun():
    path=os.path.join( os.path.dirname(__file__),'data','sfbay')

    g = trigrid.TriGrid(suntans_path=path)
    g.Ncells()
    

if __name__ == '__main__':
    g = TriGrid(sms_fname="/home/rusty/data/sfbay/grids/100km-arc/250m/250m-100km_arc.grd")
    g.plot()
    g.verify_bc(do_plot=1)



