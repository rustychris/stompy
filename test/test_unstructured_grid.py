from __future__ import print_function

import numpy as np
import os, shutil
import nose
from nose.tools import assert_raises

from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy import utils


def test_undo_00():
    ug=unstructured_grid.UnstructuredGrid()

    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[0,1])
    n3=ug.add_node(x=[0.67,0.5])

    e1=ug.add_edge(nodes=[n1,n2])
    e2=ug.add_edge(nodes=[n1,n3])
    cp=ug.checkpoint()
    e3=ug.add_edge(nodes=[n2,n3])
    c0=ug.add_cell(nodes=[n1,n2,n3])

    assert ug.Ncells()==1
    assert ug.Nedges()==3
    assert ug.Nnodes()==3
    
    ug.revert(cp)

    assert ug.Ncells()==0
    assert ug.Nedges()==2
    assert ug.Nnodes()==3

## 

sample_data=os.path.join(os.path.dirname(__file__),'data')

def test_toggle_toggle():
    ug=unstructured_grid.SuntansGrid(os.path.join(sample_data,'sfbay'))
    xy=(507872, 4159018)
    c=ug.select_cells_nearest(xy)
    nodes=ug.cell_to_nodes(c)
    chk=ug.checkpoint()
    ug.toggle_cell_at_point(xy)
    ug.revert(chk)
    ug.delete_node_cascade(nodes[0])

def test_delete_undelete():
    ug=unstructured_grid.SuntansGrid(os.path.join(sample_data,'sfbay'))
    xy=(507872, 4159018)
    c=ug.select_cells_nearest(xy)
    nodes=ug.cell_to_nodes(c)
    chk=ug.checkpoint()
    ug.delete_cell(c)
    ug.revert(chk)
    ug.delete_node_cascade(nodes[0])

##

def test_write_formats():
    g=unstructured_grid.UnstructuredGrid.read_dfm(os.path.join(sample_data,"lsb_combined_v14_net.nc"))

    def check_similar(gA,gB):
        assert gA.Ncells()==gB.Ncells()
        assert gA.Nnodes()==gB.Nnodes()
        assert gA.Nedges()==gB.Nedges()
        
    dfm_fn="test-write_net.nc"
    g.write_dfm(dfm_fn)
    g2=unstructured_grid.UnstructuredGrid.read_dfm(dfm_fn)
    check_similar(g,g2)
    os.unlink(dfm_fn)
    
    sun_dir="sun_test_out"
    os.path.exists(sun_dir) or os.makedirs(sun_dir)
    # g.write_suntans only works for triangular grids
    g.write_suntans_hybrid(sun_dir)
    g2=unstructured_grid.UnstructuredGrid.read_suntans(sun_dir)
    check_similar(g,g2)
    shutil.rmtree(sun_dir)
    
    _=g.write_to_xarray()
    g.write_ugrid("test-write.nc")
    os.unlink("test-write.nc")
    g.write_pickle("test-write.pkl")
    os.unlink("test-write.pkl")

    # Quad/tri only: fudge it and create a quad/tri grid by just deleting the
    # pents and hexes (of which there are only 6)
    large_cells=[i for i in g.valid_cell_iter() if len(g.cell_to_nodes(i))>4]
    for c in large_cells:
        g.delete_cell(c)
    g.renumber()
        
    g.write_untrim08("test-write-untrim.grd")
    os.unlink("test-write-untrim.grd")
    g.write_ptm_gridfile("test-write-ptm.grd")
    os.unlink("test-write-ptm.grd")
    

## 

def test_triangle_from_scratch():
    ug=unstructured_grid.UnstructuredGrid(max_sides=4)

    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[1,0])
    n3=ug.add_node(x=[1,1])

    j1=ug.add_edge(nodes=[n1,n2])
    j2=ug.add_edge(nodes=[n2,n3])
    j3=ug.add_edge(nodes=[n3,n1])

    c1=ug.toggle_cell_at_point([0.2,0.2])

def test_quad_from_scratch():
    ug=unstructured_grid.UnstructuredGrid(max_sides=4)

    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[1,0])
    n3=ug.add_node(x=[1,1])
    n4=ug.add_node(x=[0,1])

    j1=ug.add_edge(nodes=[n1,n2])
    j2=ug.add_edge(nodes=[n2,n3])
    j3=ug.add_edge(nodes=[n3,n4])
    j4=ug.add_edge(nodes=[n4,n1])

    c1=ug.toggle_cell_at_point([0.2,0.2])
    # and remove it..
    ug.toggle_cell_at_point([0.2,0.2])


def test_duplicate_edge():
    ug=unstructured_grid.UnstructuredGrid(max_sides=4)

    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[1,0])

    j1=ug.add_edge(nodes=[n1,n2])

    with assert_raises(unstructured_grid.GridException) as cm:
        j2=ug.add_edge(nodes=[n2,n1])

## 

def test_merge_nodes():
    gt=unstructured_grid.UnstructuredGrid()

    gt.add_rectilinear([0,0],[2,2],3,3)

    c_dupe=3

    new_nodes=[ gt.add_node(x=gt.nodes['x'][n] + np.r_[0.1,0.1] )
                for n in gt.cells['nodes'][c_dupe] ]
    gt.modify_cell(c_dupe,nodes=new_nodes)
    gt.make_edges_from_cells()
    gt.delete_orphan_nodes() 
    gt.renumber_nodes()

    # at this point, need to merge three pairs of nodes:
    # 4 and 8 -- this is the central pair, of the tree.
    # 5 and 11
    # 7 and 9

    gt.merge_nodes(7,9)
    gt.merge_nodes(5,11)
    gt.merge_nodes(4,8)

def test_mass_delete():
    g=unstructured_grid.UnstructuredGrid.read_dfm(os.path.join(sample_data,"lsb_combined_v14_net.nc"))

    g.edge_to_cells()

    clip=(577006.59313042194, 579887.89496937161, 4143066.3785693897, 4145213.4131655102)

    node_to_del=np.nonzero(g.node_clip_mask(clip))[0]

    for n in node_to_del:
        g.delete_node_cascade(n)
    g.renumber_nodes()

    
##         
def test_pickle():
    ug=unstructured_grid.UnstructuredGrid(max_sides=4)
    def cb(*a,**k):
        pass
    ug.subscribe_after('add_node',cb)
    chk=ug.checkpoint()

    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[1,0])

    j1=ug.add_edge(nodes=[n1,n2])

    pkl_fn='blah.pkl'
    ug.write_pickle(pkl_fn)
    ug2=unstructured_grid.UnstructuredGrid.from_pickle(pkl_fn)
    os.unlink(pkl_fn)


## 

def test_modify_max_sides():
    ug=unstructured_grid.SuntansGrid(os.path.join(sample_data,'sfbay') )

    ug.modify_max_sides(max_sides=6)
    
    n1=ug.add_node(x=[0,0])
    n2=ug.add_node(x=[1,0])
    n3=ug.add_node(x=[1,1])
    n4=ug.add_node(x=[0,1])

    ug.add_cell_and_edges(nodes=[n1,n2,n3,n4])

    ug.modify_max_sides(max_sides=4)

    with assert_raises(unstructured_grid.GridException) as cm:
        ug.modify_max_sides(max_sides=3)

## 

def test_boundary_polygon():
    ug=unstructured_grid.SuntansGrid(os.path.join(sample_data,'sfbay') )
    poly1=ug.boundary_polygon()
    poly2=ug.boundary_polygon_by_edges()
    poly3=ug.boundary_polygon_by_union()

    print("poly1 area:",poly1.area)
    print("poly2 area:",poly2.area)
    print("poly3 area:",poly3.area)

    # in one test, these were the same out to 12 digits.
    assert abs( (poly1.area - poly2.area) / poly1.area ) < 1e-10
    assert abs( (poly1.area - poly3.area) / poly1.area ) < 1e-10


def test_nearest_cell():
    ug=unstructured_grid.SuntansGrid(os.path.join(sample_data,'sfbay') )
    
    target=[550000,4.14e6]
    hit1=ug.select_cells_nearest(target,inside=True)
    hit2=ug.cell_containing(target)

    assert hit1==hit2

## 
    
if __name__=='__main__':
    nose.main()
