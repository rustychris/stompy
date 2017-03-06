import nose
from stompy.grid import unstructured_grid
reload(unstructured_grid)
from nose.tools import assert_raises

##

def test_merge():
    ugA=unstructured_grid.SuntansGrid('/Users/rusty/src/umbra/Umbra/sample_data/sfbay')
    ugB=unstructured_grid.SuntansGrid('/Users/rusty/src/umbra/Umbra/sample_data/sfbay')

    x_cut=568000
    n_sel=ugA.nodes['x'][:,0]<x_cut


    for n in np.nonzero(n_sel)[0]:
        ugA.delete_node_cascade(n)
    for n in np.nonzero(~n_sel)[0]:
        ugB.delete_node_cascade(n)

    if 0:
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        ugA.plot_edges(ax=ax,color='r')
        ugB.plot_edges(ax=ax,color='b')

    ugA.add_grid(ugB)

    if 0: 
        plt.figure(2).clf()
        fig,ax=plt.subplots(num=2)

        ugA.plot_edges(ax=ax,color='r')


## 
if __name__=='__main__':
    nose.main()
