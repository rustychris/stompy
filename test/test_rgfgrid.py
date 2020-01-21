"""
Read DELFT3D structured RGFGRID grd files as unstructured.
"""

from stompy.grid import unstructured_grid
import numpy as np
import six
six.moves.reload_module(unstructured_grid)

## 

class RgfGrid(unstructured_grid.UnstructuredGrid):
    max_sides=4
    class GrdTok(object):
        def __init__(self,grd_fn):
            self.fp=open(grd_fn,'rt')
            self.buff=None # unprocessed data
        def read_key_value(self):
            while self.buff is None:
                self.buff=self.fp.readline().strip()
                if self.buff[0]=='*':
                    self.buff=None
            assert '=' in self.buff
            key,value=self.buff.split('=',1)
            self.buff=None
            key=key.strip()
            value=value.strip()
            return key,value
        def read_token(self):
            while self.buff is None:
                self.buff=self.fp.readline().strip()
                if self.buff[0]=='*':
                    self.buff=None
            parts=self.buff.split(None,1)
            if len(parts)==0:
                self.buff=None
                return None
            if len(parts)==1:
                self.buff=None
                return parts[0]
            if len(parts)==2:
                self.buff=parts[1]
                return parts[0]
            raise Exception("not reached")

    def __init__(self,grd_fn):
        super(RgfGrid,self).__init__()
        
        tok=self.GrdTok(grd_fn)

        _,coord_sys=tok.read_key_value()
        _,missing_val=tok.read_key_value()
        missing_val=float(missing_val)

        m_count=int(tok.read_token())
        n_count=int(tok.read_token())
        [tok.read_token() for _ in range(3)] # docs say they aren't used

        xy=np.zeros( (n_count,m_count,2), np.float64)

        def read_coord():
            v=float(tok.read_token())
            if v==missing_val:
                return np.nan
            else:
                return v

        for comp in [0,1]:
            for row in range(n_count):
                tok.read_token()  # ETA=
                row_num=int(tok.read_token())
                assert row_num==row+1

                for col in range(m_count):
                    xy[row,col,comp]=read_coord()

        self.add_node_field('row',np.zeros(0,np.int32))
        self.add_node_field('col',np.zeros(0,np.int32))
        self.add_cell_field('row',np.zeros(0,np.int32))
        self.add_cell_field('col',np.zeros(0,np.int32))

        # Add nodes:
        node_idxs=np.zeros( (n_count,m_count), np.int32)-1
        cell_idxs=np.zeros( (n_count-1,m_count-1), np.int32)-1

        for row in range(n_count):
            for col in range(m_count):
                if np.isfinite(xy[row,col,0]):
                    node_idxs[row,col]=self.add_node(x=xy[row,col],row=row,col=col)

        # Add cells, filling in edges as needed
        for row in range(n_count-1):
            for col in range(m_count-1):
                nodes=[ node_idxs[row,col],
                        node_idxs[row,col+1],
                        node_idxs[row+1,col+1],
                        node_idxs[row+1,col] ]
                if np.any(np.array(nodes)<0): continue
                cell_idxs[row,col]=self.add_cell_and_edges(nodes=nodes,row=row,col=col)

        # Fast lookup -- but might become stale...
        self.rowcol_to_node=node_idxs
        self.rowcol_to_cell=cell_idxs
        self.grd_filename=grd_fn
    def read_enclosure(self,enc_fn):
        """
        Read the enclosure file. Saves the list of row/col indices, 0-based,
        to self.enclosure.

        Note that this is just for logical comparisons on the
        grid, not for geographic representation. The range of indices is 1 greater
        than the grid indices for nodes and 2 greater than grid indices for cells.
        This is because the vertices of the enclosure are on "ghost" cell centers
        outside the actual domain. 
        """
        with open(enc_fn,'rt') as fp:
            ijs=[]
            for line in fp:
                line=line.strip().split('*')[0]
                if not line: continue
                row,col=[int(s) for s in line.split()]
                ijs.append( [row,col] )
        self.enclosure=np.array(ijs)-1
        
##

import glob
#grd_fns=glob.glob('/home/rusty/tmp/ucb_model/Models/MouthVariation/EpMp/*.grd')
#grids=[RgfGrid(fn) for fn in grd_fns]
# grd_fn="data/WET.grd"
grd_fn="/home/rusty/src/pescadero/data/ucb-model/MouthVariation/EpMp/PDO_EpMp.grd"
enc_fn="/home/rusty/src/pescadero/data/ucb-model/MouthVariation/EpMp/PDO_EpMp.enc"
grd=RgfGrid(grd_fn)


# And the depth file?
# Hmm - have a staggering issue.  This file is 1 larger.
# docs say that the grd file has coordinates for the "depth points".
# maybe depth is given at nodes, but "depth points" is like arakawa
# C, and cell-centered?

dep_fn=grd.grd_filename.replace('.grd','.dep')
#dep_fn=grd.grd_filename.replace('.grd','_DEM.dep')

dep_data=np.fromfile(dep_fn,sep=' ')
dep2d=dep_data.reshape( (grd.rowcol_to_node.shape[0]+1,
                         grd.rowcol_to_node.shape[1]+1) )
# Seems like the staggering is off, but when I try to average down to
# the number of nodes I have, the values are bad.  Suggests that even though
# the depth data is 1 larger in each coordinate direction, it is still just
# node centered (or at least centered on what I have claimed to be nodes...)
#dep2d_centered=0.25*(dep2d[1:,1:] + dep2d[:-1,:-1] + dep2d[1:,:-1] + dep2d[:-1,1:])
dep2d_centered=dep2d[:-1,:-1]
dep_node_centered=dep2d_centered[ grd.nodes['row'], grd.nodes['col']]

grd.add_node_field('depth_node',dep_node_centered)

grd.add_cell_field('depth_cell',grd.interp_node_to_cell(dep_node_centered))

##

# max row is 595, max col is 146
# but that is one bigger than nodes.
# Suggests that the "depth points" really are at the centers of cells.
# Figure A.1 in the RGFgrid manual shows the enclosure polygon intersecting
# the water level points.  There is a better figure in the Delft3D-Flow
# manual.

grd.write_cells_shp('/home/rusty/src/pescadero/data/ucb-model/epmp-pdo-cells.shp',
                    extra_fields=[('depth',grd.cells['depth_cell'])])

##

import matplotlib.pyplot as plt
plt.figure(1).clf()

grd.plot_edges(lw=0.5)
plt.axis('equal')

ccoll=grd.plot_nodes(values=grd.nodes['depth_node'],cmap='jet')
ccoll.set_clim([-5,5])
plt.colorbar(ccoll)

# HERE
# Resolve the staggering question.  Might have to load this into Delta Shell
# to figure it out.
# But Delta Shell license is stale.

# RGFGrid manual says that grd file gives the coordinates of the orthogonal
# curvilinear grid at the "depth points"

# the mdf file for PDO gives Mmax=595 Nmax=146 Kmax=1
# D3D flow manual says that the number of cells in each direction is *2* less
# than the max values.

# That is consistent with my grd.rowcol_to_cell.shape == (144,593)
# The enclosure polygon includes points that are not valid nodes of the grid,
# so it is not necessarily surprising that it is larger.

# D3D Flow manual says *.dep file should have entries for all Nmax x Mmax
# The grd file gives dimensions 1 less than Nmax, Mmax.
# In the example dep file they give in the manual, the last row and last
# column are in fact missing values (-999).
# That suggests my interpretation above is correct.
# Appendix E of the manual mentions depth defined at the corners of the
# control volume.
