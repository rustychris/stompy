"""
Utilities related to reading DSM2 data
"""
import numpy as np

from .. import utils
from ..grid import unstructured_grid
from ..spatial import wkb2shp

def line_parser(fn):
    """
    Generator to read lines, removing comments along the way
    """
    with open(fn,'rt') as fp:
        for line in fp:
            cmt=line.find('#')
            if cmt>=0:
                line=line[:cmt]
            line=line.strip()
            if not line:
                continue
            yield line

class DSM2Grid(unstructured_grid.UnstructuredGrid):
    def __init__(self,grid_fn,node_shp,channel_shp=None):
        self.src_fn=grid_fn
        self.node_shp=node_shp
        self.channel_shp=channel_shp

        nodes=self.read_nodes()
        
        self.tok=line_parser(grid_fn)

        self.sections={}
        for section in self.tok:
            self.sections[section]=self.read_section()

        extra_node_fields=[]
        extra_edge_fields=[]
        
        for field in nodes.dtype.names:
            if field in ['x','id']: continue
            ftype=np.float64
            extra_node_fields.append( ('id',ftype) )

        for field in self.sections['CHANNEL'].dtype.names:
            if field in ['CHAN_NO','UPNODE','DOWNNODE']:
                continue
            else:
                ftype=np.float64
            extra_edge_fields.append( (field,ftype) )

        super(DSM2Grid,self).__init__(extra_node_fields=extra_node_fields,
                                      extra_edge_fields=extra_edge_fields)

        # Some extra work to make sure that ids line up with DSM ids, even
        # if DSM nodes are missing, and accounting for DSM ids being 1-based.
        # (so self.node[0] and any missing ids will be 'deleted'
        Nnodes=1+nodes['id'].max()
        self.nodes=np.zeros(Nnodes,self.node_dtype)
        self.nodes['deleted']=True
        self.nodes['x'][nodes['id']]=nodes['x']
        self.nodes['deleted'][nodes['id']]=False

        edges=self.sections['CHANNEL']
        eid=edges['CHAN_NO'].astype(np.int32)
        Nedges=1+eid.max()
        self.edges=np.zeros(Nedges,self.edge_dtype)
        self.edges['deleted']=True
        self.edges['deleted'][eid]=False
        self.edges['nodes'][eid,0]=edges['UPNODE'].astype(np.int32)
        self.edges['nodes'][eid,1]=edges['DOWNNODE'].astype(np.int32)
        
        for field in edges.dtype.names:
            if field in ['CHAN_NO','UPNODE','DOWNNODE']: continue
            self.edges[field][eid]=edges[field]

        if channel_shp:
            self.init_channel_geometry(channel_shp)

    def init_channel_geometry(self,channel_shp):
        # Load the channel centerlines
        channels=wkb2shp.shp2geom(channel_shp)
        centerline=np.full(self.Nedges(),None,dtype=np.object_)
        for rec in channels:
            centerline[int(rec['id'])]=np.array(rec['geom'])
        self.add_edge_field('centerline',centerline)

    def read_section(self):
        """
        Read header line with field names, and records up until 'END'
        is encountered. Returns result as numpy struct array.
        Currently assumes that everything is a float.
        """
        header=next(self.tok)
        fields=header.split()
        rows=[]
        dtype=[(f,np.float64) for f in fields]
        dest=np.zeros(0,dtype=dtype)
        
        while 1:
            l=next(self.tok)
            if l=='END':
                break
            dest=utils.array_append(dest)
            for fld,s in zip(fields,l.split()):
                dest[fld][-1]=float(s)
        return dest
            
    def read_nodes(self):
        node_data=wkb2shp.shp2geom(self.node_shp)
        # => dtype=[('id', '<f8'), ('geom', 'O')]

        nodes=np.zeros( len(node_data), dtype=[('id',np.int32),
                                               ('x',np.float64,2)])
        nodes['id']=node_data['id'].astype(np.int32)
        nodes['x']=[np.array(p) for p in node_data['geom']]
        return nodes

    def plot_edges(self,*a,centerlines=False,**k):
        if centerlines:
            orig_return_mask=k.pop('return_mask',False)
            k['return_mask']=True
            lcoll,mask=super(DSM2Grid,self).plot_edges(*a,**k)
            segs=lcoll.get_segments()
            for ji,j in enumerate(np.nonzero(mask)[0]):
                if self.edges['centerline'][j] is not None:
                    segs[ji]=self.edges['centerline'][j]
            lcoll.set_segments(segs)
            if orig_return_mask:
                return lcoll,mask
            else:
                return lcoll
        else:
            return super(DSM2Grid,self).plot_edges(*a,**k)

