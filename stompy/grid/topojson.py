import json
from . import unstructured_grid
import pandas as pd
import numpy as np

class GridEncoder(json.JSONEncoder):
    exclude=[]
    cell_halfedges=True
    cell_nodes=True
    subedges=None
    
    def default(self,o,strip_negative=[]):
        if isinstance(o,unstructured_grid.UnstructuredGrid):
            # extra handling for cells - unstructured_grid is node
            # centric, but C# code is half-edge based.
            # Topojson is also arc-centric.
            # Could have regions and arcs as the objects?
            # or is there one object, a conceptualMesh, which is a GeometryCollection?
            # That seems more flexible. Not quite sure. Re-evaluate once I can
            # test the result in external software.
            # It's a little weird to have a geometry collection with both polygons
            # and cells.

            # Could have several objects at the top level
            #   regions
            #   arcs (choose different object name?) - these would be single
            #      arcs.
            #   breaklines - distinct from arcs, and can be one or more arcs.
            
            # Or it could be geometryCollection of geometryCollections.
            #   one issue there is that the geometryCollection.geometries is an array,
            #   so I could not name a collection regions and another collection arcs.
            #   instead the name would have to go inside, have a convention, or interpret
            #   based on inspection of items.
         
            #dict(Cells=self.convert_cells(o,strip_negative=strip_negative),
            #     Faces=o.edges,
            #     Nodes=self.convert_nodes(o))
            
            return {'type':'Topology',
                    'objects':{
                        'regions':{
                            'type':'GeometryCollection',
                            'geometries':self.cell_geometries(o),
                        },
                        'arcs':{
                            'type':'GeometryCollection',
                            'geometries':self.edge_geometries(o),
                        },
                        'breaklines':{
                            'type':'GeometryCollection',
                            'geometries':self.breakline_geometries(o),
                        }
                    },
                    'arcs':self.arc_geometries(o)
                   }
        elif isinstance(o,np.ndarray):
            if o.dtype.names is not None:
                if o.ndim!=1:
                    raise Exception("Support only for 1D struct arrays")
                df=pd.DataFrame()
                for name in o.dtype.names:
                    if name in self.exclude: continue
                    col_values= list(o[name])
                    if name in strip_negative:
                        col_values=[val[val>=0] for val in col_values]
                    df[name] = col_values

                return df.to_dict('records')
            else:
                return list(o)
        elif isinstance(o,np.integer):
            return int(o)
        elif isinstance(o,np.bool_):
            return bool(o)
        elif isinstance(o,np.floating):
            return float(o)
        else:
            return super().default(o)

    def breakline_geometries(self,o):
        return [] # not ready.
    
    def arc_geometries(self,o):
        geoms=[]
        # Maybe it wouldn't be a problem, but not ready to find out.
        assert np.all(~self.edges['deleted'])
        
        for j in range(o.Nedges()):
            if self.subedges is not None:
                geoms.append( o.edges[self.subedges][j] )
            else:
                geoms.append( o.nodes['x'][o.edges['nodes'][j]] )
        return geoms
    
    def edge_geometries(self,o):
        geoms=[]
        for j in range(o.Nedges()):
            geoms.append({'type':'LineString',
                          'arcs':[j],
                          })
        return geoms

    def cell_geometries(self,o):
        # strip_negative?
        # def convert_cells(self,o,strip_negative=[]):
        cells=o.cells
        df=pd.DataFrame()
        for name in cells.dtype.names:
            if name in self.exclude: continue
            if name=='nodes': continue
            if name=='edges' and self.cell_halfedges: continue
            col_values= list(cells[name])
            if name in strip_negative:
                col_values=[val[val>=0] for val in col_values]
            df[name] = col_values

        if self.cell_halfedges:
            hes=[]
            for c in range(o.Ncells()):
                hes.append( [ {'fIdx':he.j,'orient':he.orient} 
                              for he in o.cell_to_halfedges(c) ] )
            df['faces'] = hes
        return df.to_dict('records')

    def convert_nodes(self,o):
        nodes=o.nodes
        df=pd.DataFrame()
        for name in nodes.dtype.names:
            if name in self.exclude: continue
            if name=='x': # unpack x coordinate into x and y 
                df['x']=nodes['x'][:,0]
                df['y']=nodes['x'][:,1]
                continue
            df[name] = list(nodes[name])

        return df.to_dict('records')

def write_grid(grid, filename=None, stream=None,
               exclude=[],cell_halfedges=True,cell_nodes=False,
               subedges=None):
    if stream is None:
        assert filename is not None
        with open(filename,'wt') as stream:
            return write_grid(grid, stream=stream,exclude=exclude,cell_halfedges=cell_halfedges,
                              cell_nodes=cell_nodes)

    class CustomGridEncoder(GridEncoder): pass
    CustomGridEncoder.exclude=exclude
    CustomGridEncoder.cell_halfedges=cell_halfedges
    CustomGridEncoder.cell_nodes=cell_nodes
    CustomGridEncoder.subedges=subedges
    json.dump(grid,stream,cls=CustomGridEncoder,skipkeys=True)

    
