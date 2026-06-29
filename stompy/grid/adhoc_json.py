# -*- coding: utf-8 -*-
"""
Read ad-hoc unstructured grids for RAS development.

Created on Mon Feb 27 15:14:03 2023

@author: rusty
"""
import json
import pandas as pd
from . import unstructured_grid
from ..spatial import field
import numpy as np

def read_json_mesh(fn):
    with open(fn) as fp:
        raw=json.load(fp)
    return json_to_mesh(raw)


def json_to_raster(raw):
    extents=[raw['MinX'],raw['MaxX'],
             raw['MinY'],raw['MaxY']]
    F=np.array([float(f) for f in raw['data']])
    F=F.reshape(raw['Rows'],raw['Cols'])[::-1,:]
    return field.SimpleGrid(extents=extents,F=F)
    
def json_to_mesh(raw):    
    nodes = pd.DataFrame(raw['Nodes'])
    edges = pd.DataFrame(raw['Faces'])
    cells = pd.DataFrame(raw['Cells'])
    
    if len(cells)==0:
        max_sides=10
    else:
        max_sides=cells.faces.apply(lambda f : len(f)).max()
    
    g=unstructured_grid.UnstructuredGrid(max_sides=max(max_sides,3))
    
    g.nodes=np.zeros(len(nodes),g.node_dtype)
    for col in nodes.columns:
        if col in ['deleted']:
            g.nodes['deleted'] = nodes.deleted
        elif col=='x':
            g.nodes['x'][:,0] = nodes.x.values
        elif col=='y':
            g.nodes['x'][:,1] = nodes.y.values
        elif col not in g.nodes.dtype.names:
            # blindly attempt adding a column
            data = nodes[col].values
            if data.dtype==object:
                try:
                    # coerce NaN to float nan
                    data = data.astype(np.float64)
                except ValueError:
                    pass
            try:
                g.add_node_field(col,data)
            except Exception as exc:
                print("Failed to add node column %s"%col)
    
    g.edges=np.zeros(len(edges), g.edge_dtype)
    for col in edges.columns:
        if col=='nodeA':
            g.edges['nodes'][:,0] = edges.nodeA.values
        elif col=='nodeB':
            g.edges['nodes'][:,1] = edges.nodeB.values
        elif col=='nodes':
            # use pandas to convert array of dicts to [N,2] array
            nodeAB=pd.DataFrame(list(edges.nodes))
            g.edges['nodes'][:,:] = nodeAB.values
        elif col=='cells':
            cellAB=pd.DataFrame(list(edges.cells))
            g.edges['cells'][:,:]=cellAB.values            
        elif col=='deleted':
            g.edges['deleted'] = edges.deleted
        elif col.endswith('XY'):
            # Kludge - if a columns has an XY suffix treat it as a
            # list of coords for each edge.
            g.add_edge_field(col, edges[col].apply(np.array).values)
        elif col not in g.edges.dtype.names: 
            # blindly attempt adding a column
            try:
                g.add_edge_field(col,edges[col].values)
            except Exception as exc:
                print("Failed to add edge column %s"%col)
        
    g.cells=np.zeros(len(cells),g.cell_dtype)
    if len(cells)>0:
        g.cells['deleted']=cells.deleted
        g.cells['edges'] = -1
        g.cells['nodes'] = -1
        g.cells['_area'] = np.nan # would be nice if cell_defaults fixed this.
        for col in cells.columns:
            if col not in ['deleted','faces']:
                # blindly attempt adding a column
                try:
                    g.add_cell_field(col,cells[col].values)
                except Exception as exc:
                    print("Failed to add cell column %s"%col)
                
        for i,rec in cells.iterrows():
            if rec['deleted']: continue
        
            cell_halfedges = rec['faces']
            cell_edges = [f['fIdx'] for f in cell_halfedges]
            g.cells['edges'][i,:len(cell_edges)] = cell_edges
            cell_nodes = [g.edges['nodes'][f['fIdx'],f['orient']]
                     for f in cell_halfedges]
            g.cells['nodes'][i,:len(cell_nodes)] = cell_nodes
    return g

class GridEncoder(json.JSONEncoder):
    exclude=[]
    cell_halfedges=True
    cell_nodes=True
    def default(self,o,strip_negative=[]):
        if isinstance(o,unstructured_grid.UnstructuredGrid):
            # extra handling for cells - unstructured_grid is node
            # centric, but C# code is half-edge based.
            # 
            return dict(Cells=self.convert_cells(o,strip_negative=strip_negative),
                        Faces=o.edges,
                        Nodes=self.convert_nodes(o))
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

    def convert_cells(self,o,strip_negative=[]):
        cells=o.cells        
        df=pd.DataFrame()
        for name in cells.dtype.names:
            if name in self.exclude: continue
            if name=='nodes' and not self.cell_nodes: continue
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
        
        
    

def write_json_mesh(grid,fn,exclude=[],cell_halfedges=True,cell_nodes=False):
    class CustomGridEncoder(GridEncoder): pass
    CustomGridEncoder.exclude=exclude
    CustomGridEncoder.cell_halfedges=cell_halfedges
    CustomGridEncoder.cell_nodes=cell_nodes
    
    with open(fn,"wt") as fp:
        json.dump(grid,fp,cls=CustomGridEncoder,skipkeys=True)
