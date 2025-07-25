# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:13:18 2024

@author: rusty
"""
import numpy as np
try:
    from stl import mesh
except ImportError:
    raise Exception("Need STL library, e.g. 'conda install numpy-stl'")
from shapely import geometry

import argparse
from .. import utils
from . import field
from ..grid import unstructured_grid


def rasterize_triangle(fld,X,Y,triple,op=np.fmax):
    """
    Write into field.SimpleGrid fld the triangle defined
    in triple [[x0,y0,z0],[x1,y1,z1],[x2,y2,z2]],
    overwriting existing information via the 
    operation op( existing array, new values array)
    """
    # Can crop fld to bounds of triple, and crop will be view
    # onto original data.
    
    # Start with expensive but functional approach:
    poly=geometry.Polygon(triple[:,:2])        
    msk=fld.polygon_mask(poly)
    
    x=X[msk]
    y=Y[msk]
    
    # Get the equation of the surface:
    # | t[0,0]  t[0,1] 1 | | a |   | t[0,2] |
    # | t[1,0]  t[1,1] 1 | | b | = | t[1,2] | 
    # | t[2,0]  t[2,1] 1 | | c | = | t[2,2] |
    M = triple.copy()
    M[:,2] = 1.0
    try:
        abc = np.linalg.solve(M,triple[:,2])
    except np.linalg.LinAlgError:
        print("Skip degenerate")
        return 
    z=abc[0]*x + abc[1]*y + abc[2]
    
    fld.F[msk] = op(fld.F[msk],z)


def transform(fp_stl,translate,R):
    translate=np.asarray(translate)
    xyz = fp_stl.points.copy().reshape([-1,3,3])
    xyz=np.tensordot(xyz,R,[2,0])
    xyz=xyz + translate
    normals = fp_stl.normals.dot(R)
    return xyz, normals

def stl_to_field(fp_stl,R,translate=[0,0,0],dx=None,dy=None,max_dim=None,pad=[0,0],nodata=np.nan,
                 cull_by_normals=True,
                 normal_tol=1e-4):
    xyz, normals = transform(fp_stl, translate, R)
    xxyy=[xyz[...,0].min()-pad[0], xyz[...,0].max()+pad[0],
          xyz[...,1].min()-pad[1], xyz[...,1].max()+pad[1]]
        
    if dx is not None:
        if dy is None: 
            dy=dx
    else:
        # with rounding there's some one-off in here, but I'm not worrying
        # about that right now.
        # try to be consistent with SimpleGrid.delta()
        # dx = (xmax-xmin)/(self.F.shape[1]-1.0)
        # i.e. xmin and xmax reference pixel centers
        # 
        dx=max( xxyy[1]-xxyy[0], xxyy[3]-xxyy[2] ) / (float(max_dim)-1.0)
        dy=dx

    # 1.0 to account for pixel centers, and 0.5 so int() will round to nearest 
    Z = np.full( (int(1.5 + (xxyy[3]-xxyy[2])/dx),
                  int(1.5 + (xxyy[1]-xxyy[0])/dx)),
                 np.float64(nodata))
    print("Output shape will be ",Z.shape)

    fld = field.SimpleGrid(extents=xxyy,F=Z)

    X,Y = fld.XY()
    
    for norm, triple in utils.progress(zip(normals,xyz)):
        # For starters handle the coordinate transfomation manually
        if cull_by_normals and norm[2]<=normal_tol: 
            continue # degen or facing away
        elif np.abs(norm[2])<=normal_tol:
            continue # degenerate 
        # rasterize a triangle into the field.
        rasterize_triangle(fld,X,Y,triple) 
        
    return fld

def stl_to_grid(fp_stl,R,translate=[0,0,0],dx=None,dy=None,max_dim=None,pad=[0,0],nodata=np.nan,
                cull_by_normals=True,
                normal_tol=1e-4):
    xyz,normals = transform(fp_stl,translate,R)

    # shove them into a grid to help with uniqueifying edges
    g=unstructured_grid.UnstructuredGrid()
    
    for norm, triple in utils.progress(zip(normals,xyz)):
        if cull_by_normals and norm[2]<=normal_tol: 
            continue # degen or facing away
        elif np.abs(norm[2])<=normal_tol:
            continue # degenerate 
        nodes,edges=g.add_linestring(triple[...,:2],closed=True,tolerance=1e-2) # handles unique nodes, edges
        
    print(f"Grid has {g.Nnodes()} nodes, {g.Nedges()} edges")
    # 0.0 tolerance: 60 165
    # 1e-2 58, 161
    # Further bump up the normal tolerance to 1e-8, 58 nodes, 160 edges

    return g


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rasterize_stl.py",
        description="Convert STL to georeferenced raster or shapefile")
    
    parser.add_argument("stl_file",help="Filename for STL input")
    parser.add_argument("out_file",help="Filename for GeoTIFF or shapefile output")
    parser.add_argument("--plot","-p",help="Plot result after conversion",action='store_true')
    
    parser.add_argument("--force","-f",help="Overwrite existing file",action='store_true')
    parser.add_argument("--proj",help="Spatial reference in GDAL text format, like EPSG:26910",default="")

    parser.add_argument("--cull",help="Disable triangle culling",action='store_false')

    parser.add_argument("--scale",nargs='+',help="Scale x [y z]",type=float)
    parser.add_argument("--rx",help="Rotate around x axis, degrees",type=float)
    parser.add_argument("--ry",help="Rotate around y axis, degrees",type=float)
    parser.add_argument("--rz",help="Rotate around z axis, degrees",type=float)
            
    parser.add_argument("--tx",help="Translate along x axis, post-scale units",type=float)
    parser.add_argument("--ty",help="Translate along y axis, post-scale units",type=float)
    parser.add_argument("--tz",help="Translate along z axis, post-scale units",type=float)

    parser.add_argument("--res",help="Resolution of output (tif output)",type=float)
    parser.add_argument("--size",help="Number of pixels in largest dimension (tif output)",type=int,default=1000)
    parser.add_argument("--pad",nargs=2,help="Pad the extent of the output DEM (tif output)",type=float,default=[0,0])
    parser.add_argument("--nodata",help="Value for missing pixels (tif output)",default=np.nan, type=float)
    
    args = parser.parse_args(argv)

    # Using an existing stl file:
    fp_stl = mesh.Mesh.from_file(args.stl_file)
    
    R = np.array( [[1,0,0],
                   [0,1,0],
                   [0,0,1]], np.float64)
    translate = np.array([0.0,0.0,0.0])
    
    # Arbitrary scaling and rotation. Switch y and z so it's more geographic
    if args.scale is not None:
        scale=args.scale
        if len(scale)==1:
            s=np.r_[scale[0],scale[0],scale[0]]
        elif len(scale)==3:
            s=np.array(args.scale)
        else:
            raise Exception("Scale only accepts 1 or 3 values.")
        R=R @ np.diag(s) 
        
    if args.rx is not None:
        theta=args.rx*np.pi/180
        Rrot = np.array( [[1,0,0],
                          [0,np.cos(theta),np.sin(theta)],
                          [0,-np.sin(theta),np.cos(theta)]])
        R = R @ Rrot
    if args.ry is not None:
        theta=args.ry*np.pi/180
        Rrot = np.array( [[np.cos(theta),0, -np.sin(theta)],
                          [0,            1, 0],
                          [np.sin(theta),0, np.cos(theta)]])
        R = R @ Rrot
    if args.rz is not None:
        theta=args.rz*np.pi/180
        Rrot = np.array( [[np.cos(theta),np.sin(theta),0],
                          [-np.sin(theta),np.cos(theta),0],
                          [0,0,1]])
        R = R @ Rrot    

    if args.tx is not None:
        translate[0] = args.tx
    if args.ty is not None:
        translate[1] = args.ty
    if args.tz is not None:
        translate[2] = args.tz
        
    kwargs={}
    if args.res is not None:
        kwargs['dx']=args.res
        kwargs['dy']=args.res
    else:
        kwargs['max_dim']=args.size

    kwargs['cull_by_normals'] = args.cull
        
    if args.out_file.lower().endswith('.tif'):
        output_type='tif'
    elif args.out_file.lower().endswith('.shp'):
        output_type='shp'
    else:
        raise Exception("Expected output to end with .tif or .shp")

    if output_type=='tif':
        fld = stl_to_field(fp_stl, R, translate=translate, 
                           pad=args.pad, **kwargs)
        fld.assign_projection(args.proj)
        fld.F = fld.F.astype(np.float32) # RAS2025 doesn't like doubles
        # RAS 6.6 defaults to DEFLATE. Stick with that for better compatibility
        fld.write_gdal(args.out_file,overwrite=args.force, nodata=args.nodata,
                       options=["COMPRESS=DEFLATE"])
    
        if args.plot:    
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(num=1,clear=1)
            ax.set_adjustable('datalim')
            img = fld.plot(ax=ax,interpolation='nearest')
            plt.colorbar(img)
            txt=[f"Input: {args.stl_file}",
                 f"Output: {args.out_file}",
                 f"Raster size: {fld.F.shape}"]
            ax.text(0.04,0.98,"\n".join(txt),va='top',size=7,transform=ax.transAxes)
            plt.show()
    elif output_type=='shp':
        g=stl_to_grid(fp_stl, R, translate=translate)
        g.write_edges_shp(args.out_file,srs_text=args.proj,overwrite=args.force)
        if args.plot:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(num=1,clear=1)
            ax.set_adjustable('datalim')
            coll = g.plot_edges(color='k',lw=1.0,alpha=0.8)
            txt=[f"Input: {args.stl_file}",
                 f"Output: {args.out_file}"]
            ax.text(0.04,0.98,"\n".join(txt),va='top',size=7,transform=ax.transAxes)
            ax.axis('equal')
            plt.show()

if __name__ == '__main__':
    main()
