import shapely.ops
from shapely import geometry

def cut_polygon(poly,line):
    # slice the exterior separately from interior, then recombine
    ext_sliced=poly.exterior.union( line )
    ext_poly=geometry.Polygon(poly.exterior)
    int_sliced=[ p.union(line)
                 for p in poly.interiors ]

    ext_parts, _dangles,_cuts,_invalids = shapely.ops.polygonize_full( ext_sliced )
    ext_parts=list(ext_parts) # so we can update entries

    # It's possible to introduce some new area here - places where the cut line
    # goes outside the exterior but forms a loop with the exterior.
    
    ext_parts=[p_ext
               for p_ext in ext_parts
               if p_ext.intersection(ext_poly).area / p_ext.area > 0.99 ]

    for p in int_sliced:
        int_parts, _dangles,_cuts,_invalids = shapely.ops.polygonize_full( p )
        # remove from an ext_part if there's overlap
        for p_int in int_parts:
            for i_ext, p_ext in enumerate(ext_parts):
                if p_ext.intersects(p_int):
                    ext_parts[i_ext] = p_ext.difference(p_int)

    return ext_parts
