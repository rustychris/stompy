
def write_points_vtu(df,fn):
    """
    Given a DataFrame with x,y,z fields
    Write an unstructured VTK file as points, and include the 
    remaining fields as point data
    """
    with open(fn,'wt') as fp:
        fp.write("\n".join(
            ['<?xml version="1.0"?>',
             '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
             '  <UnstructuredGrid>',
            f'    <Piece NumberOfPoints="{len(df)}" NumberOfCells="{len(df)}">',
             '      <PointData>\n']))
        # PointData tag can include  Scalars="Ve" Vectors="Velocity", etc. but I think this
        # is just for selecting an "active" field.
        
        for vari in df.columns:
            if vari in ['x','y','z']:
                continue
            fp.write(f'        <DataArray Name="{vari}" type="Float32" format="ascii">\n')
            fp.write(" ".join( ("%.4f"%x for x in df[vari].values) ) )
            fp.write('\n        </DataArray>\n')

        fp.write("\n".join([
            '      </PointData>',
            '      <CellData/>' # was separate open/close tags
        ]))

        fp.write( '      <Points>\n')
        fp.write( '       <DataArray NumberOfComponents="3" type="Float64" format="ascii">\n')
        fp.write( "  ".join( ("%.3f %.3f %.3f"%(p[0],p[1],p[2]) for p in df[ ['x','y','z'] ].values) ) )

        fp.write('\n</DataArray>\n  </Points>\n')

        fp.write('      <Cells>\n')
        fp.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        fp.write( "  ".join( ("%d"%x for x in range(len(df))) ) )
        fp.write('\n</DataArray>\n')

        fp.write('<DataArray type="Int32" Name="offsets" format="ascii">')
        fp.write( "  ".join( ("%d"%(x+1) for x in range(len(df))) ) )
        fp.write('</DataArray>\n')

        fp.write('<DataArray type="UInt8" Name="types" format="ascii">')
        fp.write( "  ".join( ("1" for x in range(len(df))) ) )
        fp.write('</DataArray>\n')

        fp.write("\n".join([
            '      </Cells>',
            '    </Piece>',
            '  </UnstructuredGrid>',
            '</VTKFile>\n'] ))
