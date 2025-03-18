import pandas as pd
import os
import numpy as np

from . import unstructured_grid
from .. import utils

class RasToOpenFOAM:
    # to be specified by user
    #  { 'patchName':[[x0,y0],[x1,y1]], ... }
    #  to create patches from sections of the perimeter
    label_plines={}

    # this script tries to do something reasonable with cycles, but probably best
    # to invoke createPatch to do full renumbering
    cycles=[] # list of patch pairs, e.g. [ ['inlet','outlet'], ... ]

    # use labels on 2D edges to guide the patch creationg
    label_map=dict(bed=0,atmosphere=1,fixedWalls=2)

    extrusion = lambda x: np.linspace(0,1,11)
    
    def __init__(self,ras_mesh,**kw):
        utils.set_keywords(self,kw)
        self.ras_mesh = ras_mesh
        self.load_grid()
        
        self.set_edgelabels()
        self.generate_points()
        self.generate_faces()
        
    def load_grid(self):
        self.g = unstructured_grid.UnstructuredGrid.read_ras2d(self.ras_mesh)
        self.g.modify_max_sides(4) # initial grid is triangles only, but need to at least have quads for 3D faces

    def set_edgelabels(self):
        e2c=self.g.edge_to_cells()
        edge_labels=np.where(e2c[:,1]>=0, -1, self.label_map['fixedWalls'])

        label_count=len(self.label_map)
        
        for k in self.label_plines:
            sel_edges = self.g.select_edges_by_polyline(self.label_plines[k])
            self.label_map[k] = label_count
            label_count+=1
            edge_labels[sel_edges] = self.label_map[k]
        self.edge_labels = edge_labels
        
    def generate_points(self):
        self.xy_count = self.g.Nnodes()
        z_test = self.extrusion(self.g.nodes['x'][0])
        self.z_count = len(z_test)
        
        xy3d = np.tile(self.g.nodes['x'],[self.z_count,1])

        #z3d = np.repeat(z,Npoint2D)
        z3d = np.zeros( self.xy_count * self.z_count )
        for i,point in enumerate(self.g.nodes['x']):
            # terrible memory access pattern.
            z3d[i::self.xy_count] = self.extrusion(point)
            
        self.xyz3D = np.hstack( [xy3d,z3d[:,None]] )

    def generate_faces(self):
        # Construct the faces, implicit cells

        # First, the in-plane faces. Deal with -1 vertices afterwards
        faces=[]
        face_labels=[]
        face_cells=[]

        e2c = self.g.edge_to_cells()
        
        for layer in range(self.z_count):
            # face normals follow right-hand-rule, so natively these faces
            # will all have a +z normal
            # OF requires that boundary faces have their normal pointing out of the
            # domain. This implies that for interior faces, the "owner" is the cell
            # opposite the normal vector
            layer_faces=np.where( self.g.cells['nodes']<0, -1, self.g.cells['nodes'] + layer*self.xy_count )
            if layer==0:
                layer_faces=layer_faces[:,::-1]
            faces.append(layer_faces)
            if layer==0:
                label=self.label_map['bed']
            elif layer+1==self.z_count:
                label=self.label_map['atmosphere']
            else:
                label=-1
            face_labels.append(np.full(len(layer_faces),label))

            # cell owner/nbr
            owner_nbr=np.full( (len(layer_faces),2), -1)
            if layer==0:
                owner_nbr[:,0]=np.arange(self.g.Ncells())
                owner_nbr[:,1]=-1 # exterior, 'bed' patch
            else:
                owner_nbr[:,0]=np.arange(self.g.Ncells()) + (layer-1)*self.g.Ncells()
                if layer+1==self.z_count:
                    owner_nbr[:,1]=-1 # exterior, 'atmosphere' patch
                else:
                    owner_nbr[:,1]=np.arange(self.g.Ncells()) + layer*self.g.Ncells()
            face_cells.append( owner_nbr )

        # every edge in the 2D grid becomes Nlayer-1 3D faces
        for layer in range(self.z_count-1):
            # boundary convention for 2D grid is that left of the edge is interior
            # this should keep all boundary faces pointing out
            z_faces=np.concatenate( [self.g.edges['nodes'] + layer*self.xy_count,
                                     self.g.edges['nodes'][:,::-1] + (layer+1)*self.xy_count],
                                    axis=1)
            faces.append(z_faces)
            face_labels.append( self.edge_labels )

            owner_nbr = np.full((len(z_faces),2),-1)
            owner_nbr[:,0] = e2c[:,0] + layer*self.g.Ncells()
            owner_nbr[:,1] = np.where( e2c[:,1]>=0, e2c[:,1] + layer*self.g.Ncells(), -1)
            face_cells.append( owner_nbr )

        # sanity check
        for labels,owner_nbr in zip(face_labels,face_cells):
            assert len(labels)==len(owner_nbr)
            assert np.all( (labels<0) == (owner_nbr[:,1]>=0) )

        self.faces=np.concatenate(faces,axis=0)
        self.face_labels=np.concatenate(face_labels)
        self.face_cells = np.concatenate(face_cells,axis=0)

        # special handling for cyclic patches
        # seems the corresponding faces need to start on a matching vertex
        # In each pair of cyclic patches, one of them needs to have all of its
        # faces rotated -1
        for patchA,patchB in self.cycles:
            labelA = self.label_map[patchA]
            labelB = self.label_map[patchB]
            sel=self.face_labels==labelB
            self.faces[sel,:] = np.roll(self.faces[sel,:],-1,axis=1)

        # confirmed that this change avoids the checkMesh warning
        # Note that this does *not* necessarily order the faces properly
        # OF seems to expect that (a) the order of faces within the patch
        # match, and (b) the order of vertices within each face match.
        # The code above just ensures (b). I think createPatch will double check
        # this and also fix (a).

        # At this point there is a complete mesh description across faces, face_labels, face_cells, xyz3D
        # Some reordering of faces is required:
        #   Boundary faces must come last, and it appears they must be grouped by patch
        # face_order = np.argsort(face_labels)

        # OF further wants the faces in "upper triangular" order
        face_order = np.lexsort( [self.face_cells[:,1], self.face_cells[:,0], self.face_labels ] )

        self.faces=self.faces[face_order]
        self.face_labels=self.face_labels[face_order]
        self.face_cells=self.face_cells[face_order]

        self.interior_count = (self.face_labels<0).sum()
        assert np.all(self.face_labels[:self.interior_count]<0)
        assert np.all(self.face_labels[self.interior_count:]>=0)
    
    def write_openfoam(self,mesh_path):
        # Write out in OF format:

        patch_ids = np.unique(self.face_labels)
        patch_ids = patch_ids[ patch_ids>=0]
        name_map = {v:k for k,v in self.label_map.items()}

        cycle_map={}
        for a,b in self.cycles:
            cycle_map[a]=b
            cycle_map[b]=a

        if not os.path.exists(mesh_path):
            os.mkdir(mesh_path)

        with open(os.path.join(mesh_path,"boundary"),'wt',newline='\n') as fp:
            fp.write("""/*--------------------------------*- C++ -*----------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       polyBoundaryMesh;
            location    "constant/polyMesh";
            object      boundary;
        }
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        """)

            fp.write(f"{len(patch_ids)}\n")
            fp.write("(")
            for patch_num,patch_id in enumerate(patch_ids):
                start = np.searchsorted(self.face_labels,patch_id)
                stop  = np.searchsorted(self.face_labels,patch_id+1)

                name=name_map[patch_id]
                if name=='fixedWalls':
                    patch_type='wall'
                    # RH: don't fully understand the group stuff here.
                    extra="inGroups        2(fixedWallsGroup wall);"
                elif name=='bed':
                    patch_type='wall'
                    extra="inGroups        2(bedGroup wall);"
                elif name in cycle_map:
                    patch_type='cyclic'
                    extra=f"neighbourPatch {cycle_map[name]};"
                else:
                    patch_type='patch'
                    extra="// no extra settings"
                fp.write(f"""
            {name}
            {{
               type   {patch_type};
               nFaces  {stop-start};
               startFace {start};
               {extra}
            }}
        """)
            fp.write(")\n")

        with open(os.path.join(mesh_path,'cellZones'),'wt',newline='\n') as fp:
            fp.write("""/*--------------------------------*- C++ -*----------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       regIOobject;
            location    "constant/polyMesh";
            object      cellZones;
        }

        0()
        """)

        face_n_sides=(self.faces>=0).sum(axis=1)

        with open(os.path.join(mesh_path,'faces'),'wt',newline='\n') as fp:
            fp.write("""/*--------------------------------*- C++ -*-----------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            arch        "LSB;label=32;scalar=64";
            class       faceList;
            location    "constant/polyMesh";
            object      faces;
        }
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        """)
            fp.write(f"{len(self.faces)}\n(\n")

            if 0: # 171k in 5s
                for i in utils.progress(range(len(self.faces)),func=print):
                    face = self.faces[i]
                    n_sides=face_n_sides[i]
                    pnts=str(face[:n_sides])[1:-1] # drop square brackets
                    fp.write( f"{n_sides}({pnts})\n")
            if 1: # 1M+ in 6s
                df=pd.DataFrame(face_n_sides)
                df['open'] = '('
                for v in range(self.faces.shape[1]):
                    col=f"v{v}"
                    df[col] = self.faces[:,v]
                    df[col] = df[col].astype(str).str.replace('-1','')

                df['close'] = ')'
                df.to_csv(fp, sep=' ', header=False, index=False, lineterminator='\n')

            fp.write(")\n")

        with open(os.path.join(mesh_path,'faceZones'),'wt',newline='\n') as fp:
            fp.write("""/*--------------------------------*- C++ -*-----------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       regIOobject;
            location    "constant/polyMesh";
            object      faceZones;
        }

        0()

        """)

        with open(os.path.join(mesh_path,'owner'),'wt',newline='\n') as fp:
            fp.write("""
        /*--------------------------------*- C++ -*--------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       labelList;
            location    "constant/polyMesh";
            object      owner;
        }

        """)
            fp.write(f"{len(self.faces)}\n(\n")
            for c in self.face_cells[:,0]:
                fp.write(f"{c}\n")
            fp.write(")\n")

        with open(os.path.join(mesh_path,'neighbour'),'wt',newline='\n') as fp:
            fp.write("""
        /*--------------------------------*- C++ -*--------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       labelList;
            location    "constant/polyMesh";
            object      neighbour;
        }

        """)
            fp.write(f"{self.interior_count}\n(\n")
            for c in self.face_cells[:self.interior_count,1]:
                fp.write(f"{c}\n")
            fp.write(")\n")

        with open(os.path.join(mesh_path,'points'),'wt',newline='\n') as fp:
            fp.write("""
        /*--------------------------------*- C++ -*--------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       vectorField;
            location    "constant/polyMesh";
            object      points;
        }

        """)

            fp.write(f"{self.xyz3D.shape[0]}\n(\n")
            df=pd.DataFrame()

            df['x']=self.xyz3D[:,0]
            df['y']=self.xyz3D[:,1]
            df['z']=self.xyz3D[:,2]
            df['pre']='('
            df['post']=')'
            df = df[ ['pre','x','y','z','post'] ]
            df.to_csv(fp, sep=' ', header=False, index=False, lineterminator='\n')

            fp.write(")\n")

        with open(os.path.join(mesh_path,'pointZones'),'wt',newline='\n') as fp:
            fp.write("""
        /*--------------------------------*- C++ -*--------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       regIOobject;
            location    "constant/polyMesh";
            object      pointZones;
        }
        0()
        """)

if __name__ == '__main__':
    r2of = RasToOpenFOAM(ras_mesh = "../../RAS Models/Task 1/Bombac/Fish passage designs 20250306/Geometries/Bombac two repeat.h5",
                         label_plines={'inlet':[[-1.510, 2.2], [-1.510, 0.0]],
                                       'outlet':[[4.510, 2.2], [4.510, 0.0]]},
                         extrusion=lambda xy: (xy[0]-4.5)*(-0.0167) + np.linspace(0,1.8,37),
                         cycles=[])
    r2of.write_openfoam("../../../bombac-v01/constant/polyMesh.orig")

