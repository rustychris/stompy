from stompy.grid import unstructured_grid, quad_laplacian, orthogonalize
import matplotlib.pyplot as plt
import numpy as np
import six

def to_bool_mask(grid,sel):
    sel=np.asarray(sel)
    if np.issubdtype(sel.dtype,np.int):
        bool_sel=np.zeros(grid.Ncells(),bool)
        bool_sel[sel]=True
        sel=bool_sel
    return sel
            
def trim_to_ij_convex_walk(grid,sel):
    """
    Probably something like searching for marked cells with two unmarked 
    neighbors. traverse down each side along the marked cells. If any of 
    those have a neighbor that "bumps out", unmark the original.
    probably do this with halfedges.
    """
    sel=sel.copy()
    e2c=grid.edge_to_cells_reflect()

    candidates=np.nonzero(sel)[0]
    
    while True:
        dirty=False
        # count of unmarker neighbors for each selected cell
        problems=[]
        for c in candidates: #  np.nonzero(sel)[0]:
            if not sel[c]: continue # over-zealous queueing below
            nbrs=grid.cell_to_cells(c)
            nbrs=[nbr for nbr in nbrs if nbr>=0]
            unmarked_nbrs=(~sel[nbrs]).sum()
            if unmarked_nbrs>=2:
                problems.append(c)
                
        candidates=[]

        while problems:
            problem=problems.pop(0)
            unmark=False
            for nbr in grid.cell_to_cells(problem):
                if nbr<0: continue
                if sel[nbr]: continue
                nbr_nbrs=[n for n in grid.cell_to_cells(nbr) if (n>=0) and sel[n]]
                if len(nbr_nbrs)<2: continue
                unmark=True
                #grid.plot_cells(mask=nbr_nbrs,color='g')
                break
            if unmark:
                dirty=True
                sel[problem]=False

                # queue neighbors. These will be fully checked, so no need
                # to be too careful here
                for nbr in grid.cell_to_cells(problem):
                    if (nbr>=0) and (sel[nbr]):
                        candidates.append(nbr)
                
                if 0:
                    # would be better to queue a neighbor!
                    # as is, it has bad complexity.
                    plt.clf()
                    grid.plot_edges(color='k',zorder=2,lw=0.7)
                    grid.plot_cells(mask=sel,color='0.9',zorder=0)
                    if len(problems):
                        grid.plot_cells(mask=problems,zorder=1)
                    plt.axis( (647100., 647278., 4185764., 4185913.))
                    plt.pause(0.2)
        if not dirty:
            break
    # one option:
    #  for each problem cell:
    #    for each of its unmarked neighbors:
    #      if its marked nbr count>1:
    #        unmark problem cell
    return sel

def partition(grid,cells):
    """
    delete cells from grid, and grid and a new grid with only 
    those cells. Preserves node indices in both (though some nodes
    will be deleted)
    """
    assert np.issubdtype(cells.dtype,bool)

    g_refine=grid.copy()
    for c in np.nonzero( (~cells) & (~g_refine.cells['deleted']))[0]:
        g_refine.delete_cell(c)
    g_refine.delete_orphan_edges() 
    g_refine.delete_orphan_nodes() 

    for c in np.nonzero(cells & (~grid.cells['deleted']))[0]:
        grid.delete_cell(c)
    grid.delete_orphan_edges()
    grid.delete_orphan_nodes()
    return grid,g_refine


def splice_with_doubling(grid,refined):
    # Can figure out merge nodes directly, w/o spatial matching
    # I think this is symmetric, doesn't matter which grid is finer resolution.
    # May have to think about whether one of the grids has new nodes.

    merge_nodes=[] # refined has way too many nodes
    for n in refined.valid_node_iter():
        if n<grid.Nnodes() and not grid.nodes['deleted'][n]:
            # Extra check in case nodes have been added and re-used an index.
            if np.all( refined.nodes['x'][n] == grid.nodes['x'][n]):
                merge_nodes.append(n)

    # Start with refinement-ignorant splice 
    node_map,edge_map,cell_map = grid.add_grid(refined,merge_nodes=zip(merge_nodes,merge_nodes))

    def splice_refined_edge(j):
        # Identify these nodes:

        # n2-----A
        # |      |
        # C      |
        # |      |
        # n1-----B

        n1,n2 = grid.edges['nodes'][j]
        j_cells=grid.edge_to_cells(j)
        if j_cells[0]<0:
            c_coarse=j_cells[1]    
        elif j_cells[1]<0:
            c_coarse=j_cells[0]
            n1,n2=n2,n1 # match diagram below
        else:
            assert False
        assert c_coarse>=0

        coarse_nodes=grid.cell_to_nodes(c_coarse)
        # If the refined area is concave this might fail!
        assert len(coarse_nodes)==4,"Polygon may not be convex in ij space?"
        n1i=list(coarse_nodes).index(n1)
        B=coarse_nodes[ (n1i+1)%4 ]
        A=coarse_nodes[ (n1i+2)%4 ]
        assert n2==coarse_nodes[ (n1i+3)%4 ]

        C_candidates=set(grid.node_to_nodes(n1)) & set(grid.node_to_nodes(n2)) 
        assert len(C_candidates)==1
        C=list(C_candidates)[0]

        # Now we can do the operations
        grid.delete_edge_cascade(j)
        grid.add_cell_and_edges([n2,C,A])
        grid.add_cell_and_edges([C,B,A])
        grid.add_cell_and_edges([C,n1,B])
    # End splicer

    mergers=set(merge_nodes)
    # look for edges that need some TLC
    j_fixed={}
    for n in merge_nodes:
        for j in grid.node_to_edges(n):
            n1,n2 = grid.edges['nodes'][j]
            if not ((n1 in mergers) and (n2 in mergers)):
                continue
            if j in j_fixed:
                continue
            if np.all(grid.edges['cells'][j]>=0):
                j_fixed[j]=1
                continue
            splice_refined_edge(j)
            j_fixed[j]=1

    # Use node_map to report the new nodes
    return grid,node_map

class Refiner:
    def __init__(self,grid,cells,direction='both'):
        cells=to_bool_mask(grid,cells)

        cells=trim_to_ij_convex_walk(grid,cells)
        
        grid,g_refine = partition(grid,cells)
        
        refined=self.refine(g_refine, direction)

        grid,new_nodes=splice_with_doubling(grid,refined)
        self.result=grid
        self.new_nodes=new_nodes[ new_nodes>=0 ]
    def refine(self,grid,direction):
        """
        double the cells in refined along given direction and return
        the result (which may be the same object as g_coarse, and is
        guaranteed to have the same node numbers for nodes that are original).
        """
        nodes=np.array([c for c in grid.valid_node_iter()])
        node_idxs,node_ij = grid.select_quad_subset(ctr=None,node_set=nodes)

        n_to_ij={ n:ij for n,ij in zip(node_idxs,node_ij) }
        ij2_to_n= {(2*i,2*j):n for n,(i,j) in zip(node_idxs,node_ij)}

        ij_min=node_ij.min(axis=0)
        ij_max=node_ij.max(axis=0)
        ij_range=ij_max - ij_min

        # Will double i,j, and then use i2_step/j2_step
        # to determine stride when making cells
        i2_step=1 
        j2_step=1
        if direction=='both':
            pass
        elif direction=='long':
            if ij_range[0]>ij_range[1]:
                j2_step=2
            else:
                i2_step=2
        elif direction=='lat':
            if ij_range[0]<ij_range[1]:
                j2_step=2
            else:
                i2_step=2

        def ij2_to_xy(ij2):
            i2,j2=ij2
            if (i2%2) and (j2%2):
                ij2s=[ (i2-1,j2-1),
                       (i2+1,j2-1),
                       (i2+1,j2+1),
                       (i2-1,j2+1) ]
            elif i2%2:
                ij2s=[ (i2-1,j2),
                      (i2+1,j2) ]
            elif j2%2:
                ij2s=[ (i2,j2-1),
                      (i2,j2+1) ]
            else:
                assert False
            nodes=[ij2_to_n[ij2] for ij2 in ij2s]
            xys=grid.nodes['x'][nodes]
            return xys.mean(axis=0)
                
        # Will replace all cells and some edges
        # In order to preserve any holes or weird boundaries
        # use the old cells to guide where new cells are created
        # rather than relying on nodes
        constructed={} # ordered tuple of nodes to an interpolated node
        def get_constructed(tupe):
            tupe=list(tupe)
            tupe.sort()
            tupe=tuple(tupe)
            if tupe not in constructed:
                new_x=grid.nodes['x'][list(tupe)].mean(axis=0)
                n=grid.add_node(x=new_x)
                constructed[tupe]=n
            return constructed[tupe]
            
        for c in grid.valid_cell_iter():
            c_nodes=grid.cell_to_nodes(c)
            if c==3734:
                breakpoint()
                
            grid.delete_cell(c)

            # Get the nodes in a canonical order
            c_ij=np.array([n_to_ij[n] for n in c_nodes])
            c_ij_min=c_ij.min(axis=0)
            start_idx=np.nonzero(np.all(c_ij==c_ij_min,axis=1))[0][0]
            i2_ll=2*c_ij[start_idx,0]
            j2_ll=2*c_ij[start_idx,1]

            a,b,c,d=np.concatenate( [c_nodes[start_idx:],
                                     c_nodes[:start_idx]] )
            # d----c
            # |    |
            # |    |
            # a----b

            # enumerate the new cells by node (int) or node construction (tuple)
            if i2_step==1 and j2_step==1:
                new_cells=[ [a,(a,b),(a,b,c,d),(a,d)],
                            [(a,b),b,(b,c),(a,b,c,d)],
                            [(a,d),(a,b,c,d),(c,d),d],
                            [(a,b,c,d),(b,c),c,(c,d)] ]
            elif i2_step==2: # guessing which way is which
                new_cells=[ [a,b,(b,c),(a,d)],
                            [(a,d),(b,c), c,d] ]
            elif j2_step==2:
                new_cells=[ [a,(a,b),(c,d),d],
                            [(a,b),b,c,(c,d)] ]
            else:
                assert False
                
            for nodes in new_cells:
                new_nodes=[]
                for node in nodes:
                    if isinstance(node,tuple):
                        node=get_constructed(node)
                    else:
                        pass
                    new_nodes.append(node)
                            
                grid.add_cell_and_edges(new_nodes)
        grid.delete_orphan_edges()
        return grid
        

# Coarsen in one direction
class Coarsener:
    def __init__(self,grid,cells,direction):
        self.grid=grid
        self.direction=direction

        cells=to_bool_mask(grid,cells)

        orig_cells=cells
        # I *think* it's better to flip the flags here, but not sure.
        cells=~trim_to_ij_convex_walk(grid,~cells)

        grid,g_coarse = partition(grid,cells)

        g_coarse=self.coarsen_new(g_coarse,direction)
        g_coarse.delete_orphan_nodes()

        grid,new_nodes=splice_with_doubling(grid,g_coarse)
        self.result=grid
        self.new_nodes=new_nodes[ new_nodes>=0 ]
        
    def coarsen(self,g_coarse,direction):
        """
        coarsen the cells in g_coarse along given direction and return
        the result (which may be the same object as g_coarse)
        """
        #nodes=np.unique( np.concatenate( [g_coarse.cell_to_nodes(c) for c in np.nonzero(cells)[0]]) )
        nodes=np.array([c for c in g_coarse.valid_node_iter()])
        node_idxs,node_ij = g_coarse.select_quad_subset(ctr=None,node_set=nodes)

        ij_min=node_ij.min(axis=0)
        ij_max=node_ij.max(axis=0)
        ij_range=ij_max - ij_min

        # could provide option to offset the start by 1.
        # might have more stitching to do in that case.
        i_start=0
        j_start=0
        i_step=2
        j_step=2
        if direction=='both':
            pass
        elif direction=='long':
            if ij_range[0]>ij_range[1]:
                j_step=1
            else:
                i_step=1
        elif direction=='lat':
            if ij_range[0]<ij_range[1]:
                j_step=1
            else:
                i_step=1

        # Will replace all cells and edges
        # But if we drop a row (odd count), need to add it back with
        # original size.
        # Maybe do this on demand.
        #for j in g_coarse.valid_edge_iter():
        #    g_coarse.delete_edge_cascade(j)

        ij_to_n= {tuple(ij):n for n,ij in zip(node_idxs,node_ij)}

        # this would be better with a method like Refiner --
        # go by cells. As is, it will probably fill holes, and it 
        # assumes that ij_to_n is uniquely determined.
        for i in range(ij_min[0]+i_start, ij_max[0]+1, i_step):
            for j in range(ij_min[1]+j_start, ij_max[1]+1, j_step):
                # so i,j gives the nodes of one corner.
                ij_corners=[ (i,j),
                             (i+i_step,j),
                             (i+i_step,j+j_step),
                             (i,j+j_step) ]
                nodes=np.array([ ij_to_n.get(ij_corner,-1)
                                 for ij_corner in ij_corners] )
                if np.any(nodes<0):
                    continue

                # on-demand clear out the old edges/cells:
                is_dense=True
                to_remove=[]
                for ii in range(i,i+i_step+1):
                    for jj in range(j,j+j_step+1):
                        n_to_del=ij_to_n.get( (ii,jj), -1)
                        if n_to_del<0:
                            is_dense=False
                            break
                        if g_coarse.nodes['deleted'][n_to_del]:
                            continue # somebody beat us there
                        if n_to_del in nodes:
                            continue # keep this one
                        to_remove.append(n_to_del)
                if not is_dense:
                    continue # maybe a hole in the original
                for n in to_remove:
                    g_coarse.delete_node_cascade(n)
                    
                g_coarse.add_cell_and_edges(nodes)
        return g_coarse

    def coarsen_new(self,g_coarse,direction):
        """
        coarsen the cells in g_coarse along given direction and return
        the result (which may be the same object as g_coarse)
        """
        nodes=np.array([c for c in g_coarse.valid_node_iter()])
        node_idxs,node_ij = g_coarse.select_quad_subset(ctr=None,node_set=nodes)

        ij_min=node_ij.min(axis=0)
        ij_max=node_ij.max(axis=0)
        ij_range=ij_max - ij_min

        # could provide option to offset the start by 1.
        # might have more stitching to do in that case.
        i_start=0
        j_start=0
        i_step=2
        j_step=2
        if direction=='both':
            pass
        elif direction=='long':
            if ij_range[0]>ij_range[1]:
                j_step=1
            else:
                i_step=1
        elif direction=='lat':
            if ij_range[0]<ij_range[1]:
                j_step=1
            else:
                i_step=1

        # Will replace all cells and edges
        # But if we drop a row (odd count), need to add it back with
        # original size.

        ij_to_n= {tuple(ij):n for n,ij in zip(node_idxs,node_ij)}

        def coarsen_one(iA,iB,jA,jB):
            # so i,j gives the nodes of one corner.
            ij_corners=[ (iA,jA),
                         (iB,jA),
                         (iB,jB),
                         (iA,jB) ]
            nodes=np.array([ ij_to_n.get(ij_corner,-1)
                             for ij_corner in ij_corners] )
            if np.any(nodes<0):
                return

            # on-demand clear out the old edges/cells:
            is_dense=True
            to_remove=[]
            for ii in range(iA,iB+1):
                for jj in range(jA,jB+1):
                    n_to_del=ij_to_n.get( (ii,jj), -1)
                    if n_to_del<0:
                        is_dense=False
                        break
                    if g_coarse.nodes['deleted'][n_to_del]:
                        continue # somebody beat us there
                    if n_to_del in nodes:
                        continue # keep this one
                    to_remove.append(n_to_del)
            if not is_dense:
                return # maybe a hole in the original

            for n in to_remove:
                g_coarse.delete_node_cascade(n)

            g_coarse.add_cell_and_edges(nodes)
        
        # this would be better with a method like Refiner --
        # go by cells. As is, it will probably fill holes, and it 
        # assumes that ij_to_n is uniquely determined.
        for i in range(ij_min[0]+i_start, ij_max[0]+1, i_step):
            iA=i
            # try to deal with odd rows/cols
            iB=i+i_step
            if iA<ij_max[0] and iB>ij_max[0]:
                iB=ij_max[0]
                
            for j in range(ij_min[1]+j_start, ij_max[1]+1, j_step):
                jA=j
                jB=j+j_step
                if jA<ij_max[1] and jB>ij_max[1]:
                    jB=ij_max[1]
                coarsen_one(iA,iB,jA,jB)

        # And cleanup on odd rows/columns. There are some nuanced assumptions
        # that we're operating on a nice rectangular patch. Whole method needs
        # to be rethinked. another day.
        # breakpoint()
                
        return g_coarse



