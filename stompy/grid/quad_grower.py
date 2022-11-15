from . import orthogonalize
from .. import utils
import numpy as np
from shapely import geometry

def ccw(v):
    return np.r_[ -v[1], v[0] ]

class QuadGrower:
    node_history_count=20
    
    def __init__(self,grid,nodes=None,seed_point=None, max_nodes=-1,
                 remove_fillets=True):
        self.grid=grid
        self.seed_point=seed_point
        self.tweaker=orthogonalize.Tweaker(self.grid)
        
        if nodes is None:
            assert seed_point is not None
            nodes=self.grid.enclosing_nodestring(seed_point,max_nodes)
        self.nodes=nodes

        if remove_fillets:
            print("Removing fillets")
            self.remove_fillets()
            
        if len(self.nodes)<3:
            raise Exception('QuadGrower: failed to find the hole')

        self.xy_shore=self.grid.nodes['x'][self.nodes]
        self.poly_shore=geometry.Polygon(self.xy_shore)

        # plot_wkb.plot_wkb(poly_shore, facecolor='none', edgecolor='g',lw=2)
        self.new_nodes=[]
        self.setup()

    def remove_fillets(self):
        """
        Try again, but keep halo tighter and make the check more explicit
        """
        node_halo=np.unique( np.concatenate([self.grid.node_to_nodes(n) for n in self.nodes] ) )
        new_nodes=list(self.nodes) # for updates

        for a,b in utils.circular_pairs(self.nodes):
            j=self.grid.nodes_to_edge(a,b)
            assert j>=0
            cells=self.grid.edge_to_cells(j)
            blank=cells.min()
            opp  =cells.max()
            assert blank<0
            if opp<0:
                continue # free edge, just for containing the new cells

            opp_nodes=self.grid.cell_to_nodes(opp)
            if len(opp_nodes)==3:
                print(" potential fillet")
                he=self.grid.halfedge(j,0)
                if he.cell()!=opp:
                    he=he.opposite()
                assert he.cell()==opp

                # Is this a triangle with two quads?
                c1=he.fwd().cell_opp()
                c2=he.rev().cell_opp()

                if (self.grid.cell_Nsides(c1)==4) and (self.grid.cell_Nsides(c2)==4):
                    print(" fillet!")
                    self.grid.delete_edge_cascade(j)
                    # update new_nodes to put the extra node in between a,b
                    n_add = [n for n in opp_nodes if (n!=a) and (n!=b)][0]
                    ia=new_nodes.index(a)
                    new_nodes.insert(ia+1,n_add)
                else:
                    print("Beware -- potential fillet but it wasn't flanked by quads.")
                    
        self.nodes=np.array(new_nodes)
        
    def setup(self):
        self.setup_ij()
        self.setup_norm_dir()
        self.setup_active()        
        
    def setup_ij(self):        
        # I think one halo should be sufficient, but need a bit more when there are fillets
        # so the quad search can get "around" the fillets
        node_halo=np.unique( np.concatenate([self.grid.node_to_nodes(n) for n in self.nodes] ) )

        (self.node_idxs,self.ijs) = self.grid.select_quad_subset(ctr=None,node_set=node_halo)

        self.n_to_ij={n:ij for n,ij in zip(self.node_idxs,self.ijs)}
        self.ij_to_n={ tuple(self.n_to_ij[n]):n for n in self.n_to_ij}
        
    def setup_norm_dir(self):
        # Use the distribution of ij indices to choose a flood-fill direction in ij
        # space. Careful to reolve sign ambiguity
        tangent_dir=utils.principal_vec(self.ijs.astype(np.float64))
        norm_dir = utils.rot(np.pi/2,tangent_dir)

        # resolve sign ambiguity by looking at the scores of the original nodes
        # versus the halo
        boundary_ij=np.array( [ self.n_to_ij[n] for n in self.nodes if n in self.n_to_ij] )
        boundary_score= boundary_ij.dot(norm_dir).mean()
        total_score=self.ijs.dot(norm_dir).mean()

        # I want norm_dir to point towards the open space
        # So boundary score should be higher than total_score
        if boundary_score<total_score:
            norm_dir *= -1
        self.norm_dir=norm_dir

    def setup_active(self):
        print("call to setup_active")
        active_edges=[ self.grid.nodes_to_edge(a,b)
                       for a,b in utils.circular_pairs(self.nodes)
                       if (a in self.n_to_ij) and (b in self.n_to_ij) ]
        active_hes=[]
        for j in active_edges:
            he=self.grid.halfedge(j,0)
            if he.cell()>=0:
                he=he.opposite()
            assert he.cell()<0
            if he.cell_opp()<0:
                # just a bounding edge.
                continue
            active_hes.append(he)
        self.active_hes=active_hes
        
    def grow(self):
        # That's all on nodes.
        # Maybe the easiest thing is to make a heap of half edges that are
        # potential fill candidates.
        # choose the one with smallest score against norm_dir
        # confirm it is still unpaved, then either call quad to edge
        # or construct a node position based on relaxation.

        # So take the original node string, get edges, then narrow to the
        # ones that are part of the quad selection and have adjacent quads
        new_cells=[]
        while self.active_hes:
            c=self.grow_one()
            if c>=0:
                new_cells.append(c)

            for n in self.new_nodes:
                self.tweaker.nudge_node_orthogonal(n)
            if len(self.new_nodes)>self.node_history_count:
                self.new_nodes=self.new_nodes[-self.node_history_count:]
        return new_cells
    
    def grow_one(self):
        he=self.pop_he()
        if he is None:
            return -1
        return self.add_quad_on_halfedge(he)

    def n_score(self,n):
        return self.n_to_ij[n].dot(self.norm_dir)
    def he_score(self,he):
        return self.n_score(he.node_fwd()) + self.n_score(he.node_rev())

    def pop_he(self):
        # a heap would be faster, but can't be bothered right now.
        while self.active_hes:
            he_scores = [self.he_score(he) for he in self.active_hes]
            best=np.argmin(he_scores)
            he=self.active_hes[best]
            self.active_hes.remove(he)
            # Make sure still vaild
            if he.cell()<0:
                return he
        return None

    def ij_to_xy(self,ij):
        """
        When node for ij doesn't exist, use nearby existing nodes to 
        extrapolate a reasonable location for it.
        """
        # more direct attempt to extrapolate spacing
        # we're always building off one or more existing
        # quads.
        # 1. find all quads adjacent to our hole
        #    ij => [a,b,c,d] => node_to_edges => unique, limit to edges
        #    in n_to_ij, so they are part of the quad region.
        
        #    bc--b--ab
        #     |  |  |
        #     c--O--a
        #     |  |  |
        #    cd--d--ad

        # 2. average dxy / dij over those edges.

        # 3. from each of a,b,c,d that exist, use dxy/dij to construct a point
        #    from that node.
        # 4. average the results.
        i,j=ij

        a=self.ij_to_n.get( (i,j+1), -1)
        b=self.ij_to_n.get( (i+1,j), -1)
        c=self.ij_to_n.get( (i,j-1), -1)
        d=self.ij_to_n.get( (i-1,j), -1)
        # now a,b,c,d are all missing?? ij is 9,-15
        nbrs=[a,b,c,d]
        nbrs=[nbr for nbr in nbrs if nbr>=0]
        grid=self.grid
        
        edges = np.concatenate( [grid.node_to_edges(n)
                                 for n in nbrs] ).astype(np.int32)
        assert len(edges) == len(np.unique(edges)),"Pretty sure that holds"
        
        edges=[j for j in edges
               if ( (grid.edges['nodes'][j,0] in self.n_to_ij) and
                    (grid.edges['nodes'][j,1] in self.n_to_ij))]
        assert len(edges)>0

        dxy_is=[]
        dxy_js=[]
        
        for j in edges:
            j_n=grid.edges['nodes'][j]
            dij = self.n_to_ij[j_n[1]] - self.n_to_ij[j_n[0]]
            dxy = grid.nodes['x'][j_n[1]] - grid.nodes['x'][j_n[0]]

            if dij[0]==0: 
                dxy_js.append( dxy/dij[1] )
            elif dij[1]==0: 
                dxy_is.append( dxy/dij[0] )
            else:
                # some nonlocal connection. Ignore. Can happen when the bounding
                # edges go straight across with no intervening nodes.
                continue

        assert len(dxy_js)>0
        assert len(dxy_is)>0
        dxy_di=np.mean(dxy_is,axis=0)
        dxy_dj=np.mean(dxy_js,axis=0)

        new_xy_ests=[]
        for nbr in nbrs:
            dij=ij - self.n_to_ij[nbr]
            xy=grid.nodes['x'][nbr] + dij[0]*dxy_di + dij[1]*dxy_dj
            new_xy_ests.append(xy)

        new_xy=np.mean( new_xy_ests, axis=0)
        return new_xy

    def ij_to_node_or_add(self,ij):
        n=self.ij_to_n.get(tuple(ij),-1)
        if n<0:
            xy=self.ij_to_xy(ij)
            if not self.poly_shore.contains(geometry.Point(xy)):
                return -1
            n=self.grid.add_node(x=xy)
            # Update records for later iterations
            self.ij_to_n[tuple(ij)] = n
            self.n_to_ij[n] = np.array(ij) # array to be consistent with existing entries
            self.new_nodes.append(n)

        return n

    def add_quad_on_halfedge(self,he):
        # Need a strong method to add a quad based on given halfedge.
        # In order for this to be iterative, have to stay within the
        # structure of the ij-mapping.
        # I have the ordered nodes
        n_a=he.node_rev()
        n_b=he.node_fwd()
        ij_rev=self.n_to_ij[n_a]
        ij_fwd=self.n_to_ij[n_b]
        dij=ij_fwd-ij_rev

        # From this I can get the ij for the other two nodes I'll need
        # d------c
        # |      |
        # a------b

        # sometimes will exist, sometimes not.
        ij_c=ij_fwd + ccw(dij)
        ij_d=ij_c - dij

        # will return -1 if the point would be outside poly_shore
        n_c=self.ij_to_node_or_add(ij_c)
        n_d=self.ij_to_node_or_add(ij_d)

        if n_c<0 or n_d<0:
            return -1

        new_c=self.grid.add_cell_and_edges([n_a,n_b,n_c,n_d])

        for j in self.grid.cell_to_edges(new_c):
            cell_he=self.grid.halfedge(j,0)
            if cell_he.cell()<0:
                self.active_hes.append(cell_he)
            elif cell_he.cell_opp()<0:
                self.active_hes.append(cell_he.opposite())

        return new_c
