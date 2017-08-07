from __future__ import print_function
# Explore some possibilities for optimizing the grid.

import sys

import numpy as np
from scipy.linalg import norm


from ..spatial import field

class OptimizeGridMixin(object):
    """ Meant to be mixed in with paver.Paving, to add some methods for optimizing a
    grid.
    """
    # if set, then scale reductions will be handled through an Apollonius Graph
    # otherwise, the density field is temporarily scaled down everywhere.
    use_apollo = False
    
    def __init__(self,*args,**kwargs):
        super(OptimizeGridMixin,self).__init__(*args,**kwargs)
        self.setup_optimizer()

    def setup_optimizer(self):
        # try:
        #     field.ApolloniusField
        #     self.use_apollo = True
        # except:
        #     print "No Apollonius Graph available"
        #     self.use_apollo = False
        
        self.original_density = self.density

        # These are the values that, as needed, are used to construct a reduced scale
        # apollonius field.  since right now ApolloniusField doesn't support insertion
        # and updates, we keep it as an array and recreate the field on demand.
        valid = np.isfinite(self.points[:,0])
        xy_min = self.points[valid].min(axis=0)
        xy_max = self.points[valid].max(axis=0)

        self.scale_reductions = np.array( [ [xy_min[0],xy_min[1],1e6],
                                            [xy_min[0],xy_max[1],1e6],
                                            [xy_max[0],xy_max[1],1e6],
                                            [xy_max[0],xy_min[1],1e6]] )

    apollo_rate = 1.1
    def update_apollonius_field(self):
        """ create an apollonius graph using the points/scales in self.scale_reductions,
        and install it in self.
        """
        if len(self.scale_reductions) == 0:
            self.apollo = None
            return
        
        self.apollo = field.ApolloniusField(self.scale_reductions[:,:2],
                                            self.scale_reductions[:,2],
                                            r=self.apollo_rate)
        self.density = field.BinopField( self.original_density,
                                         minimum,
                                         self.apollo )
        
        
    # Explore a cost function based on voronoi-edge distance
    def cell_scores(self,cell_ids=None,use_original_density=True):
        """ Return scores for each cell, based on the minimum distance from the
        voronoi center to an edge, normalized by local scale
        invalid cells (i.e. have been deleted) get inf score

        use_original_density: defaults to evaluating local scale using the
          original density field.  If set to false, use the current density
          field of the paver (which may be a reduced Apollonius Graph field)
        """
        if cell_ids is None:
            cell_ids = np.arange(self.Ncells())

        valid = (self.cells[cell_ids,0]>=0)

        vc = self.vcenters()[cell_ids]

        local_scale = np.zeros( len(cell_ids), np.float64)

        if use_original_density:
            local_scale[valid] = self.original_density( vc[valid,:] )
        else:
            local_scale[valid] = self.density( vc[valid,:] )
        local_scale[~valid] = 1.0 # dummy

        #
        cell_scores = np.inf*np.ones( len(cell_ids) )

        # 3 edge centers for every cell
        ec1 = 0.5*(self.points[self.cells[cell_ids,0]] + self.points[self.cells[cell_ids,1]])
        ec2 = 0.5*(self.points[self.cells[cell_ids,1]] + self.points[self.cells[cell_ids,2]])
        ec3 = 0.5*(self.points[self.cells[cell_ids,2]] + self.points[self.cells[cell_ids,0]])

        d1 = ((vc - ec1)**2).sum(axis=1)
        d2 = ((vc - ec2)**2).sum(axis=1)
        d3 = ((vc - ec3)**2).sum(axis=1)

        # could be smarter and ignore boundary edges..  later.
        # this also has the downside that as we refine the scales, the scores
        # get worse.  Maybe it would be better to compare the minimum ec value to
        # the mean or maximum, say (max(ec) - min(ec)) / med(ec)
        scores = np.sqrt(np.minimum(d1,d2,d3)) / local_scale
        
        scores[~valid] = np.inf
        
        return scores

    def relax_nodes(self,nodes):
        """ starting from the given set of nodes, relax in the area until the neighborhood score
        stops going down.
        """
        
        cells = set()
        for n in nodes:
            cells = cells.union( self.pnt2cells(n) )
        cells = np.array(list(cells))

        starting_worst = self.cell_scores(cells).min()
        worst = starting_worst

        while 1:
            cp = self.checkpoint()
            for n in nodes:
                self.safe_relax_one(n)
            new_worst = self.cell_scores(cells).min()
            
            sys.stdout.write('.') ; sys.stdout.flush()
            
            if new_worst < worst:
                #print "That made it even worse."
                self.revert(cp)
                new_worst = worst
                break

            if new_worst < 1.01*worst:
                #print "Not getting any better. ===> %g"%new_worst
                break
            worst = new_worst
            
        print("Relax: %g => %g"%(starting_worst,new_worst))
        self.commit()
        
        
    def cell_neighborhood_apollo(self,c):
        """ return the nodes near the given cell
        the apollo version means the termination condition is based on the
        expected radius of influence of a reduced scale at c
        """
        vcs = self.vcenters()
        c_vc = self.vcenters()[c]
        orig_scale = self.original_density(c_vc)
        apollo_scale = self.apollo(c_vc)
        r = (orig_scale - apollo_scale) / (self.apollo.r - 1)
        print("Will clear out a radius of %f"%r)
        
        c_set = set()

        def dfs(cc):
            if cc in c_set:
                return
            if norm(vcs[cc] - c_vc) > r:
                return
            
            c_set.add(cc)
            for child in self.cell_neighbors(cc):
                dfs(child)
        dfs(c)
        cell_list = np.array(list(c_set))
        
        return np.unique( self.cells[cell_list,:] )

    def node_neighborhood(self,n,nbr_count=2):
        """ return the nodes near the given node
        """
        c_set = set()

        def dfs(cc,i):
            if cc in c_set:
                return
            c_set.add(cc)
            if i > 0:
                for child in self.cell_neighbors(cc):
                    dfs(child,i-1)
        for c in self.pnt2cells(n):
            dfs(c,nbr_count-1)
        cell_list = np.array(list(c_set))
        
        return np.unique( self.cells[cell_list,:] )
        
    def cell_neighborhood(self,c,nbr_count=2):
        """ return the nodes near the given cell
        """
        c_set = set()

        def dfs(cc,i):
            if cc in c_set:
                return
            c_set.add(cc)
            if i > 0:
                for child in self.cell_neighbors(cc):
                    dfs(child,i-1)
        dfs(c,nbr_count)
        cell_list = np.array(list(c_set))
        
        return np.unique( self.cells[cell_list,:] )

    def relax_neighborhood(self,c,neighborhood_size=3):
        nbrnodes = self.cell_neighborhood(c,neighborhood_size)
        self.relax_nodes(nbrnodes)

    def relax_neighborhoods(self,score_threshold=0.1,count_threshold=5000,
                            neighborhood_size=2):
        """ find the worst scores and try just relaxing in the general vicinity
        """
        all_scores = self.cell_scores()

        ranking = np.argsort(all_scores)

        count = 0
        while 1:
            c = ranking[count]
            score = all_scores[c]
            if score > score_threshold:
                break
            self.relax_neighborhood(c,neighborhood_size)
            
            count += 1
            if count >= count_threshold:
                break

    def repave_neighborhoods(self,score_threshold=0.1,count_threshold=5000,
                             neighborhood_size=3,
                             scale_factor=None):
        """ find the worst scores and try just repaving the general vicinity

        see repave_neighborhood for use of scale_factor.

        if use_apollo is true, neighborhood size is ignored and instead the
        neighborhood is defined by the telescoping ratio
        """
        all_scores = self.cell_scores()

        ranking = np.argsort(all_scores)

        ## neighborhoods may overlap - and cells might get deleted.  Keep a
        #  record of cells that get deleted, and skip them later on.
        #  this could get replaced by a priority queue, and we would just update
        #  metrics as we go.
        expired_cells = {}
        def expire_cell(dc):
            expired_cells[dc] = 1
        cb_id = self.listen('delete_cell',expire_cell)

        if self.use_apollo and scale_factor is not None and scale_factor < 1.0:
            # Add reduction points for all cells currently over the limit
            to_reduce = np.nonzero(all_scores<score_threshold)[0]
            centers = self.vcenters()[to_reduce]
            orig_scales = self.original_density( centers )
            new_scales = scale_factor * orig_scales
            xyz = np.concatenate( [centers,new_scales[:,newaxis]], axis=1)
            self.scale_reductions = np.concatenate( [self.scale_reductions,xyz])
            print("Installing new Apollonius Field...")
            self.update_apollonius_field()
            print("... Done")

        count = 0
        while 1:
            c = ranking[count]
            print("Considering cell %d"%c)
            count += 1

            # cell may have been deleted during other repaving
            if self.cells[c,0] < 0:
                print("It's been deleted")
                continue

            if c in expired_cells:
                print("It had been deleted, and some other cell has taken its place")
                continue

            # cell may have been updated during other repaving
            # note that it's possible that this cell was made a bit better,
            # but still needs to be repaved.  For now, don't worry about that
            # because we probably want to relax the neighborhood before a second
            # round of repaving.
            if self.cell_scores(array([c]))[0] > all_scores[c]:
                continue
            
            score = all_scores[c]
            if score > score_threshold:
                break

            # also, this cell may have gotten updated by another repaving -
            # in which case we probably want

            print("Repaving a neighborhood")
            self.repave_neighborhood(c,neighborhood_size=neighborhood_size,scale_factor=scale_factor)
            print("Done")
            
            if count >= count_threshold:
                break
            
        self.unlisten(cb_id)

    # a more heavy-handed approach -
    # remove the neighborhood and repave
    def repave_neighborhood(self,c,neighborhood_size=3,scale_factor=None,nbr_nodes=None):
        """
        c: The cell around which to repave
        n_s: how big the neighborhood is around the cell

        scale_factor: if specified, a factor to be applied to the density field
          during the repaving.

        nbr_nodes: if specified, exactly these nodes will be removed (with their edges and
        the cells belonging to those edges).  otherwise, a neighborhood will be built up around
        c.
        """
        print("Top of repave_neighborhood - c = %d"%c)
        
        starting_score = self.cell_scores(array([c]))
        
        if nbr_nodes is None:
            if scale_factor is not None and self.use_apollo and scale_factor < 1.0:
                print("dynamically defining neighborhood based on radius of Apollonius Graph influence")
                nbr_nodes = self.cell_neighborhood_apollo(c)
            else:
                nbr_nodes = self.cell_neighborhood(c,neighborhood_size)

        # delete all non boundary edges going to these nodes
        edges_to_kill = np.unique( np.concatenate( [self.pnt2edges(n) for n in nbr_nodes] ) )
        # but don't remove boundary edges:
        # check both that it has cells on both sides, but also that it's not an
        # internal guide edge
        to_remove = (self.edges[edges_to_kill,4] >= 0) & (self.edge_data[edges_to_kill,1] < 0)
        edges_to_kill = edges_to_kill[ to_remove]

        for e in edges_to_kill:
            # print "Deleting edge e=%d"%e
            self.delete_edge(e,handle_unpaved=1)

        # the nodes that are not on the boundary get deleted:
        for n in nbr_nodes:
            # node_on_boundary includes internal_guides, so this should be okay.
            if self.node_on_boundary(n):
                # SLIDE nodes are reset to HINT so that we're free to resample
                # the boundary
                if self.node_data[n,self.STAT] == self.SLIDE:
                    # print "Setting node n=%d to HINT"%n
                    self.node_data[n,self.STAT] = self.HINT
            else:
                # print "Deleting node n=%d"%n
                self.delete_node(n)

        old_ncells = self.Ncells()

        saved_density = None
        if scale_factor is not None:
            if not ( self.use_apollo and scale_factor != 1.0):
                saved_density = self.density
                self.density = self.density * scale_factor
            
        print("Repaving...")
        self.pave_all(n_steps=inf) # n_steps will keep it from renumbering afterwards
        #print "Better?"

        if saved_density is not None:
            self.density = saved_density

        new_cells = np.arange(old_ncells,self.Ncells())
        new_scores = self.cell_scores(new_cells)
        
        print("Repave: %g => %g"%(starting_score,new_scores.min()))

    def full(self):
        self.verbose = 0

        self.relax_neighborhoods()
        self.repave_neighborhoods(neighborhood_size=2)
        self.relax_neighborhoods()
        self.repave_neighborhoods(neighborhood_size=2,scale_factor=0.9)
        self.repave_neighborhoods(neighborhood_size=2,scale_factor=0.8)
        self.repave_neighborhoods(neighborhood_size=3,scale_factor=0.8)
        self.repave_neighborhoods(neighborhood_size=3,scale_factor=0.75)
        self.relax_neighborhoods()

        
        for i in range(10):
            scores = self.cell_scores()
            print("iteration %d, %d bad cells"%(i, (scores<0.1).sum() ))
            self.write_complete('iter%02d.pav'%i)
            self.repave_neighborhoods(neighborhood_size=2,scale_factor=0.7)
            self.write_complete('iter%02d-repaved.pav'%i)
            self.relax_neighborhoods(neighborhood_size=5)

        self.stats()

    def stats(self):
        scores = self.cell_scores()

        print("Total cells with score below 0.1: %d"%( (scores<0.1).sum() ))
        return (scores<0.1).sum()
        
        
