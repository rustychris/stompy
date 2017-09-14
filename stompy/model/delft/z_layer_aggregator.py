"""
Re-implement the z-level aggregation code to do a smarter mapping
of z-level cells to aggregated levels.


Goals:
1. Planform areas with a water column are all the same.  This is necessary
   to have the bed layer include the full complement of sediment interface.

Implying that the depth of the aggregated cell is the mean depth (V/A).

In the unaggregated grid, there are cells below this elevation, as well as water
columns shallower than the mean depth.

The mapping of unaggregated cells to aggregated layers is then slightly more complex
than just retaining the layer index k.

Lexsort cells based on (i) elevation of cell centroid and (ii) bed elevation.

That's all fine for segment values, but exchanges are little tricky.  

"""
from __future__ import print_function

import os

from collections import defaultdict

import numpy as np

from . import waq_scenario 

# class ZLayerAggregatorOriginal(waq_scenario.DwaqAggregator):
#     """ Aggregates in the horizontal, but preserves the z layers
#     exactly.  There is an option to group layers near the bed, though.
#     """
#     # bed layers with a thickness less than this in the first time step 
#     # will be lumped together
#     dzmin=0.0
# 
#     def init_seg_mapping(self):
#         """ the end product here is seg_local_to_agg, and entries in agg_seg
#         """
#         self.agg_seg=[]
#         self.agg_seg_hash={} # (k,elt) => index into agg_seg
# 
#         seg_local_to_agg=np.zeros( (self.nprocs,self.max_segs_per_proc),'i4') - 1
# 
#         # this is going to vary by the deepest layer in each region
#         # n_layers_for_elt=self.n_agg_layers
# 
#         for elt_i in range(len(self.elements)):
#             print "--- Processing aggregate element %d ---"%elt_i
# 
#             for p in range(self.nprocs):
#                 nc=self.open_flowgeom(p)
# 
#                 global_ids=nc.FlowElemGlobalNr[:]-1 # make 0-based
# 
#                 sel_elt=(self.elt_global_to_agg_2d[global_ids]==elt_i)
#                 if not np.any(sel_elt):
#                     print "This proc has no local elements in this aggregated element"
#                     continue
# 
#                 hyd=self.open_hyd(p)
#                 hyd.infer_2d_elements()
#                 seg_to_2d=hyd.seg_to_2d_element
# 
#                 # local_vols=hyd.volumes(hyd.t_secs[0])
#                 # iterate over columns, so we can calculate bed depth at the same time.
#                 # elt_areas=nc.FlowElem_bac[:]
# 
#                 for local_elt in np.nonzero(sel_elt)[0]: # Loop over local water columns
#                     # these are all the segments which fall within this aggregated
#                     # segment, for this processor.
#                     segs=np.nonzero( (seg_to_2d==local_elt) )[0]
# 
#                     for seg in segs: # k_agg in range(n_layers_for_elt):
#                         agg_seg = self.get_agg_segment(agg_k=hyd.seg_k[seg],agg_elt=elt_i)
#                         seg_local_to_agg[p,seg] = agg_seg
# 
#         # and done
#         self.seg_local_to_agg=seg_local_to_agg
#                     
#     def planform_areas(self):
#         """ 
#         Return a Parameter object encapsulating variability of planform 
#         area.  Typically this is a per-segment, constant-in-time 
#         parameter, but it could be globally constant or spatially and 
#         temporally variable.
#         Old interface returned Nsegs * 'f4', which can be recovered 
#         in simple cases by accessing the .data attribute of the
#         returned parameter
# 
#         HERE: when bringing in z-layer data, this needs some extra 
#         attention.  In particular, planform area needs to be (i) the same 
#         for all layers, and (ii) should be chosen to preserve average depth,
#         presumably average depth of the wet part of the domain?  For now,
#         it's average depth over both wet and dry parts of the domain.
#         """
# 
#         # This is the old code - just maps maximum area from the grid
#         # with the new version of areas() below, probably the stock
#         # implementation of planform_areas() would be sufficient.
#         map2d3d=self.agg_seg_to_agg_elt_2d()
#         data=(self.elements['plan_area'][map2d3d]).astype('f4')
#         return waq_scenario.ParameterSpatial(data)
# 
#     def areas(self,t):
#         areas=super(ZLayerAggregatorOriginal,self).areas(t)
# 
#         # here we make all the areas equal to the max.
#         # it may be that we could deal with wetting and drying here, too.  not sure.
#         
#         # map vertical exchanges to 2d element
#         poi=self.pointers
#         self.infer_2d_elements()
# 
#         # use the to segment, since some from segments are boundary/negative
#         elt_for_exch_z=self.seg_to_2d_element[poi[:,1]-1]
#         elt_for_exch_z[:-self.n_exch_z] = -1 # limit to vertical
# 
#         for elt in range(self.n_2d_elements):
#             exch_sel=(elt_for_exch_z==elt)
#             areas[exch_sel]=areas[exch_sel].max()
#         
#         return areas


class ZLayerAggregator(waq_scenario.DwaqAggregator):
    """ Specialization of aggregation code to deal 
    with z-layer output, and more flexible mapping of unaggregated
    segments to aggregated segments

    when mode=='sort':

    sorts the segments and creates aggregated segments
    based on equal volume.  So far this is a bit problematic, as the
    exchanges end up skipping layers (i.e. vertical exchange from k=5 to
    k=2), also many vertical aggregated exchanges are actually made from
    fluxes which were horizontal in the unaggregated data.

    when mode=='original':
    
    all segments retain their original z level.  This often has
    the effect of having very thin layers near the bed, since only a small
    fraction of the elements have segments that deep.

    when mode=='combine_bed':
    
    like original, but for aggregated segments with a thickness (at time 0)
    less than dzmin, combine.  Only segments near the bed will be combined.
    
    """

    mode='sort'
    dzmin_m=0.0

    def init_seg_mapping(self):
        if self.mode=='original':
            return self.init_seg_mapping_original()
        elif self.mode=='sort':
            return self.init_seg_mapping_sort()
        elif self.mode=='combine_bed':
            return self.init_seg_mapping_combine_bed()
        else:
            assert False

    def init_seg_mapping_sort(self):
        self.agg_seg=[]
        self.agg_seg_hash={} # (k,elt) => index into agg_seg

        seg_local_to_agg=np.zeros( (self.nprocs,self.max_segs_per_proc),'i4') - 1

        n_layers_for_elt=self.n_agg_layers

        for elt_i in range(len(self.elements)):
            self.log.info("--- Processing aggregate element %d ---"%elt_i)

            # first two entries are for lexsort
            elt_segs=[] # [ (seg_z,bed_z,proc,local_seg,vol), ...]

            V_agg_elt=0.0 # accumulate total volume (approx by first time step)

            for p in range(self.nprocs):
                nc=self.open_flowgeom(p)

                global_ids=nc.FlowElemGlobalNr[:]-1 # make 0-based

                sel_elt=(self.elt_global_to_agg_2d[global_ids]==elt_i)
                if not np.any(sel_elt):
                    self.log.info("This proc has no local elements in this aggregated element")
                    continue

                hyd=self.open_hyd(p)
                hyd.infer_2d_elements()
                seg_to_2d=hyd.seg_to_2d_element

                local_vols=hyd.volumes(hyd.t_secs[0])
                # iterate over columns, so we can calculate bed depth at the same time.
                elt_areas=nc.FlowElem_bac[:]

                for local_elt in np.nonzero(sel_elt)[0]: # Loop over local water columns
                    segs=np.nonzero( (seg_to_2d==local_elt) )[0]
                    # print "Found %d segs in this water column"%len(segs)
                    # segs are already ordered surface->bed

                    Vsegs=local_vols[segs]
                    local_elt_area=elt_areas[local_elt]
                    V_agg_elt+=Vsegs.sum() # accumulate
                    D=Vsegs.sum()/local_elt_area

                    d=0
                    for seg,Vseg in zip(segs,Vsegs):
                        d+=Vseg/local_elt_area # depth of bottom of the segment
                        elt_segs.append( (d,D,p,seg,Vseg) )
            # /for p in range(self.nprocs)

            elt_segs=np.array(elt_segs)
            order=np.lexsort(elt_segs[:,1::-1].T)

            cumul_vol=np.cumsum(elt_segs[order,4])

            breaks=np.linspace(0,V_agg_elt,1+n_layers_for_elt)
            break_idxs=np.searchsorted(cumul_vol,breaks)
            break_idxs[-1]=len(elt_segs) # avoid 1-off error

            for k_agg in range(n_layers_for_elt):
                agg_seg = self.get_agg_segment(agg_k=k_agg,agg_elt=elt_i)
                segs_this_layer=elt_segs[order[break_idxs[k_agg]:break_idxs[k_agg+1]]]
                for seg in segs_this_layer:
                    d,D,p,seg,Vseg = seg
                    seg_local_to_agg[int(p),int(seg)] = agg_seg
        # and done
        self.seg_local_to_agg=seg_local_to_agg
                    
    def planform_areas(self):
        """ 
        Return a Parameter object encapsulating variability of planform 
        area.  Typically this is a per-segment, constant-in-time 
        parameter, but it could be globally constant or spatially and 
        temporally variable.
        Old interface returned Nsegs * 'f4', which can be recovered 
        in simple cases by accessing the .data attribute of the
        returned parameter

        HERE: when bringing in z-layer data, this needs some extra 
        attention.  In particular, planform area needs to be (i) the same 
        for all layers, and (ii) should be chosen to preserve average depth,
        presumably average depth of the wet part of the domain?
        """

        # This is the old code - just maps maximum area from the grid
        map2d3d=self.agg_seg_to_agg_elt_2d()
        data=(self.elements['plan_area'][map2d3d]).astype('f4')
        return waq_scenario.ParameterSpatial(data)

    def areas(self,t):
        areas=super(ZLayerAggregator,self).areas(t)

        # here we make all the areas equal to the max.
        # it may be that we could deal with wetting and drying here, too.  not sure.
        
        # map vertical exchanges to 2d element
        poi=self.pointers
        self.infer_2d_elements()

        # use the to segment, since some from segments are boundary/negative
        elt_for_exch_z=self.seg_to_2d_element[poi[:,1]-1]
        elt_for_exch_z[:-self.n_exch_z] = -1 # limit to vertical

        for elt in range(self.n_2d_elements):
            exch_sel=(elt_for_exch_z==elt)
            areas[exch_sel]=areas[exch_sel].max()
        
        return areas

    def init_seg_mapping_original(self):
        """ the end product here is seg_local_to_agg, and entries in agg_seg
        """
        self.agg_seg=[]
        self.agg_seg_hash={} # (k,elt) => index into agg_seg

        seg_local_to_agg=np.zeros( (self.nprocs,self.max_segs_per_proc),'i4') - 1

        # this is going to vary by the deepest layer in each region
        # n_layers_for_elt=self.n_agg_layers

        for elt_i in range(len(self.elements)):
            print("--- Processing aggregate element %d ---"%elt_i)

            for p in range(self.nprocs):
                nc=self.open_flowgeom(p)

                global_ids=nc.FlowElemGlobalNr[:]-1 # make 0-based

                sel_elt=(self.elt_global_to_agg_2d[global_ids]==elt_i)
                if not np.any(sel_elt):
                    print("This proc has no local elements in this aggregated element")
                    continue

                hyd=self.open_hyd(p)
                hyd.infer_2d_elements()
                seg_to_2d=hyd.seg_to_2d_element

                # local_vols=hyd.volumes(hyd.t_secs[0])
                # iterate over columns, so we can calculate bed depth at the same time.
                # elt_areas=nc.FlowElem_bac[:]

                for local_elt in np.nonzero(sel_elt)[0]: # Loop over local water columns
                    # these are all the segments which fall within this aggregated
                    # segment, for this processor.
                    segs=np.nonzero( (seg_to_2d==local_elt) )[0]

                    for seg in segs: 
                        agg_k=hyd.seg_k[seg]
                        agg_seg = self.get_agg_segment(agg_k=agg_k,agg_elt=elt_i)
                        seg_local_to_agg[p,seg] = agg_seg

        # and done
        self.seg_local_to_agg=seg_local_to_agg

    def init_seg_mapping_combine_bed(self):
        self.agg_seg=[]
        self.agg_seg_hash={} # (k,elt) => index into agg_seg

        seg_local_to_agg=np.zeros( (self.nprocs,self.max_segs_per_proc),'i4') - 1

        # n_layers_for_elt=self.n_agg_layers

        for elt_i in range(len(self.elements)):
            self.log.info("--- Processing aggregate element %d ---"%elt_i)

            elt_segs=[] # [ (seg_k,bed_z,proc,local_seg,vol), ...]

            layer_vols=defaultdict(lambda: 0.0) # accumulate volume per original layer
            agg_elt_area=0.0 # accumulate surface planform area
            
            elt_segs=[] # [(proc,seg,seg_k),...]

            for p in range(self.nprocs):
                nc=self.open_flowgeom(p)

                global_ids=nc.FlowElemGlobalNr[:]-1 # make 0-based

                sel_elt=(self.elt_global_to_agg_2d[global_ids]==elt_i)
                if not np.any(sel_elt):
                    self.log.info("This proc has no local elements in this aggregated element")
                    continue

                hyd=self.open_hyd(p)
                hyd.infer_2d_elements()
                seg_to_2d=hyd.seg_to_2d_element

                local_vols=hyd.volumes(hyd.t_secs[0])
                elt_areas=nc.FlowElem_bac[:]

                for local_elt in np.nonzero(sel_elt)[0]: # Loop over local water columns
                    agg_elt_area += elt_areas[local_elt]
                    segs=np.nonzero( (seg_to_2d==local_elt) )[0]
                    for seg in segs:
                        layer_vols[ hyd.seg_k[seg] ] += local_vols[seg]
                        elt_segs.append( (p,seg,hyd.seg_k[seg]) )

            # get a list of layers and their thicknesses:
            Nk = 1+np.max(layer_vols.keys())
            all_dz=[ layer_vols[k] / agg_elt_area
                     for k in range(Nk) ]
            
            # these don't line up precisely with what delwaq reports, but pretty close.
            # maybe using a slightly different area, or volume taken from a different time
            # print "Layer thicknesses:"
            # print all_dz

            k_map=np.arange(Nk) # this is the no-op version - no combining

            # loop from the bottom up:
            dz_combined=0.0
            count=0
            groups=[]
            this_group=[]
            dzmin=self.dzmin_m
            for k in range(Nk)[::-1]:
                this_group.append(k)
                dz_combined+=all_dz[k]
                if dz_combined < dzmin:
                    continue # keep lumping
                else:
                    groups.append(this_group)
                    this_group=[]
                    dz_combined=0.0
                if all_dz[k]>=dzmin:
                    dzmin=0.0 # no more lumping
            if this_group:
                groups.append(this_group)
                this_group=[]
                    
            self.log.info("Lumped layers are (in reverse order):")
            self.log.info(str(groups))

            groups=groups[::-1]
            for agg_k,seg_ks in enumerate(groups):
                for seg_k in seg_ks:
                    k_map[seg_k]=agg_k

            self.log.info("k_map is then:")
            self.log.info(str(k_map))

            for p,seg,seg_k in elt_segs:
                agg_k=k_map[seg_k]
                agg_seg = self.get_agg_segment(agg_k=agg_k,agg_elt=elt_i)
                seg_local_to_agg[p,seg] = agg_seg

        # and done
        self.seg_local_to_agg=seg_local_to_agg
                    



