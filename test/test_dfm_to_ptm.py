# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 05:21:30 2022

@author: rusty
"""
import six
from stompy.model.delft import dfm_to_ptm
six.moves.reload_module(dfm_to_ptm)

# Testing
#mdu_path="../flowfm.mdu"
# This one is giving some bizarre fluxes.
mdu_path="S:/Data/Hydro/UCD_CSC/DFM/runs/test_run2_dwaq_20160701/flowfm.mdu"
converter=dfm_to_ptm.DFlowToPTMHydro(mdu_path,'test_hydro.nc',time_slice=slice(0,10),
                                     grd_fn='test_sub.grd',overwrite=True)

#%%

# 105255 .. 105296 sequential, inclusive
converter.unmapped_faces # =np.nonzero(~valid)[0]
# mostly negative, all are substantial
converter.unmapped_fluxes # =Qs[~valid]

# These are indexes into the horizontal exchanges of flow.
# should be able to get those to links (probably 1:1, since it's a 1D run)
hyd=converter.hyd
# sgn is +1 for all of these
unmapped_links=hyd.exch_to_2d_link[converter.unmapped_faces]['link']

unmapped_elts=hyd.links[unmapped_links,1]

g=hyd.grid()
#%%
import matplotlib.pyplot as plt
fig,ax=plt.subplots(num=1,clear=1)
ax.set_adjustable('datalim')
g.plot_edges(lw=0.5,color='k',alpha=0.5)
def labeler(i,r):
    return str(converter.unmapped_fluxes[i==unmapped_elts])
g.plot_cells(mask=unmapped_elts,labeler=labeler)

# The locations still look like DCD
# but the fluxes at these points are large.
# is it possible that I'm pulling the wrong faces of these cells?
# like it's getting mixed up between the src/sink exchange and a real
# exchange, maybe 1-off?
# also, check the actual DCD inputs. maybe they are off.
