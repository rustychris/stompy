import numpy as np
import matplotlib.pyplot as plt
from stompy import xr_transect

def test_transect():
    """
    Test loading a 'section_hydro' formatted transect, and resample to evenly
    spaced z.
    """
    untrim_sec=xr_transect.section_hydro_to_transect("data/section_hydro.txt",name="7B")

    new_z=np.linspace( untrim_sec.z_bed.values.min(),
                       untrim_sec.z_surf.values.max(),
                       100)

    untrim_sec_eq=xr_transect.resample_z(untrim_sec,new_z)

    plt.figure(22).clf()
    fig,axs=plt.subplots(2,1,num=22,sharex=True,sharey=True)

    xr_transect.plot_scalar(untrim_sec,'Ve',ax=axs[0])
    xr_transect.plot_scalar(untrim_sec_eq,'Ve',ax=axs[1])

    axs[0].text(0.02,0.9,'Original',transform=axs[0].transAxes)
    axs[1].text(0.02,0.9,'Resample',transform=axs[1].transAxes)



