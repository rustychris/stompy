import xarray as xr
from stompy import xr_transect
import matplotlib.pyplot as plt

def test_transect_contour():
    tran=xr.open_dataset('data/temp-transect.nc')

    plt.figure(10).clf()
    fig,axs=plt.subplots(2,1,num=10,sharex=True,sharey=True)
    xr_transect.plot_scalar(tran,'temp',ax=axs[0])
    xr_transect.plot_scalar_pcolormesh(tran,'temp',ax=axs[1])

    xr_transect.contour(tran,'temp',np.linspace(5,27,28),
                        ax=axs[1],colors='k',linewidths=0.5)

