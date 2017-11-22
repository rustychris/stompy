from stompy.utils import *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


## 

def test_basic():
    # Test timeseries:
    t_h=np.arange(0,24*30,0.1) # 6 minute time steps in units of hours
    # And as np datetimes:
    t=np.datetime64("2000-01-01 00:00") + (3600*t_h)*np.timedelta64(1,'s')

    mod=xr.Dataset()
    mod['time']=('time',),t
    mod['x']=( ('time',),
               np.cos( 2*np.pi *t_h /12.4) + 0.2*np.cos( 2*np.pi *t_h /12.0) )

    obs=xr.Dataset()
    # give it a time shift:
    obs['time']=('time',),t+np.timedelta64(45,'m')
    obs['x']=( ('time',),
               (np.cos( 2*np.pi *t_h /12.4) + 0.2*np.cos( 2*np.pi *t_h /12.0) ) )
    # Add some noise to everybody
    obs['x'].values += 0.1*(np.random.random(t_h.shape) - 0.5)
    mod['x'].values += 0.1*(np.random.random(t_h.shape) - 0.5)

    # Make it shorter and coarser in time
    obs = obs.isel(time=slice(20,-100,3))

    lag=utils.find_lag_xr(mod.x,obs.x)

    print lag / np.timedelta64(1,'m')

    assert np.abs( lag - np.timedelta64(-45,'m')) < np.timedelta64(5,'m')

    try: 
        import matplotlib.pyplot as plt

        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        ax.plot(obs.time,obs.x,label='Obs')
        ax.plot(mod.time,mod.x,label='Mod')
        ax.plot(mod.time-lag,mod.x,label='Mod w/lag')

        ax.legend(fontsize=10)
    except ImportError:
        pass

