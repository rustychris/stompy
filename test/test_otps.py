from __future__ import print_function

from stompy.model.otps import read_otps, otps_model
from stompy import utils
import numpy as np

import six
six.moves.reload_module(utils)
six.moves.reload_module(read_otps)
six.moves.reload_module(otps_model)

read_otps.OTPS_DATA="data"
local_bin_path="/home/rusty/src/OTPS2"

def test_download():
    model_file=read_otps.model_path('OR1km')
    print("Downloaded model to %s"%model)
    assert os.path.exists(model_file)

def test_read_otps():
    times=np.arange( np.datetime64('2010-01-01 00:00'),
                     np.datetime64('2010-01-10 00:00'),
                     np.timedelta64(15,'m') )

    modfile=read_otps.model_path('OR1km')

    pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=[235.25], lat=[44.5],
                                             time=times)


def test_otps_model():
    # modfile=read_otps.model_path('OR1km')
    modfile=read_otps.model_path('OhS')

    otps=otps_model.OTPS(bin_path=local_bin_path,
                         data_path="data",
                         model_file="DATA/" + os.path.basename(modfile))
    # lonlats=np.array( [[235.25, 44.5]] )
    lonlats=np.array( [ [145,35] ])

    consts=otps.extract_HC(lonlats,
                           constituents=['m2','s2','n2','k2','k1','o1','p1','q1'],
                           quant='z')

def test_compare():
    lonlats=np.array( [ [145,35] ])
    times=np.arange( np.datetime64('2010-01-01 00:00'),
                     np.datetime64('2010-01-10 00:00'),
                     np.timedelta64(15,'m') )

    modfile=read_otps.model_path('OhS')
    pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=lonlats[:,0],lat=lonlats[:,1],
                                             time=times)


    otps=otps_model.OTPS(bin_path=local_bin_path,
                         data_path="data",
                         model_file="DATA/" + os.path.basename(modfile))
    consts=otps.extract_HC(lonlats,
                           constituents=['m2','s2','n2','k2','k1','o1','p1','q1'],
                           quant='z')

    pred2_h=otps_model.reconstruct(consts,times)

    diffs=pred2_h.result.isel(site=0).values - pred_h[:,0]
    rms_err=np.sqrt(np.mean(diffs**2))
    print("RMS difference %.4f"%rms_err)
    assert rms_err<0.01


    if 0:
        import matplotlib.pyplot as plt
        plt.ion()

        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        ax.plot(times,pred_h,label='read_otps')
        ax.plot(pred2_h.time.values,pred2_h.result.isel(site=0),
                label='otps_model')

