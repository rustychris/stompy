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
    otps=otps_model.OTPS(bin_path=local_bin_path,
                         data_path="data",
                         model_file="DATA/" + os.path.basename(modfile))

    for quant in ['z','u','U']:
        # Get the read_otps prediction:
        if quant in ['z','u','v']:
            pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=lonlats[:,0],lat=lonlats[:,1],
                                                     time=times)
            if quant=='z':
                pred=pred_h[:,0]
            elif quant=='u':
                pred=pred_u[:,0]
            elif quant=='v':
                pred=pred_v[:,0]
            else:
                assert False
        elif quant in ['U','V']:
            _h,pred_U,pred_V=read_otps.tide_pred(modfile,lon=lonlats[:,0],lat=lonlats[:,1],
                                                 time=times,z=1.0)
            if quant=='U':
                pred=pred_U[:,0]
            elif quant=='V':
                pred=pred_V[:,0]
            else:
                assert False
        else:
            assert False

        consts=otps.extract_HC(lonlats,
                               constituents=['m2','s2','n2','k2','k1','o1','p1','q1'],
                               quant=quant)
        pred2=otps_model.reconstruct(consts,times).result.isel(site=0).values
        if quant in ['u','v']:
            # convert cm/s to m/s
            pred2 = pred2/100.0

        diffs=pred2 - pred
        R=np.corrcoef(pred2, pred)[0,1]
        scale=min(np.std(pred2),np.std(pred))

        rms_err=np.sqrt(np.mean(diffs**2))
        print("Comparison for quantity %s"%quant)
        print("RMS difference %.4f  relative %.4f%%"%(rms_err,100*rms_err/scale))
        print("R=%.4f"%R)
        assert rms_err/scale<0.02
        assert R>0.99

