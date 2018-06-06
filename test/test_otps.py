from stompy.model.otps import read_otps

import six
six.moves.reload_module(read_otps)

modfile="data/DATA/Model_OR1km"

def test_read_otps():
    times=np.arange( np.datetime64('2010-01-01 00:00'),
                     np.datetime64('2010-01-10 00:00'),
                     np.timedelta64(15,'m') )

    pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=[235.25], lat=[44.5],
                                             time=times)



    if 0:
        # Compare to NOAA:
        # The comparison is not great - probably because this database has very few constituents, just
        #  M2, S2, N2, K2
        from stompy.io.local import noaa_coops

        cache_dir='cache'
        os.path.exists(cache_dir) or os.mkdir(cache)

        sb_tides=noaa_coops.coops_dataset_product(9435380,'water_level',
                                                  start_date=times[0],
                                                  end_date=times[-1],
                                                  days_per_request='M',
                                                  cache_dir=cache_dir)

        from stompy import utils
        plt.clf()
        plt.plot(utils.to_datetime(times),pred_h,label='h')

        water_level=sb_tides.water_level.isel(station=0)
        water_level = water_level - water_level.mean()

        plt.plot( utils.to_datetime(sb_tides.time),water_level, label='NOAA')

        plt.gcf().autofmt_xdate()
        plt.legend()
