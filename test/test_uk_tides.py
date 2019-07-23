from stompy.io.local import uk_tides
import numpy as np

cache_dir="."

# stations=get_tide_gauges(cache_dir)
uk_tides.get_tide_gauges(cache_dir='.')


station='E72124'
ds=uk_tides.fetch_tides(station=station,
                        start_date=np.datetime64("2019-07-01"),
                        end_date=np.datetime64("2019-07-10"),
                        cache_dir=cache_dir)

