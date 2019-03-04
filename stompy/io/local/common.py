import datetime
import pandas as pd

from ... import utils

def periods(start_date,end_date,days_per_request):
    start_date=utils.to_datetime(start_date)
    end_date=utils.to_datetime(end_date)

    if days_per_request is None:
        yield (start_date,end_date)
    elif isinstance(days_per_request,str):
        # This is deprecated:
        #periods=pd.PeriodIndex(start=start_date,end=end_date,freq=days_per_request)
        # This is the recommended update
        periods=pd.period_range(start=start_date,end=end_date,freq=days_per_request)
        for period in periods:
            # The round() call both avoids a warning about nanoseconds, and
            # makes this output equivalent to the days_per_request~integer
            # case below
            yield (period.start_time.to_pydatetime(),
                   period.end_time.round('h').to_pydatetime())
    else:
        interval=datetime.timedelta(days=days_per_request)

        while start_date<end_date:
            next_date=min(start_date+interval,end_date)
            yield (start_date,next_date)
            start_date=next_date
