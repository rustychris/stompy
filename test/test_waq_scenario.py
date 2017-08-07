from stompy.model.delft import waq_scenario
import datetime

def test_waq_timestep_timedelta():
    inputs=[100,"100","0010","5000000",5100]

    for x in inputs:
        td=waq_scenario.waq_timestep_to_timedelta(x)
        ts=waq_scenario.timedelta_to_waq_timestep(td)
        td2=waq_scenario.waq_timestep_to_timedelta(ts)
        assert td == td2
