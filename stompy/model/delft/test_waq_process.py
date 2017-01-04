import nose

from delft import waq_scenario
from delft import waq_process

def test_sub_lookup():
    scen=waq_scenario.Scenario(None)
    pdb=waq_process.ProcessDB(scenario=scen)
    sub=pdb.substance_by_id('Green')

    assert(sub.item_nm=='Algae (non-Diatoms)')


if __name__ == '__main__':
    nose.main()
