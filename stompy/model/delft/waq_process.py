import numpy as np
from . import nefis
from contextlib import contextmanager
import six

# utilities to grab some data from the process database.
# since the process database is specific to an installation/version,
# and scenarios already carry around paths to their installation,
# ProcessDB requires a scenario in order to find the appropriate
# database files, or explicit paths
class SubstanceDef(object):
    pass

class ProcessDB(object):
    def __init__(self,scenario=None,proc_dat=None,proc_def=None,proc=None):
        self.scenario=scenario # may be None

        if not (proc_dat and proc_def):
            if not proc:
                proc=self.scenario.proc_path
            proc_dat=proc +".dat"
            proc_def=proc +".def"
            
        self.proc_dat=proc_dat
        self.proc_def=proc_def
        
    @contextmanager
    def nef(self):
        nef=nefis.Nefis(self.proc_dat,self.proc_def)
        yield nef
        nef.close()

    def p2_idx_by_item_id(self,subst):
        with self.nef() as db:
            p2_items=db['TABLE_P2'].getelt('ITEM_ID',[0])

        for i in range(len(p2_items)):
            # py3 - have to be careful of bytes vs. str
            p2_item=p2_items[i]
            if six.PY3:
                p2_item=p2_item.decode()
            if p2_item.strip().lower() == subst.lower():
                return i
        return None

    def substance_by_id(self,subst):
        idx=self.p2_idx_by_item_id(subst)
        if idx is None:
            return None

        sub=SubstanceDef()

        with self.nef() as db:
            for elt in ['ITEM_ID','ITEM_NM','UNIT','DEFAULT','AGGREGA','DISAGGR',
                        'GROUPID','SEG_EXC','WK']:
                val=db['TABLE_P2'].getelt(elt,[0,idx])
                # old way:
                #if np.issubdtype(str,val.dtype):
                #    val=str(val.item().strip())
                #else:
                #    val=val.item()
                val=val.item()
                if six.PY3 and isinstance(val,bytes):
                    val=val.decode()
                if isinstance(val,str):
                    val=val.strip()

                setattr(sub,elt.lower(),val)
        return sub

