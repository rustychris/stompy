from collections import defaultdict
import pandas as pd
import numpy as np
import os
import re
import glob
import six

import logging
log=logging.getLogger('stompy')

try:
    from graphviz import Digraph
except ImportError:
    log.debug("graphviz unavailable")

class ProcDiagram(object):
    delft_src="/home/rusty/code/delft/d3d/master"

    def __init__(self,waq_dir,run_name='dwaq'):
        self.run_name=run_name
        self.load_tables()
        self.load_run(waq_dir)

    def load_tables(self):
        # Load the tables:
        table_dir=os.path.join( self.delft_src,
                                "src/engines_gpl/waq/packages/waq_kernel/src/waq_tables" )

        tab_inputs = pd.read_csv(os.path.join(table_dir,'inputs.csv'))
        tab_outputs= pd.read_csv(os.path.join(table_dir,'outputs.csv'))
        tab_proces = pd.read_csv(os.path.join(table_dir,'proces.csv'))

    def load_run(self,waq_dir):
        self.waq_dir=waq_dir

        self.substances=substances=defaultdict(dict)
        self.sub_map=sub_map={} # map 1-based number of substance to substance name
        self.processes=processes=defaultdict(dict)
        self.constants=constants=defaultdict(dict)
        self.links=links=defaultdict(dict) # (process,substance,process)

        lsp_fn=glob.glob( os.path.join(self.waq_dir,"*.lsp"))[0]
        lsp = open(lsp_fn,'rt')

        lines=iter(lsp)

        try:
            while 1:
                l=six.next(lines).strip()

                # substance/fluxes:
                # -fluxes for [OXY                 ]
                m=re.match(r'-fluxes for \[(.*)\]\s*',l)
                if m:
                    sub=m.group(1).strip()
                    substances[sub]['name'] = sub
                    sub_map[len(substances)] = sub
                    l=six.next(lines).strip()
                    while 1:
                        m=re.match(r'found flux  \[(.*)\].*',l)
                        if not m:
                            break
                        flux_name=m.group(1).strip()
                        l=six.next(lines).strip()
                        m=re.match(r'from proces \[(.*)\].*',l)
                        proc_name=m.group(1).strip()
                        link = (proc_name,flux_name,sub)
                        links[link]='flux'
                        l=six.next(lines).strip()
                        if l == 'process is switched on.':
                            l=six.next(lines).strip()


                # is it the start of an input stanza?
                m=re.match(r'\s*Input for \[(.*)\s*\].*',l)
                if m:
                    input_for_process=m.group(1).strip()
                    processes[input_for_process]['name']=input_for_process

                    # print("inputs for %s"%input_for_process)

                    while 1:
                        l=six.next(lines).strip()
                        m=re.match(r'\s*\[(.*)\s*\].*',l)
                        if m:
                            input_name=m.group(1).strip()
                            l=six.next(lines).strip()
                            input_src = re.match(r'\s*[Uu]sing (.*)$',l).group(1).strip()
                            # print("   %s => %s"%(input_src,input_name))

                            if input_src.startswith('substance'):
                                src=('substance',int(input_src.split()[2]))
                                # translate to name:
                                src=sub_map[src[1]]
                            elif input_src.startswith('segment function nr'):
                                src=('segfn',int(input_src.split()[3]))
                            elif input_src.startswith('parameter nr'):
                                src=('param',int(input_src.split()[2]))
                            elif input_src.startswith('default value'):
                                src=('default',input_src.split(':')[1].strip())
                            elif input_src.startswith('constant nr'):
                                src=('constant',input_src.split(':')[1].strip())
                            elif input_src.startswith('output from proces'):
                                m=re.match(r'output from proces \[(.*)\]',
                                           input_src)
                                src=('process',m.group(1).strip())
                            elif input_src.startswith('flux from proces'):
                                m=re.match(r'flux from proces \[(.*)\]',
                                           input_src)
                                src=('process',m.group(1).strip())
                            elif input_src.startswith('DELWAQ'):
                                src=('builtin',input_src)
                            else:
                                raise Exception("Failed to parse '%s'"%input_src)

                            link=( src, input_name,input_for_process )
                            links[link]='input'
                        else:
                            break
        except StopIteration:
            pass

        if 0:
            print("-="*30)
            # print the interesting ones:
            for (src,name,dest) in links:
                if isinstance(src,tuple):
                    if src[0] in ('constant','default'):
                        continue

                print(" %s ---  %s ----> %s"%(src,name,dest))

    def init_dot(self):
        # strict merges duplicate edges

        # dot is the only one which gives reasonable output
        # could also add in size="5,5"
        from graphviz import Digraph

        dot = Digraph(name="schematic-%s"%self.run_name,
                      comment='WAQ Processes',strict=True,engine='dot')
        #dot.attr('graph',ranksep="0.4",ratio='auto')
        dot.attr('graph',ranksep="0.3",ratio='0.647')
        dot.attr('node',fontname='Helvetica',style='filled',fillcolor='gray80',fontsize="32")
        dot.attr('edge',penwidth="2")

        return dot
    
    def create_dot(self):
        # results in these links, plus some more boring ones.
        # the idea is then that we draw edges of the graph between the first and third items.
        # possibly labeled with the value of the second item.
        # there are a bunch of constants and defaults - not that exciting.

        dot=self.init_dot()
        
        phys_params=['Depth','Surf','DynDepth','TotDepth','TotalDepth',
                     'Emersion']

        nodes_output={} # track which nodes have already been output

        if 1:
            sif_fp=None # too lazy to refactor properly
        else:
            sif_fp=open('graph-%s.sif'%self.run_name,'wt')


        def sel_edge(a,b):
            if (a not in phys_params) and (b not in phys_params):
                for n in [a,b]:
                    if n in self.substances:
                        dot.node(n,n, style="filled",fillcolor='green')
                dot.edge(a,b)
                if sif_fp:
                    sif_fp.write("%s linkto %s\n"%(a,b))
            
        for (src,name,dest),ltype in six.iteritems(self.links):
            if isinstance(src,tuple):
                src_type,src_val = src
                if src_type in ('constant','default','builtin'):
                    continue
                if src_type=='process':
                    src=src_val
                else:
                    src="_".join([str(s) for s in src])
            else:
                src_type=None


            if ltype=='flux':
                # drop intermediate names like dREAROXY
                sel_edge(src,dest)
            else:
                if (src!=name) and (src_type not in ['segfn','param']):
                    sel_edge(src,name)
                else:
                    if src_type in ['segfn','param'] and name not in phys_params:
                        dot.node(name,name,style="filled",fillcolor='cornflowerblue')

                sel_edge(name,dest)

        if sif_fp:
            sif_fp.close()
            sif_fp=None
            
        return dot

    def view_dot(self):
        self.create_dot().view()
    def render_dot(self):
        self.create_dot().render()

