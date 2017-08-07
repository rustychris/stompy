"""
Rough implementation of Makefile-like semantics for
python classes.

This does not work terribly well, and should not be used 
for any new code.  That is why it is squirreled away in
this suntans-specific directory, since it is in use only
in the domain.py framework.
"""
import os,time


# How to emulate Makefile behavior?
#  could it register functions, with the dependencies written like in a
#  Makefile?

# depends('depth.dat: %(original_grid_dir)/{points,cells,edges}.dat',
#         self.depth_create)

# and the function would get the actual names that matched, i.e.
# self.depth_create(deps=[<path>/points.dat,<path>/cells.dat,<path>/edges.dat],
#                   target=[depth.dat])

# maybe make it more explicit:
#  rule(target='depth.dat',
#       deps=[...] )
# because then deps could be a dict of filenames, dict of patterns, etc.

# For starters, could keep it out of the classes, just have the classes
# add their rules.
FILE_EXISTS=object()



class Node(object):
    def __init__(self,graph,target,rule,deps):
        self.graph = graph
        
        self.target = target
        self.rule = rule
        self.deps = deps
        self.tstamp = None

        
    def run_command(self):
        print("Running commands for %s"%self.target)

        ret_val = self.rule.invoke(self.target,self.deps)
        
        # sometimes even when the command is run it doesn't change the file,
        # and we should stick with that older time
        fs_timestamp = self.get_timestamp(fs_only=1)

        if fs_timestamp < 0:
            self.tstamp = time.time()
        else:
            self.tstamp = fs_timestamp
            
    def get_timestamp(self,fs_only=0):
        """ fs_only: ignore any internal timestamps, relying only on the filesystem
        """
        # in case it's a phony target, this will use the time when the command
        # was run
        if self.tstamp is not None:
            return self.tstamp

        if os.path.exists(self.target):
            # for files, we want last modified time:
            if os.path.isfile(self.target):
                return os.stat(self.target).st_mtime
            elif os.path.isdir(self.target):
                return 1 # only care that it exists.
        else:
            return -1
        
    def is_current(self):
        if self.rule.always_run:
            return False
        
        my_timestamp = self.get_timestamp()

        if my_timestamp < 0:
            return False

        if not self.graph.check_timestamps:
            # The file exists - good enough.
            return True

        for dep in self.deps:
            dep_timestamp = self.graph.nodes[dep].get_timestamp()
            if dep_timestamp > my_timestamp:
                print( "%s@%s > %s@%s"%(dep,dep_timestamp,
                                        self.target,my_timestamp) )
                return False
        return True
        

class Rule(object):
    def __init__(self,target,deps=[],func=None,always_run=False):
        self.target = target

        if deps is None:
            deps = []
        elif type(deps) != list:
            deps = [deps]
            
        self.deps = deps
        self.func = func

        # some rules we want to always fire, and then the rule will figure out
        # whether some work must be done.
        self.always_run = always_run

    def matches(self,target):
        if self.target == target:
            return True
        return False
    
    def invoke(self,target,deps):
        if self.func is not None:
            return self.func(target,deps)

    
class DependencyGraph(object):
    _base_graph = None
    check_timestamps = 1

    def __init__(self):
        self.rules = []
        
    def clear(self):
        self.rules = []
        self.nodes = None

    def rule(self,target,deps=[],func=None,always_run=False):
        """ here target is one filename or a list of filenames
        and deps is the same
        func has the signature func(target,deps)
        """
        r = Rule(target,deps,func,always_run=always_run)
        self.rules.append( r )

        return r

            
    def make(self,target):
        print( "Making target:",target)

        # ok.  CS101.
        # first figure out how we would make this thing, and
        # then see which parts are out of date and actually
        # do need to be made

        # so we create the graph.  nodes are identified by their
        # filenames.  for the moment that means that rules that create
        # multiple files are a bit harder...

        # the datastructure for each node is [target,rule,dependencies] 
        self.nodes = {}

        self.target_node = self.insert_node_for_target(target)

        # now we have the whole graph, but need to sort it topologically
        # topo sort review:
        #   start with directed acyclic graph.
        #   DFS search from the root, then order by the time when a node is
        #     left
        ordering = self.topo_sort(start=self.target_node)

        # print( "ordering:", ordering)
        for target in ordering:
            node = self.nodes[target]
            
            if node.is_current():
                print( "%s is current"%node.target)
            else:
                node.run_command()
    

    def topo_sort(self,start,exited=None,visited=None):
        """ start is a node in the tree
        appends nodes to exited once their subtree has been covered.
        visited is a dict of nodes that have already been visited.
        """
        if visited is None:
            visited = {}
            
        if exited is None:
            exited = []
            
        if start.target in visited:
            return

        for child in start.deps:
            self.topo_sort(self.nodes[child],exited,visited)

        exited.append(start.target)
        visited[start.target] = 1
        return exited

    def insert_node_for_target(self,target):
        if target not in self.nodes:
            rule = self.match_target(target)        
            n = Node(self,target,rule,rule.deps)

            self.nodes[n.target] = n

            # go ahead and fill in dependencies, DFS style
            self.enumerate_dependencies(n)

        return self.nodes[target]

    def match_target(self,target):
        for r in self.rules:
            if r.matches(target):
                return r

        # implicit rule for existing files - make new rule on the fly
        if os.path.exists(target):
            r = Rule(target)
            return r
        
        raise Exception("Target %s could not be found"%target)

    def enumerate_dependencies(self,n):
        for target in n.deps:
            self.insert_node_for_target(target)

        # next step is to depth-first through the dependencies, adding their rules in
        # until everybody has been added into nodes.
        # then topo-sort nodes, and walk through, checking timestamps for each
        # rule to see whether to invoke it's method.
        
        
        
        
DependencyGraph._base_graph=DependencyGraph()    

rule = DependencyGraph._base_graph.rule
make = DependencyGraph._base_graph.make
clear = DependencyGraph._base_graph.clear


if __name__ == '__main__':
    # Some testing:
    def touch(target,deps):
        print("%s => %s"%(deps,target))
        fp = file(target,'wt')
        fp.close()

    # foo depends on bar
    rule('foo',['bar','dog','cat'],touch)



    # bar depends on nobody
    rule('bar',None,touch)
    rule('dog','mouse',touch)
    rule('cat','mouse',touch)
    rule('mouse',None,touch)

    make('foo')
    
    

