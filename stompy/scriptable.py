"""
provide command line access to methods on classes
"""
from __future__ import print_function

import sys
import getopt

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

class Scriptable(object):
    def invoke_command(self,cmd,args):
        meth_name = "cmd_" + cmd

        try:
            f = getattr(self,meth_name)
        except AttributeError:
            print("Command '%s' is not recognized"%cmd)
            self.cmd_help()
            sys.exit(1)
        f(*args)

    def cmd_help(self):
        """ 
        List available commands
        """
        print("Available commands")
        for k in dir(self):
            if k.startswith('cmd_'):
                v=getattr(self,k)
                doc=(v.__doc__ or "undocumented").strip() 
                # 9 seems to work for the usual indentation (8 spaces)
                # plus the 9 to line up with '%15s: ' below
                doc=doc.replace("\n","\n"+" "*9)
                print("%15s: %s"%(k.replace('cmd_',''),doc))

    cli_options="h" # can be overridden in subclasses
    def cli_usage(self):
        print("python {} [command]".format(sys.argv[0]) )
        print(" Command is one of:" )
        for att in dir(self):
            if att.startswith('cmd_'):
                print("    "+att[4:])
        print("Use help for more info")
    def cli_handle_option(self,opt,val):
        if opt == '-h':
            self.cli_usage()
            sys.exit(0)
        else:
            print("Unhandled option '%s'"%opt)
            self.cli_usage()
            seys.exit(1)
        
    def main(self,args=None):
        """ parse command line and start making things
        """
        if args is None:
            args = sys.argv[1:]

        try:
            opts,rest = getopt.getopt(args, self.cli_options)
        except getopt.GetoptError:
            self.cli_usage()
            sys.exit(1)
        
        for opt,val in opts:
            self.cli_handle_option(opt,val)

        if len(rest) > 0:
            cmd = rest[0]
            rest = rest[1:]
        else:
            # default action: 
            cmd = 'default'

        # try making all output unbuffered:
        sys.stdout = Unbuffered(sys.stdout)
            
        # DependencyGraph.check_timestamps = self.check_timestamps
        self.invoke_command(cmd,rest)
