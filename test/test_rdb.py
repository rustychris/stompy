from importlib import reload
import rdb
import pdb

reload(rdb)

fp=open('test_rdb.rdb','rb')

#pdb.run( "r=rdb.Rdb(fp=fp)")
r=rdb.Rdb(fp=fp)
