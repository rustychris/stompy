import functools
import numpy as np
from collections import OrderedDict
import contextlib
import hashlib
import os
import pickle

# TODO: 
# caching may depend on the working directory -
# as it stands, it may think that the cache directory exists, but if
# the caller has changed directories, it won't find the files.
# not sure of the details, just noticed symptoms like this.
# 5/19/15.


class LRUDict(object):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        self.data = OrderedDict(*args, **kwds)
        self.check_size_limit()
  
    def __setitem__(self, key, value):
        if key in self.data:
            # remove, so that the new values is at the end
            del self[key]
        self.data[key] = value
        self.check_size_limit()
    def __delitem__(self,key):
        del self.data[key]
        
    def __getitem__(self,key):
        value = self.data[key]
        del self.data[key]
        self.data[key] = value
        return value

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return repr(self.data)
    def __contains__(self,k):
        return k in self.data

    def check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                # print "trimming LRU dict"
                self.data.popitem(last=False)

def memoize_key(*args,**kwargs):
    # new way - slower, but highly unlikely to get false positives
    return hashlib.md5(pickle.dumps( (args,kwargs) )).hexdigest()

def memoize_key_str(*args,**kwargs):
    return str(args) + str(kwargs)

def memoize_key_strhash(*args,**kwargs):
    return hashlib.md5( memoize_key_str(*args,**kwargs).encode() ).hexdigest()

def memoize_key_repr(*args,**kwargs):
    # repr is probably more appropriate than str
    return repr(args) + repr(kwargs)

def memoize(lru=None,cache_dir=None,key_method='pickle'):
    """
    add as a decorator to classes, instance methods, regular methods
    to cache results.
    Setting memoize.disabled=True will globally disable caching, though new
    results will still be stored in cache
    passing lru as a positive integer will keep only the most recent
    values

    key_method: 'pickle' use the hash of the pickle of the inputs.  overkill,
      but highly unlikely to get false hits.
      'str': use the hash of the str-ified parameters
      callable: pass key_method(*args,**kwargs) will be the key
    """
    if cache_dir is not None:
        cache_dir=os.path.abspath( cache_dir )

    def memoize1(obj,key_method=key_method):
        if lru is not None:
            cache = obj.cache = LRUDict(size_limit=lru)
        else:
            cache = obj.cache = {}

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        
        @functools.wraps(obj)
        def memoizer(*args, **kwargs):
            recalc= memoizer.recalculate or memoize.recalculate
            if key_method=='pickle':
                key = memoize_key(args,**kwargs)
            elif key_method=='str':
                key = memoize_key_str(args,**kwargs)
            elif key_method=='strhash':
                key = memoize_key_strhash(args,**kwargs)
            elif key_method=='repr':
                key = memoize_key_repr(args,**kwargs)
            else:
                key=key_method(args,**kwargs)
            value_src=None

            if cache_dir is not None:
                cache_fn=os.path.join(cache_dir,key)
            else:
                cache_fn=None
            # TODO: If pickling fails on read or write, regroup
            if memoize.disabled or recalc or (key not in cache):
                if cache_fn and not (memoize.disabled or recalc):
                    if os.path.exists(cache_fn):
                        with open(cache_fn,'rb') as fp:
                            # print "Reading cache from file"
                            value=pickle.load(fp)
                            value_src='pickle'
                if not value_src:
                    value = obj(*args,**kwargs)
                    value_src='calculated'

                if not memoize.disabled:
                    cache[key]=value
                    if value_src=='calculated' and cache_fn:
                        with open(cache_fn,'wb') as fp:
                            pickle.dump(value,fp,-1)
                            # print "Wrote cache to file"
            else:
                value = cache[key]
            return value
        # per-method recalculate flags -
        # this is somewhat murky - it depends on @functools passing
        # the original object through, since here memoizer is the return
        # value from functools.wraps, but in the body of memoizer it's
        # not clear whether memoizer is bound to the wrapped or unwrapped
        # function.
        memoizer.recalculate=False

        return memoizer
    return memoize1
memoize.recalculate=False # force recalculation, still store in cache.
memoize.disabled = False  # ignore the cache entirely, don't save new result


def imemoize(lru=None,key_method='pickle'):
    """
    like memoize, but specific to instance methods, and keeps the 
    cache on the instance.

    add as a decorator to instance methods to cache results.

    key_method: 'pickle' use the hash of the pickle of the inputs.  overkill,
      but highly unlikely to get false hits.
      'str': use the hash of the str-ified parameters
      callable: pass key_method(*args,**kwargs) will be the key
    """
    def memoize1(obj,key_method=key_method):

        @functools.wraps(obj)
        def memoizer(self,*args, **kwargs):
            if key_method=='pickle':
                key = memoize_key(args,**kwargs)
            elif key_method=='str':
                key = memoize_key_str(args,**kwargs)
            elif key_method=='strhash':
                key = memoize_key_strhash(args,**kwargs)
            else:
                key=key_method(args,**kwargs)

            # to distinguish multiple methods
            key=str(obj),key
            
            try:
                cache=self._memocache
            except AttributeError:
                if lru is not None:
                    cache = LRUDict(size_limit=lru)
                else:
                    cache = {}
                self._memocache=cache
            
            if key not in cache:
                value = obj(self,*args,**kwargs)
                cache[key]=value
            else:
                value = cache[key]
            return value
        return memoizer
    return memoize1


# returns a memoize which bases all relative path cache_dirs from
# a given location.  If the given location is a file, then use the dirname
# i.e.
#  from memoize import memoize_in
#  memoize=memoize_in(__file__)
def memoizer_in(base):
    if os.path.isfile(base):
        base=os.path.dirname(base)
    def memoize_in_path(lru=None,cache_dir=None):
        if cache_dir is not None:
            cache_dir=os.path.join(base,cache_dir)
        return memoize(lru=lru,cache_dir=cache_dir)
    return memoize_in_path

@contextlib.contextmanager
def nomemo():
    saved=memoize.disabled
    memoize.disabled=True
    try:
        yield
    finally:
        memoize.disabled=saved


def member_thunk(obj):
    """
    memoize for instance methods with no arguments.
    """
    @functools.wraps(obj)
    def memoizer(self):
        attr_name='_' + obj.__name__
        if hasattr(self,attr_name):
            return getattr(self,attr_name)
        else:
            value=obj(self)
            setattr(self,attr_name,value)
            return value
    return memoizer
