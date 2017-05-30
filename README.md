# stompy

Various python modules related to modeling and oceanographic data analysis.

## Installation

There is not yet a pip or conda installer setup. (if you use this code and would find that useful, please
add an issue on github so I know somebody cares).

### Requirements

`stompy` makes extensive use of the core packages of a modern scientific python installation,
plus a few slightly more specialized modules:

 * python 2.7 or 3
 * six
 * numpy
 * scipy
 * gdal
 * shapely
 * matplotlib
 * xarray
 * pandas
 * netCDF
 
### Installation

Python must be able to find the `stompy` subdirectory of the repository.  So on Linux, this might look like:

```
   cd $HOME/src
   git clone https://github.com/rustychris/stompy.git
   export PYTHONPATH=$PYTHONPATH:$HOME/src/stompy
```

At this point, you should be able to start python and successfully run `import stompy.grid.unstructured_grid`, for example.

 
