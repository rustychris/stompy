#!/bin/sh

# Added -f 5.0 to coarsen the scale and complete much faster.
# this should take about 1800 steps to complete, maybe 1 minute on a 2019 laptop.
python ../stompy/grid/tom.py -f 5.0 -s data/scale-lines.shp -b data/dumbarton.shp 



