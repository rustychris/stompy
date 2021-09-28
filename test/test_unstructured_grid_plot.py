from __future__ import print_function

import numpy as np
import os

import matplotlib.pyplot as plt
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid


def test_square_axes():
    sample_data=os.path.join(os.path.dirname(__file__),'data')
    g=unstructured_grid.UnstructuredGrid.read_dfm(os.path.join(sample_data,"lsb_combined_v14_net.nc"))

    expected=np.r_[490243.0, 609718.4589319953, 4137175.7739677625, 4232151.0]

    fig=plt.figure(1)
    fig.clf()
    g.plot_cells()
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)

    fig.clf()
    g.plot_nodes() 
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)

    fig.clf()
    g.plot_edges()
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)

    # What about shared axes?
    # This works:
    fig.clf()
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True,num=1)
    g.plot_nodes(ax=axs[0])
    g.plot_nodes(ax=axs[1])
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)

    # This is okay
    fig.clf()
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True,num=1)
    g.plot_edges(ax=axs[0])
    g.plot_edges(ax=axs[1])
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)

    fig.clf()
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True,num=1)
    g.plot_cells(ax=axs[0])
    g.plot_cells(ax=axs[1])
    plt.draw()
    result=plt.axis()
    diff = np.abs(np.array(result) - expected)
    assert np.all(diff<10.0)




