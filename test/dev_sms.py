from stompy.grid import unstructured_grid
import six

six.moves.reload_module(unstructured_grid)

# thanks to Jacob Zhao for sample input
g=unstructured_grid.UnstructuredGrid.read_sms("0509.grd")

##

# appears to bring in cells, edges and edge marks okay.
plt.figure(1).clf()
g.plot_edges(values=g.edges['mark'])
g.plot_cells(alpha=0.5)




