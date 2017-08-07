import matplotlib.pyplot as plt

# Scipy imports.
from scipy import linalg, special
from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
     ravel, power, atleast_1d, squeeze, sum, transpose
import numpy as np


class gaussian_kde(object):
    def __init__(self, dataset, weights, inv_cov, norm_factor):
        self.dataset = np.asarray(dataset)
        self.d, self.n = self.dataset.shape
        weights = np.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        self.inv_cov = inv_cov
        self._norm_factor = norm_factor

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters:

        points: (# of dimensions, # of points)-array
          Alternatively, a (# of dimensions,) vector can be passed in and
          treated as a single point.

        Returns:

        values: (# of points,)-array. The values at each point.

        Raises:

        ValueError
          if the dimensionality of the input points is different than
          the dimensionality of the KDE.

        """
        points = atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,), dtype=np.float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0) / 2.0
                result = result + self.weights[i]*exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = sum(self.weights*exp(-energy), axis=0)

        result = result / self._norm_factor

        return result

    __call__ = evaluate
class Covariator(object):
    def __init__(self, dataset, weights):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape
        weights = np.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        

    def scotts_factor(self):
        return power(np.ceil(self.n*(1-(self.weights**2).sum())), -1./(self.d+4))

    def silverman_factor(self):
        return power(np.ceil(self.n*(1-(self.weights**2).sum()))*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def __call__(self, bw_method=None):
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, basestring):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        return self._compute_covariance()


    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        self._data_covariance = atleast_2d(cov(self.dataset.T, self.weights))

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = linalg.pinv(self.covariance)
        self._norm_factor = (2*pi)**(self.d/2.)*sqrt(linalg.det(self.covariance))
        return self.inv_cov, self._norm_factor
        
def cov(data, weights):
  # gives biased cov. estimate
  # data.shape = n_elements, n_dimensions
  data, weights = np.array(data), np.array(weights, dtype=float)
  weights /= weights.sum()
  weights = weights[:,np.newaxis]
  mean = (data*weights).sum(axis=0)
  data -= mean
  return np.dot((data*weights).T, data)


def kde_hist(x,weights=None,bins=None,range=None,normed=False,stacked=True,
             color=None,label=None,
             histtype='step',ax=None,**line_args):
    """ a weighted kernel density estimate analog to matplotlib's hist().
    """
    ax=ax or plt.gca()

    # make x into a list of datasets
    def to_list_of_arrays(inp):
        if isinstance(inp,np.ndarray):
            if inp.ndim==1:
                inp=[inp]
        else:
            inp=list(inp)
            if np.iterable( inp[0] ):
                # make sure components are all arrays
                inp=[np.asarray(v) for v in inp]
            else:
                # looks llike it's a single list of items -
                inp=[np.asarray(inp)]
        return inp
    x=to_list_of_arrays(x)

    if weights is None:
        weights=[np.ones_like(v) for v in x]
    else:
        weights=to_list_of_arrays(weights)

    if len(weights)==1 and len(x)>1:
        weights=[weights[0]]*len(x)

    if not isinstance(color,list):
        color=[color]*len(x)
    if not isinstance(label,list):
        label=[label]*len(x)
    
    if range is None:
        xmin=np.min( [np.min(v) for v in x] )
        xmax=np.max( [np.max(v) for v in x] )
        range=[xmin,xmax]
    if bins is None:
        bins=10
    # oversample by factor of 5
    x_grid=np.linspace(range[0],range[1],5*bins)

    last_kde=np.zeros(len(x_grid))

    returns=[]

    kdes=[]
    total_weight=0

    for v,w in zip(x,weights):
        valid = np.isfinite(v*w)
        v=v[valid]
        w=w[valid]

        if w.sum()==0.0:
            kdes.append(None)
            continue
        
        w_cov = Covariator(v,w)
        w_xstd = np.sqrt(cov(v[:,None],w))[0,0]
        inv_cov,norm_factor = w_cov( float(range[1]-range[0])/(1.25*bins*w_xstd))

        gkde=gaussian_kde(v[None,:],weights=w,
                          inv_cov=inv_cov,norm_factor=norm_factor)

        # gkde will automatically normalize - 
        # denormalize so that multiple datasets can be combined meaningfully.
        kdes.append( gkde(x_grid) * w.sum() ) 
        total_weight += w.sum()

    cumul_kde=np.zeros_like(x_grid)

    hist_n=[] # collect the actual values plotted to return to the user
    patches=[]

    for kde,c,l in zip(kdes,color,label):
        if kde is None:
            continue
        if not normed:
            kde = kde*(range[1]-range[0])/bins
        else:
            kde = kde/total_weight

        hist_n.append(kde)

        style={}
        if l is not None:
            style['label']=l
        if c is not None:
            style['color']=c

        style.update(line_args)
        if histtype=='step':
            if stacked:
                lines = ax.plot(x_grid,cumul_kde+kde,**style)
            else:
                lines = ax.plot(x_grid,kde,**style)
        elif histtype=='stepfilled':
            lines = ax.fill_between(x_grid,cumul_kde,cumul_kde+kde,**style)
            # create a 'proxy' artist so this will show up in a legend
            p = plt.Rectangle((0, 0), 0, 0, **style)
            ax.add_patch(p)
        else:
            assert(False)
        patches.append(lines)
        cumul_kde+=kde        

    if len(hist_n)==1:
        return (hist_n[0],x_grid,patches[0])
    else:
        return (hist_n,x_grid,patches)
