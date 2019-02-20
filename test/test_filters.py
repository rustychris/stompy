import numpy as np
from stompy import filters

def test_modes():
    t_h=np.arange(0,24*20)

    x= 2.0+ ( 1.0 *np.cos( t_h/12.4 * 2*np.pi ) +
              0.3 *np.cos( t_h/12. * 2*np.pi ) +
              0.1 *np.cos( t_h/24.8 * 2*np.pi ) +
              0.1 *np.cos( t_h/24. * 2*np.pi ) )
    x_lp=filters.lowpass_godin(x,t_h/24.)
    x_lp_nan=filters.lowpass_godin(x,t_h/24.,ends='nan')

    if 0:
        import matplotlib.pyplot as plt # just for dev
        plt.figure(10).clf()
        plt.plot(t_h,x)
        plt.plot(t_h,x_lp)
        plt.plot(t_h,x_lp_nan)




