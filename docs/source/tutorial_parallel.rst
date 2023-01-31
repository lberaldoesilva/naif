Extracting frequencies in parallel
==================================


This tutorial is useful if you need to extract frequencies for many
orbits, which can then be parallelized, since the analysis of one
orbit is completely independent from the others. We integrate orbits in
the isochrone potential using the package agama. The code below is
only expected to work on python scripts, but not in notebooks. Also,
this is one possible solution but you may find a better one.

As usual, we start importing the relevant modules::

  >>> import numpy as np
  >>> import scipy
  >>> import agama
  >>> import naif
  >>> import time
  >>> import datetime
  >>> import multiprocessing as mp
  >>> from concurrent.futures import ProcessPoolExecutor


Then we define some relevant numbers, such as number of orbits, points
per orbit and so on::

  >>> # n. of orbits:
  >>> n_orbs = 100
  >>> # n. of circular periods to integrate for:
  >>> n_Tcirc = 100
  >>> # n. of points per orbit:
  >>> n_steps = 100_000
  >>> # n. of orbits per loop:
  >>> n_orbs_loop = 8
  >>> ndim = 6
  >>> # n. of frequencies to extract:
  >>> n_freqs = 5

The number of orbits per loop should be comparable to the number of
cores available. The function below is the auxiliary function used to
parallelize the code, and the arrays ``f_tseries`` and ``f_time`` are
global arrays::

  >>> def func_calc_freqs(k):
  >>>     out_om, out_a = naif.find_peak_freqs(f_tseries[k], f_time[k], n_freqs=n_freqs)
  >>>     return k, out_om, out_a

We can also use a function to track the elapsed time and estimate for the end time::

 >>> def calc_comp_time(t_start, cur_iter, max_iter):
 >>>     t_elapsed = time.time() - t_start
 >>>     t_estimate = (t_elapsed/cur_iter)*(max_iter)
 >>> 
 >>>     t_finish = t_start + t_estimate
 >>>     t_finish = datetime.datetime.fromtimestamp(t_finish).strftime("%H:%M:%S")
 >>> 
 >>>     t_left = (t_estimate-t_elapsed)/3600.  # in hours
 >>> 
 >>>     return (int(t_elapsed), float(t_left), t_finish) 

We generate an initial sample and loop over the initial conditions,
integrating and analyzing the orbits in blocks of
``n_orbs_loop``. This is because the parallelization can easily
explode the memory if we do it for all the orbits altogether. Here, we
apply the same procedure used in the other tutorials to identify the
fundamental frequency in the spectra. Then, we finally save the
frequencies and absolute amplitudes to a file. ::

 >>> M = 1.
 >>> Rc = 1.
 >>> isoc_pot = agama.Potential(type='Isochrone', mass=M, scaleRadius=Rc)
 >>> isoc_df = agama.DistributionFunction(type='QuasiSpherical', potential=isoc_pot, density=isoc_pot)
 >>> isoc_data,_ = agama.GalaxyModel(isoc_pot, isoc_df).sample(n_orbs)
 >>>
 >>> file_freqs_name = './test_parallel.freqs'
 >>> file_freqs = open(file_freqs_name, 'w')
 >>> file_freqs.write('   i         w_r              w_phi              A_r              A_phi\n')
 >>> file_freqs.close()
 >>>
 >>> start = time.time()
 >>> this_iter = 0
 >>> max_iter = n_orbs/n_orbs_loop
 >>> 
 >>> for i in range(0, n_orbs, n_orbs_loop):
 >>>     iend = i+n_orbs_loop
 >>>     if (iend > n_orbs):
 >>>         iend = n_orbs
 >>> 
 >>>     k = iend-i
 >>>     ic = isoc_data[i:iend]
 >>>     orbs = agama.orbit(potential=isoc_pot, ic = ic,
 >>>                        time=n_Tcirc*isoc_pot.Tcirc(ic), trajsize=n_steps+1,
 >>>                        accuracy=1e-15, dtype=float)
 >>> 
 >>>     t = np.vstack(orbs[:,0])[:,:-1] # time
 >>>     all_coords = np.vstack(orbs[:,1]).reshape(k, n_steps+1, ndim)
 >>>     x = all_coords[:,:-1,0]
 >>>     y = all_coords[:,:-1,1]
 >>>     z = all_coords[:,:-1,2]
 >>>     vx = all_coords[:,:-1,3]
 >>>     vy = all_coords[:,:-1,4]
 >>>     vz = all_coords[:,:-1,5]
 >>>         
 >>>     r = np.sqrt(x**2 + y**2 + z**2)
 >>>     phi = np.arctan2(y, x)
 >>>     Lz = (x*vy - y*vx)
 >>> 
 >>>     om_r = np.zeros((k,n_freqs))
 >>>     om_phi = np.zeros((k,n_freqs))
 >>>         
 >>>     a_r = np.zeros((k,n_freqs), dtype=np.complex128)
 >>>     a_phi = np.zeros((k,n_freqs), dtype=np.complex128)
 >>> 
 >>>     # Frequencies (not in parallel):
 >>>     # fr = r + 1j*vr
 >>>     # fphi = np.sqrt(2*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
 >>>     # for j in range(k):
 >>>     #     om_r[j], a_r[j] = naif.find_peak_freqs(fr[j], t[j], n_freqs=n_freqs)
 >>>     #     om_phi[j], a_phi[j] = naif.find_peak_freqs(fphi[j], t[j], n_freqs=n_freqs)
 >>>
 >>>     f_time = t
 >>>         
 >>>     # for r:
 >>>     # this is seen as a global array by the function func_calc_freqs:
 >>>     f_tseries = r + 1j*vr
 >>>     if __name__ == "__main__":
 >>>         with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as pool:
 >>>             for row, out_om, out_a in pool.map(func_calc_freqs, range(k)):
 >>>                 om_r[row] = out_om
 >>>                 a_r[row] = out_a
 >>> 
 >>>     # for phi:
 >>>     f_tseries = np.sqrt(2*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
 >>>     if __name__ == "__main__":
 >>>         with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as pool:
 >>>             for row, out_om, out_a in pool.map(func_calc_freqs, range(k)):
 >>>                 om_phi[row] = out_om
 >>>                 a_phi[row] = out_a
 >>> 
 >>>     # For radial component, let's take the leading frequency:
 >>>     out_om_r = np.abs(om_r[:,0])
 >>>     out_a_r = np.abs(a_r[:,0])
 >>>     # For the azimuthal:
 >>>     out_om_phi = np.zeros(k)
 >>>     out_a_phi = np.zeros(k)
 >>>     for j in range(k):
 >>>         idx_fund = np.where(np.abs(om_phi[j]) > np.abs(om_r[j])/2.)[0]
 >>>         out_om_phi[j] = om_phi[j][idx_fund][0]
 >>>         out_a_phi[j] = np.abs(a_phi[j][idx_fund][0])
 >>> 
 >>>     np.savetxt(file_freqs, np.column_stack([np.arange(i,iend),
 >>>                                             out_om_r, out_om_phi,
 >>>                                             out_a_r, out_a_phi]), 
 >>>                fmt='%6d'+' %17.8e'*4)
 >>>     file_freqs.close()
 >>>  
 >>>     this_iter += 1
 >>>     comp_time = calc_comp_time(start,this_iter ,max_iter)
 >>>     print("time until now: %s(s), time left: %5.2f(h), estimated finish time: %s"%comp_time)
 >>>     print ('---------------------------')
 >>> 
 >>> end = time.time()
 >>> print("total time elapsed: %s(s), total time elapsed: %s(h)"%(round(end-start,4),
 >>>                                                               round((end-start)/3600.,4)))

Some comments on the code above:

* As you can see, we save the results block per block. In this way, if any problem happens, you can start later from where you stopped.

* We also included the code for doing it without parallelization, in case one wants to compare the performance.

* The argument ``mp_context=mp.get_context('fork')`` in the ``ProcessPoolExecutor`` seems required in Mac OS, but not in Linux (although it works in both if we use this).

  
