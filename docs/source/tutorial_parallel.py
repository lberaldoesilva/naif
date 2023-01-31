import numpy as np
import scipy
import agama
import naif

import time
import datetime

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

np.random.seed(0)

# n. of orbits:
n_orbs = 100
# n. of circular periods to integrate for:
n_Tcirc = 100

# n. of points per orbit:
n_steps = 100_000
# n. of orbits per loop:
n_orbs_loop = 8

ndim = 6
# n. of frequencies to extract:
n_freqs = 5
# -----------------------
def func_calc_freqs(k):
    out_om, out_a = naif.find_peak_freqs(f_tseries[k], f_time[k], n_freqs=n_freqs)
    return k, out_om, out_a
# -----------------------
def calc_comp_time(t_start, cur_iter, max_iter):
    t_elapsed = time.time() - t_start
    t_estimate = (t_elapsed/cur_iter)*(max_iter)

    t_finish = t_start + t_estimate
    t_finish = datetime.datetime.fromtimestamp(t_finish).strftime("%H:%M:%S")  # in time

    t_left = (t_estimate-t_elapsed)/3600.  # in hours

    return (int(t_elapsed), float(t_left), t_finish)
#--------------------
print ('Generating initial sample...')
M = 1.
Rc = 1.

isoc_pot = agama.Potential(type='Isochrone', mass=M, scaleRadius=Rc)
isoc_df = agama.DistributionFunction(type='QuasiSpherical', potential=isoc_pot, density=isoc_pot)
isoc_data,_ = agama.GalaxyModel(isoc_pot, isoc_df).sample(n_orbs)
# -----------------------
print ('Integrating all', n_orbs, 'orbs in blocks of', n_orbs_loop, 'orbs')

file_freqs_name = './test_parallel.freqs'
file_freqs = open(file_freqs_name, 'w')
file_freqs.write('   N         w_r              w_phi              A_r              A_phi\n')
file_freqs.close()
# -----------------------
print ('Integrating all', n_orbs, 'orbs in blocks of', n_orbs_loop, 'orbs')
start = time.time()
this_iter = 0
max_iter = n_orbs/n_orbs_loop
for i in range(0, n_orbs, n_orbs_loop):
    if (i >=0):
        iend = i+n_orbs_loop
        if (iend > n_orbs):
            iend = n_orbs

        k = iend-i
        ic = isoc_data[i:iend]
        orbs = agama.orbit(potential=isoc_pot, ic = ic,
                           time=n_Tcirc*isoc_pot.Tcirc(ic), trajsize=n_steps+1,
                           accuracy=1e-15, dtype=float)
        #--------------------
        t = np.vstack(orbs[:,0])[:,:-1] # time
        all_coords = np.vstack(orbs[:,1]).reshape(k, n_steps+1, ndim)
        x = all_coords[:,:-1,0]
        y = all_coords[:,:-1,1]
        z = all_coords[:,:-1,2]
        vx = all_coords[:,:-1,3]
        vy = all_coords[:,:-1,4]
        vz = all_coords[:,:-1,5]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        vr = (x*vx + y*vy + z*vz)/r
        phi = np.arctan2(y, x)
        Lx = (y*vz - z*vy)
        Ly = (z*vx - x*vz)
        Lz = (x*vy - y*vx)
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

        v2_0 = vx[:,0]**2 + vy[:,0]**2 + vz[:,0]**2
        pos_0 = np.column_stack((x[:,0].T, y[:,0].T, z[:,0].T))
        E = isoc_pot.potential(pos_0) + 0.5*v2_0

        # Analytical values:
        an_Jr = M/np.sqrt(-2*E) - 0.5*(L[:,0] + np.sqrt(L[:,0]**2 + 4*M*Rc))
        an_Jtheta = L[:,0] - np.abs(Lz[:,0])
        an_Jphi = Lz[:,0]
        
        an_Om_r = M**2/(an_Jr + 0.5*(L[:,0] + np.sqrt(L[:,0]**2 + 4*M*Rc)))**3
        an_Om_theta = 0.5*(1 + L[:,0]/np.sqrt(L[:,0]**2 + 4.*M*Rc))*an_Om_r
        an_Om_phi = np.sign(Lz[:,0])*an_Om_theta
        
        # Frequencies (not in parallel):
        # for j in range(k):
        #     fr = r[j] + 1j*vr[j]
        #     fphi = np.sqrt(2*np.abs(Lz[j]))*(np.cos(phi[j]) + 1j*np.sin(phi[j]))

        #     om_r[i+j], a_r[i+j] = naif.find_peak_freqs(fr, t[j])
        #     om_phi[i+j], a_phi[i+j] = naif.find_peak_freqs(fphi, t[j])
        #--------------------
        # Calculate 
        #--------------------
        # Estimate frequencies (in parallel)
        file_freqs = open(file_freqs_name, 'a')

        om_r = np.zeros((k,n_freqs))
        om_phi = np.zeros((k,n_freqs))
        
        a_r = np.zeros((k,n_freqs), dtype=np.complex128)
        a_phi = np.zeros((k,n_freqs), dtype=np.complex128)
        
        f_time = t
        
        # for r:
        # this is seen as a global array by the function func_calc_freqs:
        f_tseries = r + 1j*vr
        if __name__ == "__main__":
            with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as pool:
                for row, out_om, out_a in pool.map(func_calc_freqs, range(k)):
                    om_r[row] = out_om
                    a_r[row] = out_a
        #--------------------
        # for phi:
        f_tseries = np.sqrt(2*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
        if __name__ == "__main__":
            with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as pool:
                for row, out_om, out_a in pool.map(func_calc_freqs, range(k)):
                    om_phi[row] = out_om
                    a_phi[row] = out_a
        # -----------------
        # For radial component, let's take the leading frequency:
        out_om_r = np.abs(om_r[:,0])
        out_a_r = np.abs(a_r[:,0])
        # For the azimuthal:
        out_om_phi = np.zeros(k)
        out_a_phi = np.zeros(k)
        for j in range(k):
            idx_fund = np.where(np.abs(om_phi[j]) > np.abs(om_r[j])/2.)[0]
            out_om_phi[j] = om_phi[j][idx_fund][0]
            out_a_phi[j] = np.abs(a_phi[j][idx_fund][0])

        # compare with analytical:
        delta_om_r = np.abs((an_Om_r - out_om_r)/an_Om_r)
        delta_om_phi = np.abs((an_Om_phi - out_om_phi)/an_Om_phi)

        print ('delta r:', delta_om_r)
        print ('delta phi:', delta_om_phi)
        np.savetxt(file_freqs, np.column_stack([np.arange(i,iend),
                                                out_om_r, out_om_phi,
                                                out_a_r, out_a_phi]), 
                   fmt='%6d'+' %17.8e'*4)
        file_freqs.close()
        
        this_iter += 1
        comp_time = calc_comp_time(start,this_iter ,max_iter)
        print("time until now: %s(s), time left: %5.2f(h), estimated finish time: %s"%comp_time)
        print ('---------------------------')

end = time.time()
print("total time elapsed: %s(s), total time elapsed: %s(h)"%(round(end-start,4),
                                                              round((end-start)/3600.,4)))
