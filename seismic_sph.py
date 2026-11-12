# seismic_sph module
# Ross Turner, 25 March 2026

# import packages
import os, h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import product
from scipy.signal import butter, filtfilt, lfilter

from pysph.base.nnps import LinkedListNNPS
from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme
from pysph.solver.utils import load
from pyzoltan.core.carray import UIntArray
from scipy.spatial import cKDTree

import pyprop8 as pp
from pyprop8 import utils

# fixed parameters used in the smoothed particle hydrodynamical simulation
dim = 3
nboundary_layers = 1
hdx = 1.3
ro = 1000.0
gamma = 7.0
alpha = 0.25
beta = 0.0
c0 = 10.0 * np.sqrt(2.0 * 9.81 * 0.55) #1400.

# numerical tolerance parameters
tol = 1e-9
buff = 10
kern_sigma = 2 # maximum kernel dimension
smooth_sigma = 7.5 # maximum smoothing dimension for post-processing


## Define functions to print coloured messages
class Colors:
    DogderBlue = (30, 144, 255)
    Green = (0,200,0)
    Orange = (255, 165, 0)

def _join(*values):
    return ";".join(str(v) for v in values)

def color_text(s, c, base=30):
    template = "\x1b[{0}m{1}\x1b[0m"
    t = _join(base+8, 2, _join(*c))
    return template.format(t, s)


# Define classes and functions to quickly read .hdf5 data.
def fastload(path):
    if not path.endswith('.hdf5'):
        return load(path)
    else:
        with h5py.File(path, 'r') as f:
            arrays = {}
            for k in f['particles']:
                group = f['particles'][k]['arrays']
                # Load each sub-array if array is a group
                if isinstance(group, h5py.Group):
                    arrays[k] = LiteArray({key: group[key][()] for key in group.keys()})
                else:
                    arrays[k] = LiteArray(group[()])
            solver_data = {}
            if 'solver_data' in f and 't' in f['solver_data']: # this path has not been checked
                solver_data['t'] = f['solver_data']['t'][()]
        
        return {'arrays': arrays, 'solver_data': solver_data}
    
class LiteArray:
    def __init__(self, g):
        self.x = g['x'][:]
        self.y = g['y'][:]
        self.z = g['z'][:]
        self.h = g['h'][:]
        self.rho = g['rho'][:]
        self.p = g['p'][:] if 'p' in g else None


## Define customised DamBreak class for arbitrary geometry in cartesian coordinates with initially stationary fluid based on the example in PySPH
class DamBreak3D(Application):
    # optionally modify geometry from default simulation
    def __init__(self, x, y, z, t, dx, dt, set_fluid, set_obstacle, output_times, filename):
        # set values of user defined variables
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.dx = dx
        self.dt = dt
        self.set_fluid = set_fluid
        self.set_obstacle = set_obstacle
        self.output_times = output_times
        # call __init__ function of sph Application class
        Application.__init__(self, fname=filename)
        
    # set geometry of tank
    def consume_user_options(self):
        self.geom = DamBreak3DGeometry(
                                       container_height=self.z,
                                       container_width=self.y,
                                       container_length=self.x,
                                       fluid_column_height=self.z,
                                       fluid_column_width=self.y,
                                       fluid_column_length=self.x,
                                       obstacle_center_x=self.x/2.,
                                       obstacle_center_y=0,
                                       obstacle_length=self.x,
                                       obstacle_height=self.z,
                                       obstacle_width=self.y,
                                       nboundary_layers=nboundary_layers,
                                       with_obstacle=True,
                                       dx=self.dx, hdx=hdx, rho0=ro
                                      )
        self.co = 10.0*self.geom.get_max_speed(g=9.81)

    # set fluid properties and gravity
    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary', 'obstacle'], dim=dim, rho0=ro, c0=c0,
            h0=self.dx*hdx, hdx=hdx, gz=-9.81, alpha=alpha, beta=beta, gamma=gamma,
            hg_correction=True, tensile_correction=False
        )
        return s

    # set kernal and integrator
    def configure_scheme(self):
        s = self.scheme
        kernel = WendlandQuintic(dim=dim)
        h0 = self.dx*hdx
        s.configure(h0=h0, hdx=hdx)
        dt = 0.25*h0/(1.1 * self.co)
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=self.t, dt=self.dt,
            adaptive_timestep=True, n_damp=50,
            output_at_times=self.output_times
        )

    # set locations of fluid and obstacles
    def create_particles(self):
        fluid, boundary, obstacle = self.geom.create_particles()
        
        indices = range(0, fluid.x.size)
        indices = np.delete(indices, self.set_fluid(np.asarray(fluid.x), np.asarray(fluid.y), np.asarray(fluid.z), self.dx/4.)) # tolerance to ensure fluid not on boundary
        fluid.remove_particles(indices)
        indices = range(0, obstacle.x.size)
        indices = np.delete(indices, self.set_obstacle(np.asarray(obstacle.x), np.asarray(obstacle.y), np.asarray(obstacle.z), self.dx/4.))
        if len(indices) == obstacle.x.size:
            indices = range(1, obstacle.x.size) # keep first particle to prevent numerical issues
        obstacle.remove_particles(indices)
        
        print("3D simulation with %d fluid, %d boundary, %d obstacle particles" % (fluid.get_number_of_particles(), boundary.get_number_of_particles(), obstacle.get_number_of_particles()))
        return [fluid, boundary, obstacle]


## Define function to run smoothed particle hydrodynamcial simulation; all inputs in SI units
def run_seismic_sph(x, y, z, t, dx, dt, set_fluid, set_obstacle, output_times=None, filename='seismic_sph'):
    if not isinstance(output_times, (np.ndarray, list)):
        output_times = [t]
    app = DamBreak3D(x, y, z, t, dx, dt, set_fluid, set_obstacle, output_times=output_times, filename=filename)
    app.run()


## Define function to plot output of simulation at single time step
def plot_seismic_sph(x, y, z, step, particle_type='fluid', filename=None):
    # modify the default parameters of np.load
    np_load = np.load
    np.load = lambda *a,**k: np_load(*a, allow_pickle=True, **k)

    # read-in data file at specified time step
    if filename == None:
        data = fastload('seismic_sph_output/seismic_sph_'+str(int(step))+'.hdf5')
    else:
        data = fastload(filename+'_output/'+filename+'_'+str(int(step))+'.hdf5')
    np.load = np_load

    # read in arrays of particles
    particle_arrays = data['arrays']
    particles = particle_arrays[particle_type]

    # create plot of hydrodynamical simulation
    fig, ax = plt.subplots(2, 1, figsize=(12, 3.5), sharex=True)
    
    rc('text', usetex=True)
    rc('font', size=13)
    rc('legend', fontsize=12.5)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    if particle_type == "fluid":
        sc0 = ax[0].scatter(particles.x, particles.y, c=particles.rho, s=0.1, cmap='viridis', vmin=990, vmax=1010)
        sc1 = ax[1].scatter(particles.x, particles.z, c=particles.rho, s=0.1, cmap='viridis', vmin=990, vmax=1010)
    else:
        sc0 = ax[0].scatter(particles.x, particles.y, s=0.1, c='midnightblue')
        sc1 = ax[1].scatter(particles.x, particles.z, s=0.1, c='midnightblue')
    
    ax[0].set_xlim([0, x])
    ax[0].set_ylim([-y/2., y/2.])
    ax[0].set_ylabel(r'$y$ (m)')

    #ax[1].set_xlim([0, x])
    ax[1].set_ylim([0, z])
    ax[1].set_xlabel(r'$x$ (m)')
    ax[1].set_ylabel(r'$z$ (m)')
    
    plt.subplots_adjust(hspace=0.25, wspace=0)
    
    if particle_type == "fluid":
        c = plt.colorbar(sc0, ax=ax.ravel().tolist(), pad=0.025)
        c.set_label(r'$\rho$ (kg m$^{-3}$)')

    plt.show()


## Define functions to find impulse on boundary/obstacle particles due to interactions with water--ice interface
def seismic_sph_interactions(filename=None, delta=0.01, dx=0.05, dy=0.05, dz=0.05, xrange=[4,6], yrange=None, zrange=None, boundary_layer=False):
    # read in time and fluid properties from file
    time_vector, iter_vector = __seismic_sph_times(filename=filename)
    fluid_vector, boundary_vector, obstacle_vector = __seismic_sph_particles(time_vector, iter_vector, [], [], [], filename=filename, start=0, buffer=buff)
    
    # derive interface particles, unit normals, and effective areas
    interface_index, normals, effective_area = __seismic_sph_interfaces(boundary_vector, obstacle_vector, delta=delta, xrange=xrange, yrange=yrange, zrange=zrange)
    
    # derive collision data from SPH outputs
    t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector = __seismic_sph_interactions(time_vector, iter_vector, fluid_vector, boundary_vector, obstacle_vector, interface_index, normals, effective_area, filename=filename, dx=dx, dy=dy, dz=dz, xrange=xrange, yrange=yrange, zrange=zrange, boundary_layer=boundary_layer)
    
    # create hdf5 dataframe to output data
    with h5py.File(filename+'_interactions.hdf5', 'w') as f:
        f.create_dataset('t', data=t_vector)
        f.create_dataset('x', data=x_vector)
        f.create_dataset('y', data=y_vector)
        f.create_dataset('z', data=z_vector)
        f.create_dataset('fx', data=fx_vector)
        f.create_dataset('fy', data=fy_vector)
        f.create_dataset('fz', data=fz_vector)
    
    return None


# Define function to extract output time steps
def __seismic_sph_times(filename=None):
    # create vectors for times and index numbers
    time_vector = []
    iter_vector = []
    
    # read-in log file (fast option; takes latest run only)
    if filename == None:
        path = 'seismic_sph_output/seismic_sph.log'
    else:
        path = filename+'_output/'+filename+'.log'
    if os.path.exists(path):
        df = pd.read_csv(path, sep='|')
        
        for i in range(0, len(df)):
            if (df.iloc[i,0] == '----------------------------------------------------------------------'):
                time_vector = []
                iter_vector = []
            else:
                try:
                    if (df.iloc[i,3].split(',')[0].split(' ')[0] == 'Writing') and float(df.iloc[i,3].split(',')[0].split(' ')[4]):
                        time_vector.append(df.iloc[i,3].split(',')[0].split(' ')[4])
                        iter_vector.append(df.iloc[i,3].split(',')[1].split(' ')[2])
                except:
                    pass
    
    # read-in data for .hdf5 files (slow option; risky if multiple runs in a single runs)
    else:
        # modify the default parameters of np.load
        np_load = np.load
        np.load = lambda *a,**k: np_load(*a, allow_pickle=True, **k)
        
        if filename == None:
            path = 'seismic_sph_output'
        else:
            path = filename+'_output'
        
        # list filenames in folder
        file_iters = []
        filenames = os.listdir(path)
        for f in filenames:
            if f.endswith('.hdf5') or f.endswith('.npz'):
                try:
                    iter_num = int(os.path.splitext(f)[0].rsplit("_", 1)[1])
                    file_iters.append((iter_num, f))
                except ValueError:
                    pass  # skip files that don't match the pattern
                    
        # sort files by iteration number
        file_iters.sort(key=lambda x: x[0])
        
        # load files in numeric order
        for iter_num, f in file_iters:
            file = os.path.join(path, f)
            try:
                data = fastload(file)
                time_vector.append(data['solver_data']['t'])
                iter_vector.append(int((os.path.splitext(os.path.basename(f))[0]).rsplit("_", 1)[1]))
            except (KeyError, FileNotFoundError, OSError):
                pass
                
        # revert to orginal definition
        np.load = np_load

    # convert to arrays
    time_vector, iter_vector = np.asarray(time_vector).astype(float), np.asarray(iter_vector).astype(float)
    
    # find time spacing (excluding any odd outputs)
    dt = np.median(np.diff(time_vector))
    t_unique = np.unique(np.rint(time_vector / dt) * dt)
    idx = np.abs(time_vector[:, None] - t_unique[None, :]).argmin(axis=0)
    
    return time_vector[idx], iter_vector[idx]


# Define function to read-in locations and velocites of particles at each time step
def __seismic_sph_particles(time_vector, iter_vector, fluid_vector, boundary_vector, obstacle_vector, filename=None, start=0, buffer=buff):
    # modify the default parameters of np.load
    np_load = np.load
    np.load = lambda *a,**k: np_load(*a, allow_pickle=True, **k)

    # read-in data file at specified time step
    for i in range(start, min(len(time_vector), start + buffer)):
        if filename == None:
            data = fastload('seismic_sph_output/seismic_sph_'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
        else:
            data = fastload(filename+'_output/'+filename+'_'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
        
        # store fluid particle properties in datafile
        fluid_vector.append(data['arrays']['fluid'])
        if i == 0:
            boundary_vector.append(data['arrays']['boundary'])
            obstacle_vector.append(data['arrays']['obstacle'])

    # revert to orginal definition
    np.load = np_load
    
    return fluid_vector, boundary_vector, obstacle_vector


# Define function to find unit normals and effective areas to each boundary/obstacle particle on the interface
def __seismic_sph_interfaces(boundary_vector, obstacle_vector, delta=0.01, xrange=None, yrange=None, zrange=None):
    # find particle locations
    x_slice = np.concatenate((boundary_vector[0].x, obstacle_vector[0].x))
    y_slice = np.concatenate((boundary_vector[0].y, obstacle_vector[0].y))
    z_slice = np.concatenate((boundary_vector[0].z, obstacle_vector[0].z))
    
    # define restricted range for computational efficiency
    range_index = __range_restriction(boundary_vector[0].h[0], x_slice, y_slice, z_slice, xrange=xrange, yrange=yrange, zrange=zrange)

    # convert locations to indices
    loc_vector = np.column_stack(( ((x_slice + delta/2)//delta).astype(int), ((y_slice + delta/2)//delta).astype(int), ((z_slice + delta/2)//delta).astype(int) ))
    loc_tuple = set(map(tuple, loc_vector))
        
    # determine positions of potential neighbours
    vals_3 = [-1, 0, 1]
    vals_5 = [-2, -1, 0, 1, 2]
    neighbour_offsets_3 = np.array([p for p in product(vals_3, repeat=3) if p != (0,0,0)], dtype=int) # 3x3x3 cube, missing centre
    neighbour_offsets_5 = np.array([p for p in product(vals_5, repeat=3) if p != (0,0,0)], dtype=int) # 5x5x5 cube, missing centre
    neighbour_index = np.zeros((len(neighbour_offsets_3), len(x_slice)), dtype=bool)
    
    # check if each neighbour exists for set of offsets
    for i in range(0, len(neighbour_offsets_3)):
        neighbours = loc_vector[range_index] + neighbour_offsets_3[i]
        neighbour_index[i, range_index] = [tuple(n) in loc_tuple for n in neighbours]
        
    # ignore locations comprising the rectangular boundary of the simulation
    x_min, x_max = np.min(x_slice), np.max(x_slice)
    y_min, y_max = np.min(y_slice), np.max(y_slice)
    z_min, z_max = np.min(z_slice), np.max(z_slice)
    internal_index = ( (x_min < x_slice) & (x_slice < x_max) & (y_min < y_slice) & (y_slice < y_max) & (z_min < z_slice) & (z_slice < z_max) )

    # find boundary/obstacle particles on the interface (and internal to the simulation)
    interface_index = np.logical_and(np.logical_and(np.logical_not(np.all(neighbour_index, axis=0)), internal_index), range_index)
    interface_vector = loc_vector[interface_index]
        
    # find unit normal to each interface particle
    normals = np.zeros_like(interface_vector, dtype=float)
    for i in range(0, len(interface_vector)):
        n = np.zeros(3)
        for j in range(0, len(neighbour_offsets_5)):
            neighbour = tuple(interface_vector[i] + neighbour_offsets_5[j])
            if neighbour not in loc_tuple:
                n += neighbour_offsets_5[j]/(neighbour_offsets_5[j, 0]**2 + neighbour_offsets_5[j, 1]**2 + neighbour_offsets_5[j, 2]**2) # weight by norm squared
        # normalise
        normals[i, :] = n / np.sqrt(n[0]**2 + n[1]**2 + n[2]**2) if np.sqrt(n[0]**2 + n[1]**2 + n[2]**2) > 0 else np.zeros(3)
    
    # find the effective surface area of each particle on the interface
    effective_area = delta**2 * (np.abs(normals[:, 0]) + np.abs(normals[:, 1]) + np.abs(normals[:, 2]))
        
    return interface_index, normals, effective_area


# Define function to find particles that will collide with water--ice interfaces
def __seismic_sph_interactions(time_vector, iter_vector, fluid_vector, boundary_vector, obstacle_vector, interface_index, normals, effective_area, filename=None, dx=0.05, dy=0.05, dz=0.05, xrange=None, yrange=None, zrange=None, boundary_layer=True):
    # define vectors to store collision data
    t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector = [], [], [], [], [], [], []
    
    # find particles on the interface (assuming it does not move)
    x_slice = np.concatenate([boundary_vector[0].x, obstacle_vector[0].x])[interface_index]
    y_slice = np.concatenate([boundary_vector[0].y, obstacle_vector[0].y])[interface_index]
    z_slice = np.concatenate([boundary_vector[0].z, obstacle_vector[0].z])[interface_index]
    loc_interface = np.column_stack((x_slice, y_slice, z_slice))
    
    # define bounds of coarse(r) cartesian grid
    min_x, max_x = int((np.min(x_slice) + dx/2)//dx), int((np.max(x_slice) + dx/2)//dx)
    min_y, max_y = int((np.min(y_slice) + dy/2)//dy), int((np.max(y_slice) + dy/2)//dy)
    min_z, max_z = int((np.min(z_slice) + dz/2)//dz), int((np.max(z_slice) + dz/2)//dz)
    
    # define grid of effective areas and unit normals
    area_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    nx_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    ny_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    nz_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))

    # find location of each interface particle in coarse(r) cartesian grid
    x_index = ((x_slice + dx/2)//dx - min_x).astype(int)
    y_index = ((y_slice + dy/2)//dy - min_y).astype(int)
    z_index = ((z_slice + dz/2)//dz - min_z).astype(int)
    
    # select a random interface particle for each location in the coarse(r) grid
    loc_unique, inverse = np.unique(np.column_stack((x_index, y_index, z_index)), axis=0, return_inverse=True)
    loc_random = np.zeros(len(loc_unique), dtype=int)
    for j in range(0, len(loc_unique)):
        # select a random particle
        loc_random[j] = np.random.choice(np.flatnonzero(inverse == j))
    
    # fine total area of interface particles and average unit normal for each grid point
    np.add.at(area_grid, (x_index, y_index, z_index), effective_area)
    np.add.at(nx_grid, (x_index, y_index, z_index), normals[:, 0])
    np.add.at(ny_grid, (x_index, y_index, z_index), normals[:, 1])
    np.add.at(nz_grid, (x_index, y_index, z_index), normals[:, 2])
    length = np.sqrt(nx_grid**2 + ny_grid**2 + nz_grid**2)
    nx_grid, ny_grid, nz_grid = nx_grid/np.maximum(1, length), ny_grid/np.maximum(1, length), nz_grid/np.maximum(1, length)
        
    # find time-averaged pressure and force vectors
    for i in range(1, len(time_vector)):
        print(i)

        # find pressure and force vector of boundary/obstacle particle
        p_slice = __seismic_sph_pressure(loc_random, x_slice, y_slice, z_slice, fluid_vector[i], loc_interface, boundary_layer=boundary_layer, xrange=xrange, yrange=yrange, zrange=zrange)
        
        # add grid locations and forces to output lists
        t_vector.extend([time_vector[i]] * len(loc_unique[:, 0]))
        x_vector.extend((loc_unique[:, 0] + min_x) * dx)
        y_vector.extend((loc_unique[:, 1] + min_y) * dy)
        z_vector.extend((loc_unique[:, 2] + min_z) * dz)
        fx_vector.extend(p_slice * nx_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]] * area_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]])
        fy_vector.extend(p_slice * ny_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]] * area_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]])
        fz_vector.extend(p_slice * nz_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]] * area_grid[loc_unique[:, 0], loc_unique[:, 1], loc_unique[:, 2]])
        
        # remove current time step and read-in one extra time step
        fluid_vector[i-1] = None
        fluid_vector, boundary_vector, obstacle_vector = __seismic_sph_particles(time_vector, iter_vector, fluid_vector, boundary_vector, obstacle_vector, filename=filename, start=i+buff-1, buffer=1)

    return t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector


# Define function to reconstruct pressure from fluid particles.
def __seismic_sph_pressure(loc_random, x_slice, y_slice, z_slice, fluid_vector, loc_interface, boundary_layer=False, xrange=None, yrange=None, zrange=None):
    # define kernel dimension
    h = fluid_vector.h[0]
    r_kern = kern_sigma * h
    r_smooth = smooth_sigma * h
    
    # build a KD tree for the fluid particles
    range_index = __range_restriction(h, fluid_vector.x, fluid_vector.y, fluid_vector.z, xrange=xrange, yrange=yrange, zrange=zrange)
    valid_indices = np.flatnonzero(range_index)
    fluid_tree = cKDTree(np.column_stack((fluid_vector.x[valid_indices], fluid_vector.y[valid_indices], fluid_vector.z[valid_indices])))
    
    p_interface = np.zeros(len(loc_random))
    for k, idx in enumerate(loc_random):
        # find neighbors within smoothing radius
        nbrs = fluid_tree.query_ball_point([x_slice[idx], y_slice[idx], z_slice[idx]], r_smooth)

        if len(nbrs) == 0:
            continue

        p_sum = 0.
        w_sum = 0.
        particle_within_kernel = False
        
        # calculate weighted pressure of fluid particles within smoothing radius (but not in boundary layer)
        for j in nbrs:
            global_j = valid_indices[j]

            dx = fluid_vector.x[global_j] - x_slice[idx]
            dy = fluid_vector.y[global_j] - y_slice[idx]
            dz = fluid_vector.z[global_j] - z_slice[idx]
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            # apply weight to particles
            w = 1.
            p_sum += fluid_vector.p[global_j] * w
            w_sum += w
            if r < r_kern:
                particle_within_kernel = True

        if w_sum > 0 and particle_within_kernel:
            p_interface[k] = p_sum / w_sum

    return p_interface
    

# Define function to restrict the range of particle locations.
def __range_restriction(h, x_slice, y_slice, z_slice, xrange=None, yrange=None, zrange=None):
    # define kernel dimension
    r_smooth = smooth_sigma * h
    
    # define restricted range for computational efficiency
    range_index = np.zeros_like(x_slice, dtype=bool) + 1
    if isinstance(xrange, (list, np.ndarray)):
        range_index[np.logical_or(x_slice < xrange[0] - r_smooth, xrange[1] + r_smooth < x_slice)] = 0
    if isinstance(yrange, (list, np.ndarray)):
        range_index[np.logical_or(y_slice < yrange[0] - r_smooth, yrange[1] + r_smooth < y_slice)] = 0
    if isinstance(zrange, (list, np.ndarray)):
        range_index[np.logical_or(z_slice < zrange[0] - r_smooth, zrange[1] + r_smooth < z_slice)] = 0
        
    return range_index


def ___seismic_sph_pressure(interface_index, fluid_vector, boundary_vector, obstacle_vector):
    
    # define kernel dimension
    h = fluid_vector.h[0]
    r_kern = 2 * h
    r_smooth = 5 * h
    
    # force pysph linked list to consider more distant particles
    fluid_vector.h[:], boundary_vector.h[:], obstacle_vector.h[:] = r_smooth, r_smooth, r_smooth
    
    # apply kernel to nearest neighbours to interface particles
    p_boundary = __seismic_sph_kernel(fluid_vector, boundary_vector, r_kern, r_smooth, interface_index[:len(boundary_vector.x)])
    p_obstacle = __seismic_sph_kernel(fluid_vector, obstacle_vector, r_kern, r_smooth, interface_index[len(boundary_vector.x):])
    p_interface = np.concatenate((p_boundary, p_obstacle))

    return p_interface
    

# Define function to apply smoothing kernel to pressure of fluid particles.
def __seismic_sph_kernel(particle_src, particle_dst, r_kern, r_smooth, interface_index=None):
    
    # create arrays for outputs
    p_smooth, w_smooth = np.zeros(len(particle_dst.x)), np.zeros(len(particle_dst.x))
    
    # create instance of pysph linked list
    nnps = LinkedListNNPS( dim=dim, particles=[particle_src, particle_dst] )
    nnps.update()
    
    # apply kernel to nearest neighbours of each specified particle
    if not isinstance(interface_index, (list, np.ndarray)):
        indices = np.arange(0, len(particle_dst.x))
    else:
        indices = np.arange(0, len(particle_dst.x))[interface_index]
    for i in indices:
        nbrs = UIntArray()

        # compare each nearby particle to destination particle
        nnps.get_nearest_particles( 0, 1, i, nbrs )
        x_i, y_i, z_i = particle_dst.x[i], particle_dst.y[i], particle_dst.z[i],
        
        # calculate weighted pressure from nearby fluid particles
        particle_within_kernel = 0
        for j in nbrs.get_npy_array():
            dx = particle_src.x[int(j)] - x_i
            dy = particle_src.y[int(j)] - y_i
            dz = particle_src.z[int(j)] - z_i
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            if r < r_smooth:
                w = 1.
                p_smooth[i] += particle_src.p[int(j)] * w
                w_smooth[i] += w
            if r < r_kern:
                particle_within_kernel = 1
        p_smooth[i] *= particle_within_kernel

    mask = w_smooth > 0
    p_smooth[mask] /= w_smooth[mask]
    
    if not isinstance(interface_index, (list, np.ndarray)):
        return p_smooth
    else:
        return p_smooth[interface_index]


## Define functions to find force vectors for particle collisions; all inputs in SI units, z0 is a depth
def seismic_sph_forces(x0, y0, z0, filename=None, dx=0.1, dy=0.1, dz=0.1, average_height=False, crit_force=1e-6, f_cutoff=1.):
    # define vectors to store collision data
    t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector = [], [], [], [], [], [], []
    
    # read particle interaction datafile
    with h5py.File(filename+'_interactions.hdf5', 'r') as f:
        t_h5 = f['t'][:]
        x_h5 = f['x'][:]
        y_h5 = f['y'][:]
        z_h5 = f['z'][:]
        try:
            fx_h5 = f['fx'][:]
            fy_h5 = f['fy'][:]
        except:
            # simulation used for gravity loading
            fx_h5, fy_h5 = 0*f['fz'][:], 0*f['fz'][:]
        fz_h5 = f['fz'][:]

    # calculate the maximum and cut-off force for the interactions
    f_h5 = np.sqrt(fx_h5**2 + fy_h5**2 + fz_h5**2)
    crit_index = f_h5 > crit_force*np.max(f_h5)
    
    # define reduced set of times, locations and forces
    t_slice = t_h5[crit_index]
    x_slice, y_slice, z_slice = x_h5[crit_index], y_h5[crit_index], z_h5[crit_index]
    fx_slice, fy_slice, fz_slice = fx_h5[crit_index], fy_h5[crit_index], fz_h5[crit_index]
    f_slice = f_h5[crit_index]
    
    # find unique times for interactions
    t_unique = np.unique(t_slice)
    # find time spacing (excluding any odd outputs)
    dt = np.median(np.diff(t_unique))
    t_unique = np.unique(np.rint(t_slice / dt) * dt)
    
    # define bounds of coarse(r) cartesian grid (similar to previous function)
    min_x, max_x = int((np.min(x_slice) + dx/2 - x0)//dx), int((np.max(x_slice) + dx/2 - x0)//dx)
    min_y, max_y = int((np.min(y_slice) + dy/2 - y0)//dy), int((np.max(y_slice) + dy/2 - y0)//dy)
    if average_height == True:
        min_z, max_z = 0, 0
    else:
        min_z, max_z = int((np.min(z_slice) + dz/2 - z0)//dz), int((np.max(z_slice) + dz/2 - z0)//dz)
    
    # define grid of force vectors
    fx_grid = np.zeros((len(t_unique), max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    fy_grid = np.zeros((len(t_unique), max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    fz_grid = np.zeros((len(t_unique), max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
    
    # add forces to appropriate grid point
    for i in range(0, len(t_unique)):
        # find indices (as a mask) with the correct time step
        time_index = np.abs(t_slice - t_unique[i]) < tol
    
        if np.sum(time_index) > 0:
            # add each force vector to a cartesian grid
            x_index = ((x_slice[time_index] + dx/2 - x0)//dx - min_x).astype(int)
            y_index = ((y_slice[time_index] + dy/2 - y0)//dy - min_y).astype(int)
            if average_height == True:
                z_index = 0
            else:
                z_index = ((z_slice[time_index] + dz/2 - z0)//dz - min_z).astype(int)
            np.add.at(fx_grid, (i, x_index, y_index, z_index), fx_slice[time_index])
            np.add.at(fy_grid, (i, x_index, y_index, z_index), fy_slice[time_index])
            np.add.at(fz_grid, (i, x_index, y_index, z_index), fz_slice[time_index])
        
    # define (and apply) eighth-order Butterworth filter to remove slow-varying signal
    if f_cutoff > 1:
        b, a = butter(8, f_cutoff / (0.5 * (1. / dt)), btype='low')
        w = 0.5 * (1 - np.cos(np.pi * (t_unique - t_unique[0]) / (t_unique[-1] - t_unique[0])))
        fx_detrended = w[:,None,None,None] * lfilter(b, a, fx_grid, axis=0) + (1 - w[:,None,None,None]) * lfilter(b, a, fx_grid[::-1], axis=0)[::-1]
        fy_detrended = w[:,None,None,None] * lfilter(b, a, fy_grid, axis=0) + (1 - w[:,None,None,None]) * lfilter(b, a, fy_grid[::-1], axis=0)[::-1]
        fz_detrended = w[:,None,None,None] * lfilter(b, a, fz_grid, axis=0) + (1 - w[:,None,None,None]) * lfilter(b, a, fz_grid[::-1], axis=0)[::-1]
        
        # scale data based on initial forces acting on each cell
        fx_detrended[:,:,:,:] = fx_detrended[:,:,:,:] - fx_detrended[0,:,:,:][None,:,:,:]
        fy_detrended[:,:,:,:] = fy_detrended[:,:,:,:] - fy_detrended[0,:,:,:][None,:,:,:]
        fz_detrended[:,:,:,:] = fz_detrended[:,:,:,:] - fz_detrended[0,:,:,:][None,:,:,:]
    else:
        fx_detrended, fy_detrended, fz_detrended = fx_grid, fy_grid, fz_grid
    
    # add non-zero grid points to vector outputs
    f_detrended = np.sqrt(fx_detrended**2 + fy_detrended**2 + fz_detrended**2)
    f_crit = crit_force*np.max(f_detrended)
    for i in range(0, len(t_unique)):
        # find unique non-zero locations in grid
        x_unique, y_unique, z_unique = np.nonzero(np.abs(f_detrended[i,:,:,:]) > f_crit)
        
        # add grid locations and forces to output lists
        if len(x_unique) > 0:
            t_vector.extend([t_unique[i]] * len(x_unique))
            x_vector.extend((x_unique + min_x) * dx)
            y_vector.extend((y_unique + min_y) * dy)
            z_vector.extend((z_unique + min_z) * dz)
            fx_vector.extend(fx_detrended[i, x_unique, y_unique, z_unique])
            fy_vector.extend(fy_detrended[i, x_unique, y_unique, z_unique])
            fz_vector.extend(fz_detrended[i, x_unique, y_unique, z_unique])
            
    # create pandas dataframe to output data
    forces_df = pd.DataFrame(columns=['t', 'loc_1', 'loc_2', 'loc_3', 'force_1', 'force_2', 'force_3'])
    forces_df['t'] = np.array(t_vector)
    forces_df['loc_1'] = np.array(x_vector)
    forces_df['loc_2'] = np.array(y_vector)
    if average_height == True:
        # calculate force-weighted average height
        weight = np.sqrt(np.array(fx_vector)**2 + np.array(fy_vector)**2 + np.array(fz_vector)**2)
        forces_df['loc_3'] = np.nansum(weight * np.array(z_vector))/np.nansum(weight) - z0
    else:
        forces_df['loc_3'] = np.array(z_vector)
    forces_df['force_1'] = np.array(fx_vector)
    forces_df['force_2'] = np.array(fy_vector)
    forces_df['force_3'] = np.array(fz_vector)
    
    return forces_df


## Define functions to calculate synthetic seismic signals using pyprop8; all inputs in SI units, z is a depth
def seismic_signals(x, y, z, earth_model=None, forces_df=None, filename=None, duration=10., sampling_rate=500., spec=False):
    # read collision datafile if not passed as an input
    if not isinstance(forces_df, pd.DataFrame):
        forces_df = pd.read_csv(filename+'.csv', sep=',', index_col=0)
    time_vector, x_vector, y_vector, z_vector = forces_df['t'].values, forces_df['loc_1'].values, forces_df['loc_2'].values, -forces_df['loc_3'].values # z-coordinate is depth (need to flip); the force is in cartesian oriented sensibly
    f1_vector, f2_vector, f3_vector = -forces_df['force_1'].values, -forces_df['force_2'].values, -forces_df['force_3'].values # input forces are normal response of medium (need to flip)
    
    # create Earth layer model if none passed in
    if earth_model == None:
        earth_model = pp.LayeredStructureModel([(0, 3.6, 1.8, 0.916)], interface_depth_form=True)
    
    # test if seismometer is above or below the channel (this is to avoid a hardcoded check in pyprop)
    if z > z_vector[0]:
        x_vector, y_vector = -x_vector, -y_vector
        z = 2*z_vector[0] - z
        f1_vector, f2_vector, f3_vector = -f1_vector, -f2_vector, -f3_vector

    # seismometer locations
    loc_vector = []
    index_vector = np.zeros_like(forces_df.index)
    for i in range(0, len(forces_df.index)):
        loc = [x_vector[i], y_vector[i]]
        if loc in loc_vector:
            index_vector[i] = [l for l in loc_vector].index(loc)
        else:
            loc_vector.append(loc)
            index_vector[i] = len(loc_vector) - 1
    loc_vector = np.array(loc_vector)

    # define receiver locations
    receivers = pp.ListOfReceivers((x - loc_vector[:,0])/1000., (y - loc_vector[:,1])/1000., z/1000.)
                    # correction to seismometer locations
                    
    # find time spacing (excluding any odd outputs)
    dt = np.median(np.diff(np.unique(time_vector)))
    dt_bin = int(1./(sampling_rate*dt) + 0.5) # number of simulation bins per sample (integer; work required for non-integer ratios)
    # sampling in time and frequency
    nt = int(duration*sampling_rate) # number of time-series points
    freqs = 10**np.arange(-1, 3 + 1e-9, 0.01)
    
    # add seismic source to model at each timestep
    seismograms, derivatives, spectra = [], [], []
    for i in range(0, 3):
        # define moment tensor and force vector
        M = np.array([[0,0,0], [0,0,0], [0,0,0]])
        if i == 0:
            F = np.array([[1e-9], [0], [0]])
        elif i == 1:
            F = np.array([[0], [1e-9], [0]])
        else:
            F = np.array([[0], [0], [1e-9]])
        
        for j in range(0, dt_bin):
            # create source object
            source = pp.PointSource(0., 0., z_vector[0]/1000., M, F, j*(1./sampling_rate)/dt_bin)

            # create discplacement waveforms (and derivatives) at the location of each seismometer
            times, seismogram, derivative = pp.compute_seismograms(earth_model, source, receivers, nt, 1./sampling_rate, derivatives=pp.DerivativeSwitches(time=True), squeeze_outputs=False, show_progress=False)
            seismograms.append(seismogram)
            derivatives.append(derivative)

            # create velocity spectrum at the location of each seismometer
            if spec:
                spectrum = pp.compute_spectra(earth_model, source, receivers, 2*np.pi*freqs, squeeze_outputs=False, show_progress=False)
                spectra.append(spectrum)

    # convert lists to arrays
    seismograms = np.asarray(seismograms)
    derivatives = np.asarray(derivatives)
    spectra = np.asarray(spectra)
    
    # appy time shifts and weight seimograms/spectra by forces
    displacements, velocities, velocity_spectra = __time_shifter(time_vector, f1_vector, f2_vector, f3_vector, index_vector, seismograms, derivatives, spectra, sampling_rate, dt, spec, freqs)

    # create pandas dataframe to output data
    seismogram_df = pd.DataFrame(columns=['t', 'disp_1', 'disp_2', 'disp_3', 'vel_1', 'vel_2', 'vel_3'])
    seismogram_df['t'] = times
    seismogram_df['disp_1'], seismogram_df['disp_2'], seismogram_df['disp_3'] = displacements[0,:]*1e-3, displacements[1,:]*1e-3, displacements[2,:]*1e-3
    seismogram_df['vel_1'], seismogram_df['vel_2'], seismogram_df['vel_3'] = velocities[0,:]*1e-3, velocities[1,:]*1e-3, velocities[2,:]*1e-3
    if spec:
        spectra_df = pd.DataFrame(columns=['f', 'spec_1', 'spec_2', 'spec_3'])
        spectra_df['f'] = freqs
        spectra_df['spec_1'], spectra_df['spec_2'], spectra_df['spec_3'] = velocity_spectra[0,:]*1e-6, velocity_spectra[1,:]*1e-6, velocity_spectra[2,:]*1e-6
        return seismogram_df, spectra_df
    else:
        return seismogram_df


# Define function to appy time shifts and weight seimograms/spectra by forces
def __time_shifter(time_vector, f1_vector, f2_vector, f3_vector, index_vector, seismograms, derivatives, spectra, sampling_rate, dt, spec, freqs):

    # find appropriate time bin and shift in samples
    dt_bin = int(1./(sampling_rate*dt) + 0.5)
    time_shift = np.floor((time_vector + dt/2) * sampling_rate).astype(int)
    time_bin = np.clip(np.floor((time_vector + dt/2 - time_shift / sampling_rate) / dt).astype(int), 0, dt_bin - 1)

    # create data structures for output seismograms and spectra
    nsamples = seismograms.shape[-1]
    displacements = np.zeros((3, nsamples))
    velocities = np.zeros((3, nsamples))
    velocity_spectra = np.zeros((3, len(freqs)), dtype=complex)
    
    for i in range(0, len(time_vector)):
        # add x, y and z-components of force to seismogram; force is in Newtons (strictly N/s/m^3; volume is handled through grid and time through bin width)
        displacements[:,time_shift[i]:] = displacements[:,time_shift[i]:] \
                        + f1_vector[i] * dt * seismograms[time_bin[i], 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]] \
                        + f2_vector[i] * dt * seismograms[time_bin[i] + dt_bin, 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]] \
                        + f3_vector[i] * dt * seismograms[time_bin[i] + 2*dt_bin, 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]]
        velocities[:,time_shift[i]:] = velocities[:,time_shift[i]:] \
                        + f1_vector[i] * dt * derivatives[time_bin[i], 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]] \
                        + f2_vector[i] * dt * derivatives[time_bin[i] + dt_bin, 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]] \
                        + f3_vector[i] * dt * derivatives[time_bin[i] + 2*dt_bin, 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]]

        # add x, y and z-components of force to spectrum; force is in Newtons (strictly N/s/m^3; volume is handled through grid and time through bin width)
        if spec:
            velocity_spectra[:,:] = velocity_spectra[:,:] \
                            + f1_vector[i] * dt * spectra[time_bin[i], 0, index_vector[i], :, :] \
                            + f2_vector[i] * dt * spectra[time_bin[i] + dt_bin, 0, index_vector[i], :, :] \
                            + f3_vector[i] * dt * spectra[time_bin[i] + 2*dt_bin, 0, index_vector[i], :, :]
    
    return displacements, velocities, velocity_spectra


# Define function to specify earth/ice layer model; all inputs in SI units, array of layers each with element: (z_top, v_p, v_s, rho)
def earth_model(layers):
    return pp.LayeredStructureModel(np.array(layers)/1000., interface_depth_form=True)


## Define custom functions for analysis in Turner et al. (2026)
def seismic_sph_forces_defaults(x0, y0, z0, filename):
    
    # define size of reduced grid
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # calculate force database (apply low pass filter to grid cell occupied by initial fluid)
    forces_df = seismic_sph_forces(x0, y0, z0, filename='{}'.format(filename), dx=dx, dy=dy, dz=dz, \
                                  average_height=True, crit_force=1e-3, f_cutoff=0)

    # exclude straight sections of channel away from collision site
    forces_df = forces_df[np.logical_and(-1.001 <= forces_df['loc_1'], forces_df['loc_1'] <= 1.001)].reset_index(drop=True)
    
    # remove vertical component of force (should be net zero)
    forces_df['force_3'] = 0
    
    return forces_df
