# seismic_sph module
# Ross Turner, 27 January 2026

# import packages
import os, h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc

from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme

from pysph.solver.utils import load

import pyprop8 as pp
from pyprop8 import utils

# fixed parameters used in the smoothed particle hydrodynamical simulation
dim = 3
sigma = 2 # critical length-scale in units of h*dx
nboundary_layers = 1
hdx = 1.3
ro = 1000.0
gamma = 7.0
alpha = 0.25
beta = 0.0
c0 = 10.0 * np.sqrt(2.0 * 9.81 * 0.55) #1400.

# fixed parameter used in bisection method to find location and normal vector to interface
tol = 1e-5
pert = 1e-3 # must be much greater than tolerance
buff = 30
kernel_frac = 0.8


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
        data = load('seismic_sph_output/seismic_sph_'+str(int(step))+'.hdf5')
    else:
        data = load(filename+'_output/'+filename+'_'+str(int(step))+'.hdf5')
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
def seismic_sph_interactions(filename=None, delta=0.01, dx=0.01, dy=0.01, dz=0.01):
    # read in time and fluid properties from file
    time_vector, iter_vector = __seismic_sph_times(filename=filename)
    boundary_vector, obstacle_vector = __seismic_sph_particles(time_vector, iter_vector, [], [], filename=filename, start=0, buffer=buff)
    
    # derive interface particles, unit normals, and effective areas
    interface_index, normals, effective_area = __seismic_sph_interfaces(boundary_vector, obstacle_vector, delta=delta)
    
    # derive collision data from SPH outputs
    t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector = __seismic_sph_interactions(time_vector, iter_vector, boundary_vector, obstacle_vector, interface_index, normals, effective_area, filename=filename, dx=dx, dy=dy, dz=dz)
    
    # create hdf5 dataframe to output data
    with h5py.File(filename+'_interactions.hdf5', 'w') as f:
        f.create_dataset('t', data=t_vector)
        f.create_dataset('x', data=x_vector)
        f.create_dataset('y', data=y_vector)
        f.create_dataset('z', data=z_vector)
        f.create_dataset('fx', data=fx_vector)
        f.create_dataset('fy', data=fy_vector)
        f.create_dataset('fz', data=fz_vector)
    
    # create pandas dataframe to output data
    #collision_df = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'fx', 'fy', 'fz'])
    #collision_df['t'] = t_vector
    #collision_df['x'] = x_vector
    #collision_df['y'] = y_vector
    #collision_df['z'] = z_vector
    #collision_df['fx'] = fx_vector
    #collision_df['fy'] = fy_vector
    #collision_df['fz'] = fz_vector
    
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
                data = load(file)
                time_vector.append(data['solver_data']['t'])
                iter_vector.append(int((os.path.splitext(os.path.basename(f))[0]).rsplit("_", 1)[1]))
            except (KeyError, FileNotFoundError, OSError):
                pass
                
        # revert to orginal definition
        np.load = np_load

    return np.asarray(time_vector).astype(float), np.asarray(iter_vector).astype(float)

# Define function to read-in locations and velocites of particles at each time step
def __seismic_sph_particles(time_vector, iter_vector, boundary_vector, obstacle_vector, filename=None, start=0, buffer=buff):
    # modify the default parameters of np.load
    np_load = np.load
    np.load = lambda *a,**k: np_load(*a, allow_pickle=True, **k)

    # read-in data file at specified time step
    for i in range(start, min(len(time_vector), start + buffer)):
        if filename == None:
            try:
                data = load('seismic_sph_output/seismic_sph_'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
            except:
                data = load('seismic_sph_output/seismic_sph_'+'{:.0f}'.format(iter_vector[i])+'.npz')
        else:
            try:
                data = load(filename+'_output/'+filename+'_'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
            except:
                data = load(filename+'_output/'+filename+'_'+'{:.0f}'.format(iter_vector[i])+'.npz')
        
        # store fluid particle properties in datafile
        boundary_vector.append(data['arrays']['boundary'])
        obstacle_vector.append(data['arrays']['obstacle'])

    # revert to orginal definition
    np.load = np_load
    
    return boundary_vector, obstacle_vector

# Define function to find unit normals and effective areas to each boundary/obstacle particle on the interface
def __seismic_sph_interfaces(boundary_vector, obstacle_vector, delta=0.01):
    # find particle locations
    x_slice = np.zeros(len(boundary_vector[0].x) + len(obstacle_vector[0].x))
    y_slice, z_slice = np.zeros_like(x_slice), np.zeros_like(x_slice)
    
    x_slice[:len(boundary_vector[0].x)], x_slice[len(boundary_vector[0].x):] = boundary_vector[0].x, obstacle_vector[0].x
    y_slice[:len(boundary_vector[0].x)], y_slice[len(boundary_vector[0].x):] = boundary_vector[0].y, obstacle_vector[0].y
    z_slice[:len(boundary_vector[0].x)], z_slice[len(boundary_vector[0].x):] = boundary_vector[0].z, obstacle_vector[0].z
    
    # convert locations to indices
    loc_vector = np.column_stack(( ((x_slice + delta/2)//delta).astype(int), ((y_slice + delta/2)//delta).astype(int), ((z_slice + delta/2)//delta).astype(int) ))
    loc_tuple = set(map(tuple, loc_vector))
        
    # determine positions of potential neighbours
    neighbour_offsets_3 = np.array([p for p in product(vals, repeat=3) if p != (0,0,0)], dtype=int) # 3x3x3 cube, missing centre
    neighbour_offsets_5 = np.array([p for p in product(vals, repeat=3) if p != (0,0,0)], dtype=int) # 5x5x5 cube, missing centre
    neighbour_index = np.zeros((len(neighbour_offsets), len(x_slice)), dtype=bool)
    
    # check if each neighbour exists for set of offsets
    for i in range(0, len(neighbour_offsets_3)):
        neighbours = loc_vector + neighbour_offsets_3[i]
        neighbour_index[i,:] = [tuple(n) in loc_tuple for n in neighbours]
        
    # ignore locations comprising the rectangular boundary of the simulation
    x_min, x_max = np.min(x_slice), np.max(x_slice)
    y_min, y_max = np.min(y_slice), np.max(y_slice)
    z_min, z_max = np.min(z_slice), np.max(z_slice)
    internal_index = ( (x_min < x_slice) & (x_slice < x_max) & (y_min < y_slice) & (y_slice < ymax) & (z_min < z_slice) & (z_slice < zmax) )

    # find boundary/obstacle particles on the interface (and internal to the simulation)
    interface_index = np.logical_and(np.logical_not(np.all(neighbour_index, axis=0)), internal_index)
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
def __seismic_sph_interactions(time_vector, iter_vector, boundary_vector, obstacle_vector, interface_index, normals, effective_area, filename=None, gx=0, gy=0, gz=0, dx=0.01, dy=0.01, dz=0.01):
    # define vectors to store collision data
    t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector = [], [], [], [], [], [], []
    
    # find time-averaged pressure and force vectors
    for i in range(1, len(time_vector)):
        # find particle location and mass
        x_slice = np.concatenate([boundary_vector[i].x, obstacle_vector[i].x])[interface_index]
        y_slice = np.concatenate([boundary_vector[i].y, obstacle_vector[i].y])[interface_index]
        z_slice = np.concatenate([boundary_vector[i].z, obstacle_vector[i].z])[interface_index]
        #m_slice = np.concatenate([boundary_vector[i].m, obstacle_vector[i].m])[interface_index]

        # find pressure and force vector of boundary/obstacle particle
        p_slice = np.concatenate([boundary_vector[i].p, obstacle_vector[i].p])[interface_index]
        f_slice = p_slice * normals * effective_area
        
        # define bounds of coarse(r) cartesian grid
        min_x, max_x = int((np.min(x_slice) + dx/2)//dx), int((np.max(x_slice) + dx/2)//dx)
        min_y, max_y = int((np.min(y_slice) + dy/2)//dy), int((np.max(y_slice) + dy/2)//dy)
        min_z, max_z = int((np.min(z_slice) + dz/2)//dz), int((np.max(z_slice) + dz/2)//dz)
        
        # define grid of force vectors
        fx_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
        fy_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))
        fz_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1))

        # add each acceleration vector (as a force) to a cartesian grid
        x_index = ((x_slice + dx/2)//dx - min_x).astype(int)
        y_index = ((y_slice + dy/2)//dy - min_y).astype(int)
        z_index = ((z_slice + dz/2)//dz - min_z).astype(int)
        np.add.at(fx_grid, (x_index, y_index, z_index), f_slice[:, 0])
        np.add.at(fy_grid, (x_index, y_index, z_index), f_slice[:, 1])
        np.add.at(fz_grid, (x_index, y_index, z_index), f_slice[:, 2])
        
        # find unique non-zero locations in grid
        x_unique, y_unique, z_unique = np.unique(np.column_stack((x_index, y_index, z_index)), axis=0).T

        # add grid locations and forces to output lists
        t_vector.extend([time_vector[i]] * len(x_unique))
        x_vector.extend((x_unique + min_x) * dx)
        y_vector.extend((y_unique + min_y) * dy)
        z_vector.extend((z_unique + min_z) * dz)
        fx_vector.extend(fx_grid[x_unique, y_unique, z_unique])
        fy_vector.extend(fy_grid[x_unique, y_unique, z_unique])
        fz_vector.extend(fz_grid[x_unique, y_unique, z_unique])

        # remove current time step and read-in one extra time step
        boundary_vector[i-1] = None
        obstacle_vector[i-1] = None
        boundary_vector, obstacle_vector = __seismic_sph_particles(time_vector, iter_vector, boundary_vector, obstacle_vector, filename=filename, start=i+buff-1, buffer=1)

    return t_vector, x_vector, y_vector, z_vector, fx_vector, fy_vector, fz_vector


## Define functions to find force vectors for particle collisions; all inputs in SI units, z0 is a depth
def seismic_sph_forces(x0, y0, z0, filename=None, dx=0.1, dy=0.1, dz=0.1, dt=1/1000., average_height=False, crit_force=1e-6):
    # read particle interaction datafile
    with h5py.File(filename+'_interactions.hdf5', 'r') as f:
        t_h5 = f['t'][:]
        x_h5 = f['x'][:]
        y_h5 = f['y'][:]
        z_h5 = f['z'][:]
        fx_h5 = f['fx'][:]
        fy_h5 = f['fy'][:]
        fz_h5 = f['fz'][:]

    # calculate the maximum and cut-off force for the interactions
    f_h5 = np.sqrt(fx_h5**2 + fy_h5**2 + fz_h5**2)
    crit_index = f_h5 > crit_force*np.max(f_h5)
    
    # define reduced set of times, locations and forces
    t_vector = t_h5[crit_index]
    x_vector, y_vector, z_vector = x_h5[crit_index], y_h5[crit_index], z_h5[crit_index]
    fx_vector, fy_vector, fz_vector = fx_h5[crit_index], fy_h5[crit_index], fz_h5[crit_index]
    f_vector = f_h5[crit_index]
    
    # add each collision to a cartesian grid to minimise number of calculations
    loc_vector = []
    index_vector = np.zeros_like(t_vector, dtype='int')
    height, weight = 0., 0.
    for i in range(0, len(t_vector)):
        loc = [(x_vector[i] + dx/2)//dx * dx, (y_vector[i] + dy/2)//dy * dy, (z_vector[i] + dz/2)//dz * dz]
        # find index of grid location and add to array if not present
        if loc in loc_vector:
            idx = [z for z in loc_vector].index(loc)
            if average_height == True:
                height = height + z_vector[i]*f_vector[i]
                weight = weight + f_vector[i]
        else:
            loc_vector.append(loc)
            if height == True:
                height = height + z_vector[i]*f_vector[i]
                weight = weight + f_vector[i]
            idx = len(loc_vector) - 1

        # add index of non-zero element to array
        index_vector[i] = idx
    loc_vector = np.array(loc_vector)

    # create pandas dataframe to output data
    forces_df = pd.DataFrame(columns=['t', 'loc_1', 'loc_2', 'loc_3', 'force_1', 'force_2', 'force_3'])
    forces_df['t'] = (np.array(t_vector) + dt/2)//dt * dt
    forces_df['loc_1'] = loc_vector[index_vector, 0] - x0
    forces_df['loc_2'] = loc_vector[index_vector, 1] - y0
    if average_height == True:
        forces_df['loc_3'] = height/weight - z0
    else:
        forces_df['loc_3'] = loc_vector[index_vector, 2] - z0
    forces_df['force_1'] = np.array(fx_vector)
    forces_df['force_2'] = np.array(fy_vector)
    forces_df['force_3'] = np.array(fz_vector)
    
    return forces_df


## Define functions to calculate synthetic seismic signals using pyprop8; all inputs in SI units, z is a depth
def seismic_signals(x, y, z, earth_model=None, forces_df=None, filename=None, duration=10., sampling_rate=200., dt=1/1000., spec=True):
    # read collision datafile if not passed as an input
    if not isinstance(forces_df, pd.DataFrame):
        forces_df = pd.read_csv(filename+'.csv', sep=',', index_col=0)
    time_vector, x_vector, y_vector, z_vector = forces_df['t'].values, forces_df['loc_1'].values, forces_df['loc_2'].values, -forces_df['loc_3'].values
    f1_vector, f2_vector, f3_vector = forces_df['force_1'].values, forces_df['force_2'].values, forces_df['force_3'].values
    
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
        
        for j in range(0, int(1./(sampling_rate*dt))):
            # create source object
            source = pp.PointSource(0., 0., z_vector[0]/1000., M, F, j*dt)

            # create discplacement waveforms (and derivatives) at the location of each seismometer
            times, seismogram, derivative = pp.compute_seismograms(earth_model, source, receivers, nt, 1./sampling_rate, derivatives=pp.DerivativeSwitches(time=True), squeeze_outputs=False, show_progress=False)
            seismograms.append(seismogram)
            derivatives.append(derivative)

            # create velocity spectrum at the location of each seismometer
            if spec:
                spectrum = pp.compute_spectra(earth_model, source, receivers, 2*np.pi*freqs, squeeze_outputs=False, show_progress=False)
                spectra.append(spectrum)

    # find appropriate time bin and shift in samples
    time_bin = (time_vector % (1./(sampling_rate)) / dt).astype(int)
    time_shift = (time_vector // (1./(sampling_rate))).astype(int)

    # create data structures for output seismograms and spectra
    nsamples = len(np.array(seismograms)[0,0,0,0,:])
    displacements = np.zeros((3, nsamples))
    velocities = np.zeros((3, nsamples))
    if spec:
        velocity_spectra = np.zeros((3, len(freqs)), dtype=complex)

    for i in range(0, len(time_vector)):
        # add x, y and z-components of force to seismogram; force is in Newtons
        displacements[:,time_shift[i]:] = displacements[:,time_shift[i]:] \
                        + f1_vector[i]*np.array(seismograms)[time_bin[i], 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]] \
                        + f2_vector[i]*np.array(seismograms)[time_bin[i] + int(1./(sampling_rate*dt)), 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]] \
                        + f3_vector[i]*np.array(seismograms)[time_bin[i] + 2*int(1./(sampling_rate*dt)), 0, \
                                                            index_vector[i], :, 0:nsamples-time_shift[i]]
        velocities[:,time_shift[i]:] = velocities[:,time_shift[i]:] \
                        + f1_vector[i]*np.array(derivatives)[time_bin[i], 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]] \
                        + f2_vector[i]*np.array(derivatives)[time_bin[i] + int(1./(sampling_rate*dt)), 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]] \
                        + f3_vector[i]*np.array(derivatives)[time_bin[i] + 2*int(1./(sampling_rate*dt)), 0, \
                                                            index_vector[i], 0, :, 0:nsamples-time_shift[i]]

        # add x, y and z-components of force to spectrum; force is in Newtons
        if spec:
            velocity_spectra[:,:] = velocity_spectra[:,:] \
                            + f1_vector[i]*np.array(spectra)[time_bin[i], 0, index_vector[i], :, :] \
                            + f2_vector[i]*np.array(spectra)[time_bin[i] + int(1./(sampling_rate*dt)), 0, \
                                                                index_vector[i], :, :] \
                            + f3_vector[i]*np.array(spectra)[time_bin[i] + 2*int(1./(sampling_rate*dt)), 0, \
                                                                index_vector[i], :, :]

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
    
# Define function to specify earth/ice layer model; all inputs in SI units, array of layers each with element: (z_top, v_p, v_s, rho)
def earth_model(layers):
    return pp.LayeredStructureModel(np.array(layers)/1000., interface_depth_form=True)
