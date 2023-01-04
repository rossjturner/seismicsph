# seismicsph_collisions module
# Ross Turner, 1 January 2023

# import packages
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

## Define function to run smoothed particle hydrodynamcial simulation
def run_seismic_sph(x, y, z, t, dx, dt, set_fluid, set_obstacle, output_times=None, filename=None):
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

## Define functions to find particles that collide with identified water--ice interfaces
def seismic_sph_collisions(x, y, z, dx, set_obstacle, filename=None, sigma=sigma):
    # read in time and fluid properties from file
    time_vector, iter_vector = __seismic_sph_times(filename=filename)
    fluid_vector = __seismic_sph_particles(time_vector, iter_vector, [], filename=filename, start=0, buffer=buff)
    
    # derive collision data from SPH outputs
    t_vector, x_vector, y_vector, z_vector, nx_vector, ny_vector, nz_vector, ix_vector, iy_vector, iz_vector, bx_vector, by_vector, bz_vector, rx_vector, ry_vector, rz_vector, m_vector, rho_vector = __seismic_sph_collisions(x, y, z, dx, time_vector, iter_vector, fluid_vector, set_obstacle, filename=filename, sigma=sigma)
    
    # create pandas dataframe to output data
    collision_df = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'nx', 'ny', 'nz', 'ix', 'iy', 'iz', 'bx', 'by', 'bz', 'rx', 'ry', 'rz', 'm', 'rho'])
    collision_df['t'] = t_vector
    collision_df['x'] = x_vector
    collision_df['y'] = y_vector
    collision_df['z'] = z_vector
    collision_df['nx'] = nx_vector
    collision_df['ny'] = ny_vector
    collision_df['nz'] = nz_vector
    collision_df['ix'] = ix_vector
    collision_df['iy'] = iy_vector
    collision_df['iz'] = iz_vector
    collision_df['bx'] = bx_vector
    collision_df['by'] = by_vector
    collision_df['bz'] = bz_vector
    collision_df['rx'] = rx_vector
    collision_df['ry'] = ry_vector
    collision_df['rz'] = rz_vector
    collision_df['m'] = m_vector
    collision_df['rho'] = rho_vector
    
    return collision_df

# Define function to extract output time steps
def __seismic_sph_times(filename=None):
    # read-in log file
    if filename == None:
        df = pd.read_csv('seismic_sph_output/seismic_sph.log', sep='|')
    else:
        df = pd.read_csv(filename+'_output/'+filename+'.log', sep='|')

    time_vector = []
    iter_vector = []

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

    return np.asarray(time_vector).astype(float), np.asarray(iter_vector).astype(float)

# Define function to read-in locations and velocites of particles at each time step
def __seismic_sph_particles(time_vector, iter_vector, fluid_vector, filename=None, start=0, buffer=buff):
    # modify the default parameters of np.load
    np_load = np.load
    np.load = lambda *a,**k: np_load(*a, allow_pickle=True, **k)

    # read-in data file at specified time step
    for i in range(start, min(len(time_vector), start + buffer)):
        if filename == None:
            try:
                data = load('__main___output/__main___'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
            except:
                data = load('__main___output/__main___'+'{:.0f}'.format(iter_vector[i])+'.npz')
        else:
            try:
                data = load(filename+'_output/'+filename+'_'+'{:.0f}'.format(iter_vector[i])+'.hdf5')
            except:
                data = load(filename+'_output/'+filename+'_'+'{:.0f}'.format(iter_vector[i])+'.npz')
        
        # store fluid particle properties in datafile
        fluid_vector.append(data['arrays']['fluid'])
        
    # revert to orginal definition
    np.load = np_load
    
    return fluid_vector

# Define function to find particles that will collide with water--ice interfaces
def __seismic_sph_collisions(x, y, z, dx, time_vector, iter_vector, fluid_vector, set_obstacle, filename=None, sigma=sigma):
    # define vectors to store collision data
    t_vector, x_vector, y_vector, z_vector, nx_vector, ny_vector, nz_vector, ix_vector, iy_vector, iz_vector, bx_vector, by_vector, bz_vector, rx_vector, ry_vector, rz_vector, m_vector, rho_vector = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
   
    # define critical radius of kernel
    kern_radius = sigma*dx*hdx #dx
    
    # find particles that have crossed within kernel radius of ice interface
    for i in range(1, len(time_vector)):
        # find velocity of particle between current and previous timesteps
        vx_vector = (fluid_vector[i].x - fluid_vector[i-1].x)/(time_vector[i] - time_vector[i-1])
        vy_vector = (fluid_vector[i].y - fluid_vector[i-1].y)/(time_vector[i] - time_vector[i-1])
        vz_vector = (fluid_vector[i].z - fluid_vector[i-1].z)/(time_vector[i] - time_vector[i-1])
        v_vector = np.sqrt(vx_vector**2 + vy_vector**2 + vz_vector**2)
                
        # find particles at time t = i predicted to pass into obstacle within kernel radius
        current_indices = set_obstacle(fluid_vector[i].x, fluid_vector[i].y, fluid_vector[i].z, kern_radius) # particles predicted to pass into obstacle
        # find particles at time t = i - 1 predicted NOT to pass into obstacle within kernel radius
        previous_indices = set_obstacle(fluid_vector[i-1].x, fluid_vector[i-1].y, fluid_vector[i-1].z, kern_radius) # particles predicted to pass into obstacle
        # find inidices not in both arrays; i.e. in fluid at t = i - 1 then boundary at t = i xor boundary at t = i - 1 then fluid at t = i
        if len(current_indices) <= 0 or len(previous_indices) <= 0:
            incident_indices = current_indices # if not empty is in the boundary, but previous time step not
        else:
            incident_indices = np.setxor1d(current_indices, previous_indices, assume_unique=True)
        # find indices the are in boundary at t = i; not the fluid at t = i
        incident_indices = np.intersect1d(incident_indices, current_indices, assume_unique=True)
        
        if len(incident_indices) > 0:
            # determine properties of collision for each particle
            for j in range(0, len(incident_indices)):
                k = i + 1
                critical_distance = False
                # define variables to store path length and velocity vector
                path_length = [0]
                path_x, path_y, path_z = [fluid_vector[i].x[incident_indices[j]]], [fluid_vector[i].y[incident_indices[j]]], [fluid_vector[i].z[incident_indices[j]]]
                path_vx, path_vy, path_vz = [vx_vector[incident_indices[j]]], [vy_vector[incident_indices[j]]], [vz_vector[incident_indices[j]]]
                
                # check if particle is within kernel radius of an ice boundary
                try:
                    ind = set_obstacle(fluid_vector[k].x[incident_indices[j]], fluid_vector[k].y[incident_indices[j]], fluid_vector[k].z[incident_indices[j]], kern_radius)
                except:
                    pass
                
                while k - i < buff and k < len(fluid_vector) and not ((isinstance(ind, tuple) and len(ind[0]) == 0) or (not isinstance(ind, tuple) and len(ind) == 0)):
                    # track path of particle
                    path_length.append(np.sqrt((fluid_vector[k].x[incident_indices[j]] - fluid_vector[k-1].x[incident_indices[j]])**2 + (fluid_vector[k].y[incident_indices[j]] - fluid_vector[k-1].y[incident_indices[j]])**2 + (fluid_vector[k].z[incident_indices[j]] - fluid_vector[k-1].z[incident_indices[j]])**2) + np.sum(path_length))
                    path_x.append(fluid_vector[k].x[incident_indices[j]])
                    path_y.append(fluid_vector[k].y[incident_indices[j]])
                    path_z.append(fluid_vector[k].z[incident_indices[j]])
                    path_vx.append((fluid_vector[k].x[incident_indices[j]] - fluid_vector[k-1].x[incident_indices[j]])/(time_vector[k] - time_vector[k-1]))
                    path_vy.append((fluid_vector[k].y[incident_indices[j]] - fluid_vector[k-1].y[incident_indices[j]])/(time_vector[k] - time_vector[k-1]))
                    path_vz.append((fluid_vector[k].z[incident_indices[j]] - fluid_vector[k-1].z[incident_indices[j]])/(time_vector[k] - time_vector[k-1]))

                    # check if particle ever passes within 95% the kernel radius of the collision site; if not it will be ignored
                    ind = set_obstacle(fluid_vector[k].x[incident_indices[j]], fluid_vector[k].y[incident_indices[j]], fluid_vector[k].z[incident_indices[j]], kernel_frac*kern_radius)
                    if not ((isinstance(ind, tuple) and len(ind[0]) == 0) or (not isinstance(ind, tuple) and len(ind) == 0)):
                        # update minimum distance variable
                        critical_distance = True
                    k = k + 1
                    if k - i < buff and k < len(fluid_vector):
                        ind = set_obstacle(fluid_vector[k].x[incident_indices[j]], fluid_vector[k].y[incident_indices[j]], fluid_vector[k].z[incident_indices[j]], kern_radius)
                
                if k - i == buff or k == len(fluid_vector):
                    k = k - 1 # reflected velocity is taken as velocity after 0.03 seconds
                if critical_distance == True:
                    # interpolate to find exact time that particle passes within and out of kernel
                    path_start, path_end = 0, 0
                    for q in range(0, 2):
                        # define method so that a is not within kernel but b is within kernel
                        if q == 0:
                            xa, ya, za = fluid_vector[i-1].x[incident_indices[j]], fluid_vector[i-1].y[incident_indices[j]], fluid_vector[i-1].z[incident_indices[j]]
                            xb, yb, zb = fluid_vector[i].x[incident_indices[j]], fluid_vector[i].y[incident_indices[j]], fluid_vector[i].z[incident_indices[j]]
                        else:
                            xa, ya, za = fluid_vector[k].x[incident_indices[j]], fluid_vector[k].y[incident_indices[j]], fluid_vector[k].z[incident_indices[j]]
                            xb, yb, zb = fluid_vector[k-1].x[incident_indices[j]], fluid_vector[k-1].y[incident_indices[j]], fluid_vector[k-1].z[incident_indices[j]]
                        
                        a, b = 0, 1
                        c = (a + b)/2.
                        # apply bisection method to find location on kernel edge
                        while np.abs(b - a) > tol:
                            # test if location sits either side of kernel edge
                            xc = xa + c*(xb - xa)
                            yc = ya + c*(yb - ya)
                            zc = za + c*(zb - za)
                            ind_c = set_obstacle(xc, yc, zc, kern_radius)
                            
                            if (isinstance(ind_c, tuple) and len(ind_c[0]) == 0) or (not isinstance(ind_c, tuple) and len(ind_c) == 0):
                                a = c
                            else:
                                b = c
                            c = (a + b)/2.
                        if q == 0:
                            path_start = np.sqrt((xb - xc)**2 + (yb - yc)**2 + (zb - zc)**2) # add this to start
                        else:
                            path_end = np.sqrt((xb - xc)**2 + (yb - yc)**2 + (zb - zc)**2) # remove this from end

                    # find reflected velocity of particle between current and previous timesteps
                    rx = (fluid_vector[k].x[incident_indices[j]] - fluid_vector[k-1].x[incident_indices[j]])/(time_vector[k] - time_vector[k-1])
                    ry = (fluid_vector[k].y[incident_indices[j]] - fluid_vector[k-1].y[incident_indices[j]])/(time_vector[k] - time_vector[k-1])
                    rz = (fluid_vector[k].z[incident_indices[j]] - fluid_vector[k-1].z[incident_indices[j]])/(time_vector[k] - time_vector[k-1])
                    
                    # find mid-point time, position and velocity of particle between current and previous timesteps
                    l = 0
                    while l + 1 < len(path_length) and path_length[l] + path_start < (path_length[-1] + path_start + path_end)/2:
                        l = l + 1
                    fraction = ((path_length[-1] + path_start + path_end)/2 - (path_length[l-1] + path_start))/(path_length[l] - path_length[l-1])
                    x = path_x[l-1] + fraction*(path_x[l] - path_x[l-1])
                    y = path_y[l-1] + fraction*(path_y[l] - path_y[l-1])
                    z = path_z[l-1] + fraction*(path_z[l] - path_z[l-1])
                    bx = path_vx[l-1] + fraction*(path_vx[l] - path_vx[l-1])
                    by = path_vy[l-1] + fraction*(path_vy[l] - path_vy[l-1])
                    bz = path_vz[l-1] + fraction*(path_vz[l] - path_vz[l-1])
                    time_collision = time_vector[i+l-1] + fraction*(time_vector[i+l] - time_vector[i+l-1]) # approximately valid and avoids infinities

                    # find coordinates and unit normal of location where each incident particle passes through kernel
                    x_location, y_location, z_location, nx, ny, nz = __get_interfaces(x, y, z, set_obstacle, kern_radius)
            
                    # add collision to vectors
                    if x_location == -99 or np.isinf(bx) or np.isnan(bx) or np.isinf(rx) or np.isnan(rx):
                        pass
                    else:
                        t_vector.append(time_collision)
                        x_vector.append(x_location)
                        y_vector.append(y_location)
                        z_vector.append(z_location)
                        nx_vector.append(nx)
                        ny_vector.append(ny)
                        nz_vector.append(nz)
                        ix_vector.append(vx_vector[incident_indices[j]])
                        iy_vector.append(vy_vector[incident_indices[j]])
                        iz_vector.append(vz_vector[incident_indices[j]])
                        bx_vector.append(bx)
                        by_vector.append(by)
                        bz_vector.append(bz)
                        rx_vector.append(rx)
                        ry_vector.append(ry)
                        rz_vector.append(rz)
                        m_vector.append(fluid_vector[i].m[incident_indices[j]])
                        rho_vector.append(fluid_vector[i].rho[incident_indices[j]])

        # remove current time step and read-in one extra time step
        fluid_vector[i-1] = None
        fluid_vector = __seismic_sph_particles(time_vector, iter_vector, fluid_vector, filename=filename, start=i+buff-1, buffer=1)

    return t_vector, x_vector, y_vector, z_vector, nx_vector, ny_vector, nz_vector, ix_vector, iy_vector, iz_vector, bx_vector, by_vector, bz_vector, rx_vector, ry_vector, rz_vector, m_vector, rho_vector

# Define function to find normal vector to water--ice interface
def __get_interfaces(x, y, z, set_obstacle, max_radius):
    
    # define initial values for bisection method (main fit)
    a = 0 # must not be in boundary
    b = max_radius # must be in boundary
    c = (a + b)/2.
    
    # check a is not in boundary and b is in boundary
    ind_a = set_obstacle(x, y, z, a)
    ind_b = set_obstacle(x, y, z, b)
    if not ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0)) or ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
        print(color_text('Particle at x='+'{:.2f}'.format(x)+', y='+'{:.2f}'.format(y)+', z='+'{:.2f}'.format(z)+' not located correctly with respect to fluid and boundary.', Colors.Orange))
        return -99, -99, -99, -99, -99, -99
    else:
        # apply bisection method to find impact site
        while np.abs(b - a) > tol*max_radius:
            # test if radius sits either side of boundary or not
            ind_c = set_obstacle(x, y, z, c)
            if (isinstance(ind_c, tuple) and len(ind_c[0]) == 0) or (not isinstance(ind_c, tuple) and len(ind_c) == 0):
                a = c
            else:
                b = c
            c = (a + b)/2.
        radius = c
        
        # find two other locations the same distance from the boundary
        x1, y1, z1 = -99, -99, -99
        x2, y2, z2 = -99, -99, -99
        # x perturbations
        x1, y1, z1 = __get_intersection([x - pert*max_radius, x + 2*pert*max_radius], y - pert*max_radius, z - pert*max_radius, set_obstacle, radius, max_radius)
        if x1 == -99:
            x1, y1, z1 = __get_intersection([x + pert*max_radius, x - 2*pert*max_radius], y + pert*max_radius, z + 2*pert*max_radius, set_obstacle, radius, max_radius)
        else:
            x2, y2, z2 = __get_intersection([x + pert*max_radius, x - 2*pert*max_radius], y + pert*max_radius, z + 2*pert*max_radius, set_obstacle, radius, max_radius)
        # y perturbations
        if x1 == -99:
            x1, y1, z1 = __get_intersection(x - pert*max_radius, [y - pert*max_radius, y + 2*pert*max_radius], z - pert*max_radius, set_obstacle, radius, max_radius)
        elif x2 == -99:
            x2, y2, z2 = __get_intersection(x - pert*max_radius, [y - pert*max_radius, y + 2*pert*max_radius], z - pert*max_radius, set_obstacle, radius, max_radius)
        if x1 == -99:
            x1, y1, z1 = __get_intersection(x + pert*max_radius, [y + pert*max_radius, y - 2*pert*max_radius], z + 2*pert*max_radius, set_obstacle, radius, max_radius)
        elif x2 == -99:
            x2, y2, z2 = __get_intersection(x + pert*max_radius, [y + pert*max_radius, y - 2*pert*max_radius], z + 2*pert*max_radius, set_obstacle, radius, max_radius)
        # z perturbations
        if x1 == -99:
            x1, y1, z1 = __get_intersection(x - pert*max_radius, y - pert*max_radius, [z - pert*max_radius, z + 2*pert*max_radius], set_obstacle, radius, max_radius)
        elif x2 == -99:
            x2, y2, z2 = __get_intersection(x - pert*max_radius, y - pert*max_radius, [z - pert*max_radius, z + 2*pert*max_radius], set_obstacle, radius, max_radius)
        if x2 == -99:
            x2, y2, z2 = __get_intersection(x + pert*max_radius, y + 2*pert*max_radius, [z + pert*max_radius, z - 2*pert*max_radius], set_obstacle, radius, max_radius)

        # find normal to interface
        if not (x1 == -99 or x2 == -99):
            nx = (y1 - y)*(z2 - z) - (z1 - z)*(y2 - y)
            ny = (z1 - z)*(x2 - x) - (x1 - x)*(z2 - z)
            nz = (x1 - x)*(y2 - y) - (y1 - y)*(x2 - x)
            # normalise vector
            n = np.sqrt(nx**2 + ny**2 + nz**2)
            nx, ny, nz = nx/n, ny/n, nz/n
            
            # check normal vector points in direction of fluid
            x_norm = x + nx*pert
            y_norm = y + ny*pert
            z_norm = z + nz*pert
            ind = set_obstacle(x_norm, y_norm, z_norm, radius)
            if (isinstance(ind, tuple) and len(ind[0]) == 0) or (not isinstance(ind, tuple) and len(ind) == 0):
                nx = -nx
                ny = -ny
                nz = -nz
                
            # calculate location of collision on boundary
            x_int = x + nx*radius
            y_int = y + ny*radius
            z_int = z + nz*radius
            
            return x_int, y_int, z_int, nx, ny, nz
        else:
            print(color_text('Particle at x='+'{:.2f}'.format(x)+', y='+'{:.2f}'.format(y)+', z='+'{:.2f}'.format(z)+' excluded as boundary perturbations not defined.', Colors.Orange))
            return -99, -99, -99, -99, -99, -99
    

def __get_intersection(x, y, z, set_obstacle, radius, max_radius):

    # apply algorithm on direction with vector min/max values
    if isinstance(x, (tuple, list, np.ndarray)):
        ind_a = set_obstacle(x[0], y, z, radius)
        ind_b = set_obstacle(x[1], y, z, radius)
        if ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0)) and not ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = x[0]
            b = x[1]
        elif (not ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0))) and ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = x[1]
            b = x[0]
        else:
            return -99, -99, -99
        
        c = (a + b)/2.
        while np.abs(b - a) > tol*max_radius:
            # test if radius sits either side of boundary or not
            ind_c = set_obstacle(c, y, z, radius)
            if (isinstance(ind_c, tuple) and len(ind_c[0]) == 0) or (not isinstance(ind_c, tuple) and len(ind_c) == 0):
                a = c
            else:
                b = c
            c = (a + b)/2.
            
        return c, y, z
    elif isinstance(y, (tuple, list, np.ndarray)):
        ind_a = set_obstacle(x, y[0], z, radius)
        ind_b = set_obstacle(x, y[1], z, radius)
        if ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0)) and not ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = y[0]
            b = y[1]
        elif (not ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0))) and ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = y[1]
            b = y[0]
        else:
            return -99, -99, -99
        
        c = (a + b)/2.
        while np.abs(b - a) > tol*max_radius:
            # test if radius sits either side of boundary or not
            ind_c = set_obstacle(x, c, z, radius)
            if (isinstance(ind_c, tuple) and len(ind_c[0]) == 0) or (not isinstance(ind_c, tuple) and len(ind_c) == 0):
                a = c
            else:
                b = c
            c = (a + b)/2.
            
        return x, c, z
    else:
        ind_a = set_obstacle(x, y, z[0], radius)
        ind_b = set_obstacle(x, y, z[1], radius)
        if ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0)) and not ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = z[0]
            b = z[1]
        elif (not ((isinstance(ind_a, tuple) and len(ind_a[0]) == 0) or (not isinstance(ind_a, tuple) and len(ind_a) == 0))) and ((isinstance(ind_b, tuple) and len(ind_b[0]) == 0) or (not isinstance(ind_b, tuple) and len(ind_b) == 0)):
            a = z[1]
            b = z[0]
        else:
            return -99, -99, -99
        
        c = (a + b)/2.
        while np.abs(b - a) > tol*max_radius:
            # test if radius sits either side of boundary or not
            ind_c = set_obstacle(x, y, c, radius)
            if (isinstance(ind_c, tuple) and len(ind_c[0]) == 0) or (not isinstance(ind_c, tuple) and len(ind_c) == 0):
                a = c
            else:
                b = c
            c = (a + b)/2.
            
        return x, y, c
