# seismicsph_propagation module
# Ross Turner, 1 January 2023

# Functions expect a pandas dataframe as an input with at least the following columns: t (time of impact), x, y, z (cartesian coordinates of impact), ix, iy, iz (cartesian components of incident particle velocity), rx, ry, rz (cartesian components of reflected particle velocity), m, rho (mass and density of fluid particle). The normal to the surface (directed inwards to the ice) can optionally be specified using nx, ny, nz.

# import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
import time

### Main function to run code
def icewater_collisions(x_seis, y_seis, z_seis, collision_df, endtime, starttime=0, sample_rate=1./200, sound_speed=1400., temperature=-10, sp_amplitude_ratio=0.5, amplitude_type='acceleration'):

    # create a copy of the input pandas dataframe
    if collision_df.columns[0] == 't':
        collision_data = collision_df.values
    else:
        collision_data = collision_df.values[:,1:]
    collision_dict = {'t':collision_data[:,0], 'x':collision_data[:,1], 'y':collision_data[:,2], 'z':collision_data[:,3], 'nx':collision_data[:,4], 'ny':collision_data[:,5], 'nz':collision_data[:,6], 'ix':collision_data[:,7] - collision_data[:,10], 'iy':collision_data[:,8] - collision_data[:,11], 'iz':collision_data[:,9] - collision_data[:,12], 'rx':collision_data[:,13] - collision_data[:,10], 'ry':collision_data[:,14] - collision_data[:,11], 'rz':collision_data[:,15] - collision_data[:,12], 'm':collision_data[:,16], 'rho':collision_data[:,17]}

    # derive constants describing the damped oscillation at the ice-water interface
    if not (amplitude_type == 'energy' or amplitude_type == 'fourier'):
        collision_dict = collision_constants(collision_dict, sample_rate=sample_rate, noise=False)
    else:
        collision_dict = collision_constants(collision_dict, sample_rate=sample_rate, noise=True)
    
    # derive wavenumber and its derivative for the longitudinal seismic wave
    collision_dict = wavenumber_constants(collision_dict, temperature)
    # derive attenuation constant and its derivative for the longitudinal seismic wave
    collision_dict = attenuation_constants(collision_dict, temperature)
    
    if not (amplitude_type == 'energy' or amplitude_type == 'fourier'):
        # define factor to over sample acceleration
        if sample_rate > 1./200:
            sample_factor = int(200*sample_rate*18) + 1 # ensure sample rate is never too small
        else:
            sample_factor = 18
        
        # create list of times at the seismometer
        times = np.arange(starttime, endtime + sample_rate*(1 + 0.5/float(sample_factor)), sample_rate/float(sample_factor))
        # create pandas dataframe to store simulated seismic signal at the seismometer
        waveform_df = pd.DataFrame(columns=['t', 'etax', 'etay', 'etaz', 'px', 'py', 'pz', 'sx', 'sy', 'sz'])
        
        k = 0
        for i in range(0, len(times)):
            if i%sample_factor == 0:
                if i > 0:
                    # append time-averaged acceleration amplitudes to date structure
                    waveform_df.loc[k] = list([times[i - sample_factor], np.sum(px_vector)/sample_factor + np.sum(sx_vector)/sample_factor, np.sum(py_vector)/sample_factor + np.sum(sy_vector)/sample_factor, np.sum(pz_vector)/sample_factor + np.sum(sz_vector)/sample_factor, np.sum(px_vector)/sample_factor, np.sum(py_vector)/sample_factor, np.sum(pz_vector)/sample_factor, np.sum(sx_vector)/sample_factor, np.sum(sy_vector)/sample_factor, np.sum(sz_vector)/sample_factor])
                    k = k + 1
                
                # define empty vectors to add over sampled acceleration amplitudes
                px_vector, py_vector, pz_vector, sx_vector, sy_vector, sz_vector = [], [], [], [], [], []
            
            if i < len(times):
                if times[i] < np.min(collision_df['t']):
                    # set amplitudes to zero if no impacts have occurred
                    px_vector.append(0.)
                    py_vector.append(0.)
                    pz_vector.append(0.)
                    sx_vector.append(0.)
                    sy_vector.append(0.)
                    sz_vector.append(0.)
                else:
                    # derive amplitude of seismic wave at time t for s- and p-waves
                    collision_dict = amplitude(x_seis, y_seis, z_seis, times[i], collision_dict, temperature, amplitude_type, wave_type='p')
                    collision_dict = amplitude(x_seis, y_seis, z_seis, times[i], collision_dict, temperature, amplitude_type, wave_type='s')
                    # derive component of seismic wave normal and parallel to the direction of propagation
                    collision_dict = sp_seismic_separation(x_seis, y_seis, z_seis, collision_dict, sp_amplitude_ratio)
                    
                    # add amplitudes to data structure
                    px_vector.append(np.sum(collision_dict['px']))
                    py_vector.append(np.sum(collision_dict['py']))
                    pz_vector.append(np.sum(collision_dict['pz']))
                    sx_vector.append(np.sum(collision_dict['sx']))
                    sy_vector.append(np.sum(collision_dict['sy']))
                    sz_vector.append(np.sum(collision_dict['sz']))
        
        # derive velocity or discplacement amplitude from acceleration amplitude if required
        #if amplitude_type == 'velocity':
        #    waveform_df['etax'] = waveform_df['etax']*sample_rate
        #    waveform_df['etay'] = waveform_df['etay']*sample_rate
        #    waveform_df['etaz'] = waveform_df['etaz']*sample_rate
        #    waveform_df['px'] = waveform_df['px']*sample_rate
        #    waveform_df['py'] = waveform_df['py']*sample_rate
        #    waveform_df['pz'] = waveform_df['pz']*sample_rate
        #    waveform_df['sx'] = waveform_df['sx']*sample_rate
        #    waveform_df['sy'] = waveform_df['sy']*sample_rate
        #    waveform_df['sz'] = waveform_df['sz']*sample_rate
        #elif not amplitude_type == 'acceleration':
        #    waveform_df['etax'] = waveform_df['etax']*sample_rate**2
        #    waveform_df['etay'] = waveform_df['etay']*sample_rate**2
        #    waveform_df['etaz'] = waveform_df['etaz']*sample_rate**2
        #    waveform_df['px'] = waveform_df['px']*sample_rate**2
        #    waveform_df['py'] = waveform_df['py']*sample_rate**2
        #    waveform_df['pz'] = waveform_df['pz']*sample_rate**2
        #    waveform_df['sx'] = waveform_df['sx']*sample_rate**2
        #    waveform_df['sy'] = waveform_df['sy']*sample_rate**2
        #    waveform_df['sz'] = waveform_df['sz']*sample_rate**2

        return waveform_df
    else:
        # create list of frequencies at the seismometer
        frequencies = 10**np.arange(np.log10(0.01), np.log10(1/sample_rate) + 0.005, 0.01)

        # create pandas dataframe to store simulated seismic signal at the seismometer
        spectrum_df = pd.DataFrame(columns=['f', 'foux', 'fouy', 'fouz', 'px', 'py', 'pz', 'sx', 'sy', 'sz'], index=range(len(frequencies)))
        
        k = 0
        for f_seis in frequencies:
            # derive amplitude of seismic wave at frequency f for s- and p-waves
            collision_dict = fourier(x_seis, y_seis, z_seis, f_seis, collision_dict, temperature, amplitude_type, wave_type='p')
            collision_dict = fourier(x_seis, y_seis, z_seis, f_seis, collision_dict, temperature, amplitude_type, wave_type='s')
            # derive component of seismic wave normal and parallel to the direction of propagation
            collision_dict = sp_seismic_separation(x_seis, y_seis, z_seis, collision_dict, sp_amplitude_ratio)
            
            # add amplitudes to pandas dataframe
            spectrum_df.loc[k] = list([f_seis, np.sum(collision_dict['px']**2) + np.sum(collision_dict['sx']**2), np.sum(collision_dict['py']**2) + np.sum(collision_dict['sy']**2), np.sum(collision_dict['pz']**2) + np.sum(collision_dict['sz']**2), np.sum(collision_dict['px']**2), np.sum(collision_dict['py']**2), np.sum(collision_dict['pz']**2), np.sum(collision_dict['sx']**2), np.sum(collision_dict['sy']**2), np.sum(collision_dict['sz']**2)])
            k = k + 1
    
        return spectrum_df

### Functions to define constants for collision at ice-water interface
## Define function to calculate the constants describing the damped collision at ice-water interface.
def collision_constants(collision_dict, sample_rate, noise=False):

    # properties of water
    gamma = 7.
    ref_density = 1000.
    
    # calculate the amplitude of the incident and reflected velocities (across all components)
    collision_dict['i'] = np.sqrt(collision_dict['ix']**2 + collision_dict['iy']**2 + collision_dict['iz']**2)
    collision_dict['r'] = np.sqrt(collision_dict['rx']**2 + collision_dict['ry']**2 + collision_dict['rz']**2)

    # calculate the damping ratio at each collision
    collision_dict['zeta'] = 1./np.sqrt(np.pi**2/(np.log(np.abs(collision_dict['i'])/np.abs(collision_dict['r'])))**2 + 1)

    # calculate the duration of the collision
    if noise == True:
        #collision_dict['tau'] = (collision_dict['m']/collision_dict['rho'])**(1./3)/(0.2*sound_speed*(collision_dict['rho']/ref_density)**((gamma - 1)/2.))
        collision_dict['tau'] = sample_rate/2.
    else:
        collision_dict['tau'] = sample_rate/2.
    
    # calculate the damping constant at each collision
    collision_dict['alpha'] = np.pi*collision_dict['zeta']/(collision_dict['tau']*np.sqrt(1 - collision_dict['zeta']**2))

    # calculate the critical frequency at each collision
    collision_dict['fc'] = 0.01 # minimum frequency of most seismometers
    #collision_dict['fc'] = np.sqrt(np.pi**2 - collision_dict['alpha']*collision_dict['tau']**2)/(2*np.pi*collision_dict['tau'])

    return collision_dict

## Define function to calculate the wavenumber and its derivative at the critical frequency.
def wavenumber_constants(collision_dict, temperature=-10):
    
    # calculate density of ice at the specified temperature (in degrees Celcius)
    density = ice_density(temperature)
    
    # calculate the temperature correction for the elastic modulus at the specified temperature (in degrees Celcius)
    a_constant = 0.0516 #1.418e-3 # Dantl (1969) temperature correction
    temperature_correction = (1 - a_constant*temperature)/(1 - a_constant*(-10))
    
    # calculate the wavenumber (for p-wave) at the critical frequency at each collision
    collision_dict['k'] = 2*np.pi*collision_dict['fc']/np.sqrt(1.482*temperature_correction*elastic_modulus(collision_dict['fc'])/density)
    
    # calculate the derivative of the wavenumber at the critical frequency at each collision
    collision_dict['dk'] = (2*np.pi - np.pi*collision_dict['fc']/elastic_modulus(collision_dict['fc'])*elastic_derivative(collision_dict['fc']))/np.sqrt(1.482*temperature_correction*elastic_modulus(collision_dict['fc'])/density)

    return collision_dict

def ice_density(temperature):
    return 1./(1./916.8*(1 + 1.576e-4*temperature - 2.778e-7*temperature**2 + 8.850e-9*temperature**3 - 1.778e-10*temperature**4))

def elastic_modulus(frequency):
    #return (6.15588553 + 0.999234995*np.log10(frequency) - 0.0256639989*np.log10(frequency)**2 - 0.00590345173*np.log10(frequency)**3)*1e9
    return np.minimum(6.041 + 0.966*np.log10(frequency), 9.254)*1e9

def elastic_derivative(frequency):
    #return (0.433962244556 - 0.022291466225*np.log10(frequency) - 0.007691509534*np.log10(frequency)**2)*1e9/frequency
    return 0.419*1e9/frequency # approximation as tends to zero for large f

## Define function to calculate the attenuation constant and its derivative at the critical frequency.
def attenuation_constants(collision_dict, temperature=-10):

    # calculate the attenuation constant (for p-wave) at the critical frequency at each collision
    collision_dict['beta'] = (0.15*collision_dict['fc'] + 3)*1e-5

    # calcualte the derivative of the attenuation constant at the critical frequency at each collision
    collision_dict['dbeta'] = 0.15*1e-5
        
    return collision_dict

### Functions to calculate initial displacement and dispersion relation
## Define function to calculate the initial displacement at the impact site.
def amplitude(x_seis, y_seis, z_seis, t_seis, collision_dict, temperature=-10, amplitude_type='displacement', wave_type='p'):
    
    # calculate the unit displacement vector from the impact site to the seismometer
    x_vector = x_seis - collision_dict['x']
    y_vector = y_seis - collision_dict['y']
    z_vector = z_seis - collision_dict['z']
    displacement_vector = np.sqrt(x_vector**2 + y_vector**2 + z_vector**2)
    
    # set corrections for transverse seismic waves (if applicable)
    if wave_type == 'p':
        sp_kcorrection = 1
        sp_betacorrection = 1
    else:
        sp_kcorrection = np.sqrt(1.482/0.376)
        sp_betacorrection = 1.482/0.376
    
    # calculate the complex-valued time from the Taylor series expansion of the dispersion relationship (pre-multiplied by the displacement)
    time_vector = np.array(t_seis - (collision_dict['dk']*sp_kcorrection)*displacement_vector/(2*np.pi), dtype=complex)
    time_vector.imag = (collision_dict['dbeta']*sp_betacorrection)*displacement_vector/(2*np.pi)
    
    # calculate the complex-valued propagation constant at the seismometer (pre-multiplied by the displacement)
    gamma_vector = np.array((collision_dict['beta'] - collision_dict['dbeta']*collision_dict['fc'])*sp_betacorrection*displacement_vector, dtype=complex)
    gamma_vector.imag = (collision_dict['k'] - collision_dict['dk']*collision_dict['fc'])*sp_kcorrection*displacement_vector

    # calculate the initial wave amplitude at the impact site as a function of complex-valued time from the dispersion relationship
    if amplitude_type == 'acceleration':
        amplitude_vector = (initial_acceleration(time_vector, collision_dict['t'], collision_dict['alpha'], collision_dict['zeta'], collision_dict['i'], collision_dict['rho'], collision_dict['tau'], temperature)*amplitude_correction(gamma_vector, displacement_vector, collision_dict['m']/collision_dict['rho'])).real
    elif amplitude_type == 'velocity':
        amplitude_vector = (initial_velocity(time_vector, collision_dict['t'], collision_dict['alpha'], collision_dict['zeta'], collision_dict['i'], collision_dict['rho'], collision_dict['tau'], temperature)*amplitude_correction(gamma_vector, displacement_vector, collision_dict['m']/collision_dict['rho'])).real
    else:
        amplitude_vector = (initial_displacement(time_vector, collision_dict['t'], collision_dict['alpha'], collision_dict['zeta'], collision_dict['i'], collision_dict['rho'], collision_dict['tau'], temperature)*amplitude_correction(gamma_vector, displacement_vector, collision_dict['m']/collision_dict['rho'])).real
    
    # calculate direction of perturbation
    perturb_x = collision_dict['rx'] - collision_dict['ix']
    perturb_y = collision_dict['ry'] - collision_dict['iy']
    perturb_z = collision_dict['rz'] - collision_dict['iz']
    perturb_vector = np.sqrt(perturb_x**2 + perturb_y**2 + perturb_z**2)
    perturb_x, perturb_y, perturb_z = perturb_x/perturb_vector, perturb_y/perturb_vector, perturb_z/perturb_vector
    
    # calculate the seismic wave amplitude at the seismometer as a function of local time at the seismometer
    if wave_type == 'p':
        collision_dict['px'] = amplitude_vector*perturb_x
        collision_dict['py'] = amplitude_vector*perturb_y
        collision_dict['pz'] = amplitude_vector*perturb_z
    else:
        collision_dict['sx'] = amplitude_vector*perturb_x
        collision_dict['sy'] = amplitude_vector*perturb_y
        collision_dict['sz'] = amplitude_vector*perturb_z
    
    return collision_dict
    
def initial_displacement(time, incident, alpha, zeta, velocity, fluid_density, tau, temperature=-10):

    # calculate density of ice at the specified temperature (in degrees Celcius)
    #density = ice_density(temperature)

    # calculate the initial acceleration at the impact site
    displacement_vector = np.array(0.*zeta, dtype=complex)
    displacement_vector.imag = -tau/np.pi
    displacement_vector = displacement_vector*(velocity*np.exp(-alpha*(time - incident))*np.exp(1j*np.pi/tau*(time - incident)))
    # multiply by 1000/916 afterwards, or whatever the density ratio is

    # remove times before the collision
    displacement_vector[np.logical_or(time.real - incident < 0, incident < 0)] = 0

    return displacement_vector
    
def initial_velocity(time, incident, alpha, zeta, velocity, fluid_density, tau, temperature=-10):

    # calculate density of ice at the specified temperature (in degrees Celcius)
    #density = ice_density(temperature)

    # calculate the initial acceleration at the impact site
    velocity_vector = np.array(1. + 0.*zeta, dtype=complex)
    velocity_vector.imag = zeta/np.sqrt(1 - zeta**2)
    velocity_vector = velocity_vector*(velocity*np.exp(-alpha*(time - incident))*np.exp(1j*np.pi/tau*(time - incident)))
    # multiply by 1000/916 afterwards, or whatever the density ratio is

    # remove times before the collision
    velocity_vector[np.logical_or(time.real - incident < 0, incident < 0)] = 0

    return velocity_vector

def initial_acceleration(time, incident, alpha, zeta, velocity, fluid_density, tau, temperature=-10):

    # calculate density of ice at the specified temperature (in degrees Celcius)
    #density = ice_density(temperature)

    # calculate the initial acceleration at the impact site
    acceleration_vector = np.array(-2*alpha, dtype=complex)
    acceleration_vector.imag = np.pi/tau - alpha**2*tau/np.pi
    acceleration_vector = acceleration_vector*(velocity*np.exp(-alpha*(time - incident))*np.exp(1j*np.pi/tau*(time - incident)))
    # multiply by 1000/916 afterwards, or whatever the density ratio is

    # remove times before the collision
    acceleration_vector[np.logical_or(time.real - incident < 0, incident < 0)] = 0

    return acceleration_vector

def amplitude_correction(gamma, displacement, volume):
    #return np.asarray(volume**(2./3)/(2*np.pi*displacement**2)*np.exp(-gamma*displacement)).astype(complex).real
    return volume**(2./3)/(2*np.pi*displacement**2)*np.exp(-gamma)

### Functions to separate wave into transverse and longitudinal components
## Define function to separate the transverse and longitudinal wave amplitudes into components normal and parallel to the direction of propagation
def sp_seismic_separation(x_seis, y_seis, z_seis, collision_dict, sp_amplitude_ratio=0.5):
    
    # calculate the displacement vector from the impact site to the seismometer
    x_vector = x_seis - collision_dict['x']
    y_vector = y_seis - collision_dict['y']
    z_vector = z_seis - collision_dict['z']
    displacement_vector = np.sqrt(x_vector**2 + y_vector**2 + z_vector**2)
    x_vector, y_vector, z_vector = x_vector/displacement_vector, y_vector/displacement_vector, z_vector/displacement_vector

    # calculate the amplitude resulting from the p-waves at the seismometer
    projection_vector = collision_dict['px']*x_vector + collision_dict['py']*y_vector + collision_dict['pz']*z_vector
    collision_dict['px'] = projection_vector*x_vector*(1./(sp_amplitude_ratio + 1))
    collision_dict['py'] = projection_vector*y_vector*(1./(sp_amplitude_ratio + 1))
    collision_dict['pz'] = projection_vector*z_vector*(1./(sp_amplitude_ratio + 1))

    # calculate the amplitude resulting from the s-waves at the seismometer
    projection_vector = collision_dict['sx']*x_vector + collision_dict['sy']*y_vector + collision_dict['sz']*z_vector
    collision_dict['sx'] = (collision_dict['sx'] - projection_vector*x_vector)*(sp_amplitude_ratio/(sp_amplitude_ratio + 1))
    collision_dict['sy'] = (collision_dict['sy'] - projection_vector*y_vector)*(sp_amplitude_ratio/(sp_amplitude_ratio + 1))
    collision_dict['sz'] = (collision_dict['sz'] - projection_vector*z_vector)*(sp_amplitude_ratio/(sp_amplitude_ratio + 1))
    
    # check that line-of-sight doesn't pass through water (if pandas dataframe has appropriately named columns)
    try:
        # calculate projection of displacement vector along normal to surface (pointing inwards into ice)
        projection_vector = collision_dict['nx']*x_vector + collision_dict['ny']*y_vector + collision_dict['nz']*z_vector
        idx = projection_vector < 0
        # set amplitude to zero if line-of-sight passes through air/water
        collision_dict['px'][idx] = 0.
        collision_dict['py'][idx] = 0.
        collision_dict['pz'][idx] = 0.
        collision_dict['sx'][idx] = 0.
        collision_dict['sy'][idx] = 0.
        collision_dict['sz'][idx] = 0.
    except:
        # return all collisions if normal to surface is not specified
        pass
    
    return collision_dict

### Functions to calculate initial fourier transform and dispersion relation
## Define function to calculate the initial fourier transform at the impact site.
def fourier(x_seis, y_seis, z_seis, f_seis, collision_dict, temperature=-10, amplitude_type='displacement', wave_type='p'):
    
    # calculate the unit displacement vector from the impact site to the seismometer
    x_vector = x_seis - collision_dict['x']
    y_vector = y_seis - collision_dict['y']
    z_vector = z_seis - collision_dict['z']
    displacement_vector = np.sqrt(x_vector**2 + y_vector**2 + z_vector**2)
    
    # set corrections for transverse seismic waves (if applicable)
    if wave_type == 'p':
        sp_kcorrection = 1
        sp_betacorrection = 1
    else:
        sp_kcorrection = np.sqrt(1.482/0.376)
        sp_betacorrection = 1.482/0.376
    
    # calculate the complex-valued propagation constant at the seismometer (pre-multiplied by the displacement)
    gamma_vector = np.array((collision_dict['beta'] + collision_dict['dbeta']*(f_seis - collision_dict['fc']))*sp_betacorrection*displacement_vector, dtype=complex)
    gamma_vector.imag = (collision_dict['k'] + collision_dict['dk']*(f_seis - collision_dict['fc']))*sp_kcorrection*displacement_vector

    # calculate the initial wave amplitude at the impact site as a function of complex-valued time from the dispersion relationship
    amplitude_vector = (initial_fourier(f_seis, collision_dict['t'], collision_dict['alpha'], collision_dict['zeta'], collision_dict['i'], collision_dict['rho'], collision_dict['tau'], temperature)*fourier_correction(gamma_vector, displacement_vector, collision_dict['m']/collision_dict['rho'])).real
    
    # calculate direction of perturbation
    perturb_x = collision_dict['rx'] - collision_dict['ix']
    perturb_y = collision_dict['ry'] - collision_dict['iy']
    perturb_z = collision_dict['rz'] - collision_dict['iz']
    perturb_vector = np.sqrt(perturb_x**2 + perturb_y**2 + perturb_z**2)
    perturb_x, perturb_y, perturb_z = perturb_x/perturb_vector, perturb_y/perturb_vector, perturb_z/perturb_vector
    
    # calculate the seismic wave amplitude at the seismometer as a function of local time at the seismometer
    if wave_type == 'p':
        collision_dict['px'] = amplitude_vector*perturb_x
        collision_dict['py'] = amplitude_vector*perturb_y
        collision_dict['pz'] = amplitude_vector*perturb_z
    else:
        collision_dict['sx'] = amplitude_vector*perturb_x
        collision_dict['sy'] = amplitude_vector*perturb_y
        collision_dict['sz'] = amplitude_vector*perturb_z
    
    return collision_dict
    
def initial_fourier(frequency, incident, alpha, zeta, velocity, fluid_density, tau, temperature=-10):

    # calculate density of ice at the specified temperature (in degrees Celcius)
    density = ice_density(temperature)
    
    # calculate the initial fourier transform at the impact site
    #fourier_vector = np.asarray(np.pi*velocity/((1 - zeta**2))*np.exp(-omega*incident)*((np.pi*(1 - 2*zeta**2) + 2*tau*zeta*np.sqrt(1 - zeta**2)*(alpha + omega))/(np.pi**2 + tau**2*(alpha + omega)**2))).astype(complex)
    fourier_vector = -np.asarray(velocity/(1 - zeta**2)*((2*zeta*np.sqrt(1 - zeta**2) + 1j*(2*zeta**2 - 1))/(2*np.pi*frequency/(np.pi/tau) + zeta/np.sqrt(1 - zeta**2) - 1j))).astype(complex)
    # multiply by 1000/916 afterwards, or whatever the density ratio is
    
    return fourier_vector

def fourier_correction(gamma, displacement, volume):
    #return np.asarray(volume**(2./3)/(2*np.pi*displacement**2)*np.exp(-gamma*displacement)).astype(complex).real
    return volume**(2./3)/(2*np.pi*displacement**2)*np.exp(-gamma)

## Functions to plot output waveforms
def icewater_waveforms(waveform_df, amplitude_type='acceleration'):
    # p- and s-waves
    # create plot of seismic signal
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 6))
    fig.subplots_adjust(hspace=0)

    rc('text', usetex=True)
    rc('font', size=13)
    rc('legend', fontsize=12.5)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    axs[0].plot(waveform_df['t'], waveform_df['etax'], 'b')
    axs[1].plot(waveform_df['t'], waveform_df['etay'], 'r')
    axs[2].plot(waveform_df['t'], waveform_df['etaz'], 'k')

    axs[0].set_xlim([0, np.max(waveform_df['t'])])
    #ax.set_ylim([0, 2])
    axs[2].set_xlabel(r'Time (s)')
    if amplitude_type == 'acceleration':
        axs[1].set_ylabel(r'Wave acceleration ($\rm m\,s^{-2}$)')
    elif amplitude_type == 'velocity':
        axs[1].set_ylabel(r'Wave velocity ($\rm m\,s^{-1}$)')
    else:
        axs[1].set_ylabel(r'Wave amplitude ($\rm m$)')

    plt.show()

    # p-wave
    # create plot of seismic signal
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 6))
    fig.subplots_adjust(hspace=0)

    rc('text', usetex=True)
    rc('font', size=13)
    rc('legend', fontsize=12.5)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    axs[0].plot(waveform_df['t'], waveform_df['px'], 'b')
    axs[1].plot(waveform_df['t'], waveform_df['py'], 'r')
    axs[2].plot(waveform_df['t'], waveform_df['pz'], 'k')

    axs[0].set_xlim([0, np.max(waveform_df['t'])])
    #ax.set_ylim([0, 2])
    axs[2].set_xlabel(r'Time (s)')
    if amplitude_type == 'acceleration':
        axs[1].set_ylabel(r'$P$-wave acceleration ($\rm m\,s^{-2}$)')
    elif amplitude_type == 'velocity':
        axs[1].set_ylabel(r'$P$-wave velocity ($\rm m\,s^{-1}$)')
    else:
        axs[1].set_ylabel(r'$P$-wave amplitude ($\rm m$)')
        
    plt.show()

    # s-wave
    # create plot of seismic signal
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 6))
    fig.subplots_adjust(hspace=0)

    rc('text', usetex=True)
    rc('font', size=13)
    rc('legend', fontsize=12.5)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    axs[0].plot(waveform_df['t'], waveform_df['sx'], 'b')
    axs[1].plot(waveform_df['t'], waveform_df['sy'], 'r')
    axs[2].plot(waveform_df['t'], waveform_df['sz'], 'k')

    axs[0].set_xlim([0, np.max(waveform_df['t'])])
    #ax.set_ylim([0, 2])
    axs[2].set_xlabel(r'Time (s)')
    if amplitude_type == 'acceleration':
        axs[1].set_ylabel(r'$S$-wave acceleration ($\rm m\,s^{-2}$)')
    elif amplitude_type == 'velocity':
        axs[1].set_ylabel(r'$S$-wave velocity ($\rm m\,s^{-1}$)')
    else:
        axs[1].set_ylabel(r'$S$-wave amplitude ($\rm m$)')
        
    plt.show()
