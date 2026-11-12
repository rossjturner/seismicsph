# hyglac

Hyglac ([`hyglac.py`](hyglac.py)) is a Python package for smoothed particle hydrodynamic simulations, calculation of dynamic pressure forces at fluid–solid interfaces, and seismic wave propagation, as described by Turner et al. (2026), in press. These three components can be used independently to simulate fluid flows, extract time-dependent forces from existing hydrodynamic simulations, or calculate synthetic seismic waveforms from prescribed force distributions.

## Standard workflow

Worked examples of the standard workflow for the four channel geometries and dam-break validation case considered by Turner et al. (2026) are included in the corresponding Jupyter notebooks in this repository.

### Hydrodynamic simulation

The first step in the workflow is to run a hydrodynamic simulation for the relevant channel geometry. This is achieved using the *run_seismic_sph* function in the *hyglac.py* module, but an existing hydrodynamic simulation can be used. Regions in the computational domain to be occupied by ice (boundary) particles and the initial location of fluid particles must be specified. See the worked examples for the expected form of these functions.

### Dynamic pressure calculation

The second step would in practice be run after the previous step, but can be run in isolation using the outputs of an existing smoothed particle hydrodynamical simulation (the function will likely require minor modifications to handle different simulation output formats). The *seismic_sph_interactions* function in the *hyglac.py* module identifies the water–ice interface and calculates the dynamic pressure acting on the ice boundary from the fluid particle pressures, using spatial averaging to reduce numerical fluctuations. The pressure is combined with the effective surface areas and unit normals of the interface particles to calculate the three components of the force acting on the ice. The outputs are saved to an HDF5 file for subsequent calculation of seismic waves.

### Seismic wave propagation

The final step in the workflow is the generation of seismic waves from the interaction data calculated in the previous step. The *seismic_sph_forces* function in the *hyglac.py* module combines the forces on a coarser spatial grid for computational efficiency, returning a *pandas* dataframe. The *seismic_sph_forces_defaults* function applies the settings used in Turner et al. (2026), including the exclusion of the vertical force component.

The location of a mock seismometer relative to the computational domain must be specified, in addition to the sample rate, duration and seismic velocity structure. The *earth_model* function defines the layered seismic velocity structure, while the *seismic_signals* function uses *pyprop8* to generate three-component displacement and velocity waveforms, returning a *pandas* dataframe. This function can be run for a range of seismometer locations around the channel to generate the suite of figures presented in Turner et al. (2026).

## Verification test

The [`seismicsph_testcase.ipynb`](verification_test/seismicsph_testcase.ipynb) notebook demonstrates the complete workflow and provides a computationally inexpensive test to verify that the code runs as expected. The simulation parameters are chosen to minimise runtime and output file sizes, allowing reference outputs to be included in the [`verification_test/`](verification_test/) directory for comparison. These parameters are not intended to be physically realistic, and the resulting outputs should not be interpreted as scientifically validated predictions.

## Contact

Ross Turner <turner.rj@icloud.com>
