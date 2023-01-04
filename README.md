# seismicsph

Smoothed particle hydrodynamic simulation and seismic wave propagation code described by Turner et al. (2023), in press.

## Standard workflow

Worked examples of the standard workflow are included in the following Jupyter notebook: [seismicsph_bend_45.ipynb](https://github.com/rossjturner/RAiSEHD/blob/main/seismicsph_bend_45.ipynb).

### Hydrodynamic simulation

The first step in the workflow is to run a hydrodyanic simulation for the relevant channel geometry. This is acheived using the _run_seismic_sph_ function in the _seismicsph_collisions.py_ module, but an existing hydrodynamic simulation can be used. Regions in the computational domain to be occupied by ice (boundary) particles must be specified in either case, and the inital location of fluid particles specifed in the former case. See the worked example for the expected form of thexe functions.

_git_ using:

```bash
git clone https://github.com/rossjturner/RAiSEHD.git
```

The package is installed by running the following command as an administrative user:

```bash
python setup.py install
```

## Contact

Ross Turner <<turner.rj@icloud.com>>
