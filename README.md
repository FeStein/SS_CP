# README.md

## cp_matroutines/
* cp_base.py - general Material class, contains slip system definitions, material parameters, elastic stress tensor, rotation handeling, some outpuits, history defintion, etc.
* solver.py - interface to dispatch the material point solver, e.g. make the material routines configurabel
* mat_XX.py - material routine
    * IPM_classic - classic staggered IPM
    * IPM_accelerated - accelerated solution of the IPM based on warm-starting
    * VP - visco_plastic


## doc/
contains the markdown/latex documentation of what I am implementing
* definitions.md - definitions for smalls train crystal plasticity
    * ipm.md - interior point method definition
    * vp.md - visco-plastic defintion

## examples/
Contains the numerical examples, which use the respective material routines
* ex_simple_shear.py - simple shear example purely deformation controlled, without global iterations
* ex_uniaxial_tension.py - uniaxial tension test, with stress values as a residual, uses partitioned material tangent

## Coding Style
* for Derivatives you can use Jax
