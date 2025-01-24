# Fluorescence Fluctuations Simulation: MUFASA Simulator

<div align="center">
<img src="img/Logo MUFASA.png" alt="Fluorescence Simulation" width="300"/>
</div>

This repository provides tools to simulate super-resolution imaging techniques such as STORM and PALM, as well as fluorescence fluctuations, using Markov chains to model transitions between quantum states. The simulations are designed to help researchers and students understand the underlying processes and visualize the effects of photophysics and camera imperfections on the final images.


| PALM Simulator | STORM Simulator | Fluctuations Simulator |
|:--------------:|:---------------:|:----------------------:|
| <img src="img/emitted_photons_poisson_palm.gif" width="300"> | <img src="img/emitted_photons_poisson_storm.gif" width="300"> | <img src="img/emitted_photons_poisson_FF.gif" width="300"> |




## Features 

- **Graphical User Interface (GUI):**
  - Simulate single-molecule behavior.
  - Simulate fluorescence in structured biological samples.
  - Predict parameters for various experimental setups.
  - Visualize laser power ranges and camera results.

- **Multi-Protocol Support:**
  - Includes blinking and fluorescence fluctuation protocols: Fluorescence Fluctuations (FF), STORM, PALM, Blinking
  - Models molecule transitions using Continuous-Time Markov Chains (CTMC).
 
- **Advanced Camera Effects:**
  - Simulates point spread function (PSF), undersampling, and noise.
  - Provides realistic output for evaluating super-resolution techniques.
  - Supports EMCCD, CCD, CMOS.


## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   
bash
   git clone https://github.com/YourUsername/Fluorescence_Fluctuations_Simulation.git
   cd Fluorescence_Fluctuations_Simulation


2.Run the application

Launch the main GUI by running the following command:

  
bash
   python GUI/main_page.py


## Usage 

### Single-Molecule Simulation
Navigate to the "Simulate One Molecule" page in the GUI to configure molecule parameters, such as excitation power, lifetime, and activation rates. Generate simulation results and visualize photon emission distributions.

### Structured Sample Simulation
Use the "Simulate an Image" page to upload biological structures and simulate fluorescence behavior across the structure.

### Parameter Prediction
Utilize the parameter prediction tools to optimize experimental setups for desired outcomes.

### Camera Effects
Explore the "Camera Results" and "Laser Range" tools to evaluate how noise, PSF, and laser power influence imaging.


## src Folder Description

The src folder contains the core scripts and modules responsible for simulation and processing. Here is an overview of its key components:

- simulate_ctmc.py: Handles simulations using Continuous-Time Markov Chains (CTMC) for a  single molecule.

- sofi_tools.py: A python implementation of SOFI SimulationTools. inspired by the following reference:

Girsault A, Lukes T, Sharipov A, Geissbuehler S, Leutenegger M, Vandenberg W, Dedecker P, Hofkens J, Lasser T.
SOFI Simulation Tool: A Software Package for Simulating and Testing Super-Resolution Optical Fluctuation Imaging.
PLoS One. 2016 Sep 1;11(9):e0161602.
DOI: 10.1371/journal.pone.0161602. PMID: 27583365; PMCID: PMC5008722.

- camera.py: Simulates camera effects like noise, point spread function (PSF), and undersampling.

- utils.py: Provides utility functions and helper methods for the simulation framework.

- Camera_output.ipynb: A Jupyter notebook for visualizing camera output simulations.

- Simulator.ipynb: A Jupyter notebook that demonstrates the overall simulation process.

- This folder is designed to provide a modular and expandable structure for different fluorescence simulation protocols.

