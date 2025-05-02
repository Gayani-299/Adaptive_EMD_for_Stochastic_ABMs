# Adaptive Evolutionary Model Discovery (AGP-EMD) for Stochastic ABMs

This repository provides an adapted implementation of the **Evolutionary Model Discovery (EMD)** framework by Gunaratne & Garibay (2020), extended to support **stochastic agent rules** and **Adaptive Genetic Programming (AGP)** for Python-based Agent-Based Models (ABMs).

We apply this framework to a simulated **pre-evacuation scenario** using a Python-based ABM (`StationSim_IGSS.py`), replacing the original NetLogo integration with a fully Python-native pipeline.

---

## 🔧 Key Features

- ✅ Genetic Programming-based rule discovery
- ✅ Support for **stochastic primitives** (e.g., random rule selection)
- ✅ **Adaptive GP enhancements**:
  - Dynamic population resizing
  - Local early stopping
  - Elite individual retention across runs
- ✅ Works with **Python-based ABMs** (no NetLogo required)
- ✅ Evaluation based on multiple simulation replications and RMSE-based fitness

---

## 📁 Project Structure

AGP_2/
│
├── example/ # Application-specific case study files
│ ├── Factors.py # Custom primitives for the evacuation case
│ ├── FactorImportances.py # using Random Forest
│ ├── FactorScores.csv # Scoring output of rules (generated)
│ ├── results_all.csv # Empirical benchmark data
│ ├── run_emd.py # Main script to launch evolution
│ └── StatioSim_IGSS.py # Python-based ABM (StationSim variant)
│
├── src/
│ └── EvolutionaryModelDiscovery/ # Core AGP-EMD framework
│ ├── init.py # Modified: to incorporate elite individuals between runs
│ ├── ABMEvaluator.py # Modified: links evolved rules to Python ABM
│ ├── Factor.py # Unchanged from Gunaratne (2020)
│ ├── FactorGenerator.py # Unchanged from Gunaratne (2020)
│ ├── FactorImportances.py # Unchanged from Gunaratne (2020)
│ ├── PrimitiveSetGenerator.py # Unchanged from Gunaratne (2020)
│ ├── PythonWriter.py # Modified: replaces NetLogoWriter
│ ├── SimpleDEAPGP.py # Modified: AGP logic
│ └── Util.py # Unchanged from Gunaratne (2020)
│
├── setup.py 
├── LICENSE.txt # GPL v3 license from original work
└── README.md # This file


---

## 🚀 How to Run
This repo uses conda to easily manage Python installation and dependencies! You can set up the required python environment with:
conda env create -f environment.yml

And then load it with:
conda activate emd



This work is adapted from the open-source Evolutionary Model Discovery (EMD) framework:

Gunaratne, C., & Garibay, I. (2020). Evolutionary model discovery of causal factors behind the socio-agricultural behavior of the Ancestral Pueblo. PLOS ONE, 15(12), e0239922. https://doi.org/10.1371/journal.pone.0239922

The original code is licensed under the GNU General Public License v3 (GPL-3.0). All adaptations in this repository, including AGP logic, stochastic evaluation, and Python model integration, are shared under the same license.

Modified components:
SimpleDEAPGP.py: Introduced Adaptive GP with dynamic population and elite retention

ABMEvaluator.py: Replaced NetLogo model simulation with Python ABM support

PythonWriter.py: Injects evolved rules into Python ABMs (replaces NetLogoWriter)

Other core components (FactorGenerator.py, Util.py, etc.) remain unchanged from the original EMD repository.