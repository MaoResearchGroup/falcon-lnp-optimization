# falcon-lnp-optimization
Repository for the paper "FALCON: A Machine Learning-Driven Platform for Multi-Objective Optimization of mRNA Lipid Nanoparticle Composition Towards Cell Type-Selective Transfection"

### 🦅 What is FALCON?

**FALCON** (_**F**ramework for **A**ctive-**L**earning driven **C**ompositional **O**ptimization of **N**anoparticles_) is a closed-loop experimental-computational pipeline developed by the Hai-Quan Mao lab for intelligent and accelerated design of cell type-selective lipid nanoparticle (LNP) formulations.  

### 🔑 Key Capabilities

- **Multi-objective optimization.** Learns to simultaneously *maximize* delivery to desired cell types while *minimizing* off-target effects, improving the efficacy and safety profile of LNPs.

- **Data-efficient formulation design.** Requires only a sparse initial dataset to begin optimization and can rapidly identify informative candidates to test via active learning, reducing experimental burden.

- **Exhaustive and rational search.** Surrogate model-guided search algorithms test thousands of formulation candiates in silico, outperform brute-force grid search in experiments, and enable interpretability of cell type-selective design principles.
  
- **Modular and flexible architecture**. Supports different cell types, cargo types, and optimization goals.

- **End-to-end integration.** FALCON processes raw input data and outputs formatted formulation tables compatible with automated liquid handlers (e.g., MANTIS)
  
![FALCON Pipeline Schematic](manuscript_figures/fig1_0530.jpeg)
<details>
<summary><strong>Figure 1. Schematic Overview of Full FALCON workflow</strong></summary>

**a.** Pipeline starting point. An initial sparse grid-search library is designed by varying parameters influencing the relative ratio of LNP lipid components and payload. Cell type objectives (e.g., maximize or minimize transfection) are defined to enable cell-selective optimization.  
**b.** Illustration of the DTBL framework for ML-driven optimization. Each cycle consisted of 4 steps:  
  1. Candidate LNPs are formulated and treated to cells  
  2. Transfection is quantified by luciferase expression  
  3. The resulting composition-function dataset is used to train or refine a surrogate ML model  
  4. The trained model guides an exhaustive search to identify LNPs for the next round.
**c.** Pipeline endpoint. After convergence, the expanded dataset can be used to train a final model, which is interpreted using SHAP analysis to uncover design rules for cell-selective LNPs.  

*This figure was created with BioRender.com and is released under a [Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International license](https://creativecommons.org/licenses/by-nc-nd/4.0/).*

</details>

source data for experimentally obtained values reported in the manuscript are available in source_data. Figures derived from model outputs or computational analysis are reproducible with the codebase and script provided

## 📁 Repository Overview

## ⚙️ Environment Setup

We recommend using **Anaconda** to create a clean, reproducible environment that supports both Python and R.

Clone this repo and run:

```bash
conda env create -f environment.yml
conda activate falcon-env
jupyter notebook
