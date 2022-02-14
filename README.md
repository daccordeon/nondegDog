# nondegDog
*Nondegenerate internal squeezing for gravitational-wave detection*

James Gardner, February 2022 (started February 2021)

Current build found [here](https://github.com/daccordeon/nondegDog).

---
Guide to replicating the results from the paper:

- Download the repository and check that your system satisfies the requirements below.
- Run all cells of source/paper_figures_nondegenerate_internal_squeezing.ipynb in Jupyter. This will save Figures 3--6 from the paper into the source/ directory and create data files in optimal_angles/ and data_of_tolerance_to_detection_loss/. This notebook uses the functions in interferometer.py which are exported from Mathematica (solve_model_nondegenerate_internal_squeezing.nb and ExportToPython.nb --- the former also compares the results to stable optomechanical filtering and credit to Juan Ripoll for the latter). The raw export is found in nIS_Mathematica_to_Python_via_Ripoll.py.
- Other results from the text of the paper are found throughout the remaining Jupyter and Mathematica notebooks, particularly 
workshop_nondegenerate_internal_squeezing.ipynb and stability_and_other_results_nondegenerate_internal_squeezing.nb (for Figure 2), although some code provided is legacy only. The squeezing threshold is calculated in threshold_nondegenerate_internal_squeezing.nb. Figure 1 is generated from source/plots/diagrams.svg
- Similar code is available for degenerate internal squeezing (see files named \*_degenerate_internal_squeezing\* respectively).
- The other directories (thesis/, talks/, paper/figures_for_publication/) are for version control and do not relate to the results pipeline. 
- Contact the authors for any technical enquiries at <james.gardner@anu.edu.au>.

Requirements:
- ipython==5.5.0
- jupyter==1.0.0
- jupyter-client==6.1.12
- jupyter-console==6.0.0
- jupyter-core==4.7.1
- matplotlib==3.0.3
- multiprocess==0.70.12.2
- numpy==1.16.2
- p-tqdm==1.3.3
- scipy==1.2.1

---
file structure
```bash
.
├── .gitignore
├── LICENSE
├── README.md
├── source
│   ├── ...
│   ├── data_of_tolerance_to_detection_loss
│   │   └── ...
│   ├── optimal_angles
│   │   └── ...
│   └── plots
│       └── ...
├── paper
│   └── figures_for_publication
│       └── ...
├── talks
│   └── ...
└── thesis
    └── ...
```
[//]: # (tree -I '*.pdf|*.png')
