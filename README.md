# Cost-Sensitive Data Acquisition (CDA) for Incomplete Datasets

This repository contains the experimental functions conducted for the paper "Cost-Sensitive Data Acquisition for Incomplete Datasets". Here's an overview of the contents:

- **`results`**: Contains important results for the experiments.
- **`data-workload`**: Includes four example datasets and workloads. The TPCDS datasets are too big, thus a link to the data generator is given.
- **`utils`**: Contains functional blocks used in the experiments, including functions like baseline methods, missing value generation, and workload loading, as well as cda subfunctions (continue updating).

## Quick Start Example

To conduct a simple quick start example using the forest dataset from the UCI data repository, considering using the MICEforest model as the predictive model, a complete random missing mechanism, and a missing rate of 30%, follow these steps:

1. **Set Up Your Python Environment**:
   - If you are using a virtual environment, create and activate it:
     ```bash
     virtualenv venv
     source venv/bin/activate  # for Unix/Linux
     venv\Scripts\activate     # for Windows
     ```
   - If you are not using a virtual environment, skip this step.

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Project:**:
   ```bash
   python3 quick-start.py
   ```

   - The running process may take more than 10 minutes, depending on your hardware. You may soon get the sense of the benefit from employing a CDA algorithm for incomplete datasets.

The users can set the following parameters:
```bash
missing_rate = 0.3            # the ratio of missing values in raw table T.
pattern = 0                   # the pattern of missing values in raw table T. 0: MCAR, 1:MAR, 2: MNAR 
budget = 0.05                 # the ratio of missing values we can comlete by acquiring data
acquisition_rounds = 3        # the number of rounds to use up the budgets
dataset = 'tpcds'             # Here we have four datasets to select: 'census13', 'forest10', 'nursery', 'tpcds'
menu_type = 'RMenu'           # valid value including 'RMenu', 'OMenu', and 'SMenu'
confidence_interval = 0.9     # the confidence interval for conformal confidence control
```
