# Software environments

Recorded original analysis environment: Python 3.13.1, NumPy 2.4.2, SciPy 1.17.0, pandas 3.0.0, scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.6.0, Matplotlib 3.10.8, and openpyxl 3.1.5. The original environment also included seaborn 0.13.2, but the included notebook source uses Matplotlib styles and does not import seaborn.

Recorded revised nested-analysis environment: Python 3.12.13 with the same core NumPy, SciPy, pandas, scikit-learn, joblib, and threadpoolctl versions. An earlier standalone data audit used NumPy 2.4.4; that audit is not the primary model-selection environment.

`requirements.txt` pins the scientific dependencies, including the original plotting/Excel-reader versions. `.python-version` records Python 3.12.13. Optional JupyterLab is installed separately and is not pinned to a historically recorded version. A complete operating-system-specific dependency lock is not supplied.

The cleaned-release smoke-check environment is recorded in `release_validation.json`; validation with an available renderer is distinct from reinstalling and testing every pinned package on a fresh system. PyTorch and deployment/export dependencies are omitted because those proposed extensions are outside the analyses distributed here.
