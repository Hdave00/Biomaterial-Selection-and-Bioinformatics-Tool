# Biomaterial-Selection-and-Bioinformatics-Tool

## NOTE-- This repository is DEPRECATED! Refer to [This New Repo](https://github.com/Hdave00/Biomat-test)

## IMPORTANT -- Due to a python module failure, in matplotlib, the database is running into seg faults on streamlit due to how the plotting (using plotly for local search and pymatgen for Materials Project API) works, and sql databases are being queried. As a result I went down the debug rabbit hole and found that there was heavy virtual environment corruption and pycache files (*pyc) were also heavily corrupted. Those were pushed to the Streamlit Deployment cloud for production and as a result of that, even after countless redeployments, re-caching, complete "root-up" refactoring, the app kept crashing on streamlit with Seg-Faults and even local segfaults.




