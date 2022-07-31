K-dens data augmentation project

Kdens uses UMAP (McInnes2018) to sample synthetic data points based on a measure of densitiy in thier low dimensonal representation.


Usage:
kdens.py [-h] [-nc NUM_CAND] [-uk UMAP_K] [-kdk KDENS_K] [-gnstd GNOISE_STD] -i INPUT [-pmd P_MIN_DENS]

Inputs are required to be .csv files with the first row being empty or feature labels, first column being empty or sample indicies 
and second column being binary class labels in the form of "0" or "1".

Required python packages:
- numpy 
- math
- sklearn.preprocessing 
- sklearn.metrics
- scipy.spatial.distance 
- umap
- csv
- time
- os
- argparse
