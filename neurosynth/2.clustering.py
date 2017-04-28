# cluster data for dimensionality reduction
# for use in encoding model
# based on: https://nilearn.github.io/auto_examples/03_connectivity/plot_rest_clustering.html#sphx-glr-auto-examples-03-connectivity-plot-rest-clustering-py

from nilearn import input_data
import pickle

# The NiftiMasker will extract the data on a mask. We do not have a
# mask, hence we need to compute one.
#
# This is resting-state data: the background has not been removed yet,
# thus we need to use mask_strategy='epi' to compute the mask from the
# EPI images
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)

func_filename = 'all_ns_data.nii.gz'
# The fit_transform call computes the mask and extracts the time-series
# from the files:
fmri_masked = nifti_masker.fit_transform(func_filename)

# We can retrieve the numpy array of the mask
mask = nifti_masker.mask_img_.get_data().astype(bool)

# Compute connectivity matrix: which voxel is connected to which
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

# Computing the ward for the first time, this is long...
from sklearn.cluster import FeatureAgglomeration
# If you have scikit-learn older than 0.14, you need to import
# WardAgglomeration instead of FeatureAgglomeration
import time
start = time.time()
ward = FeatureAgglomeration(n_clusters=1000, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(fmri_masked)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism
start = time.time()
ward = FeatureAgglomeration(n_clusters=2000, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(fmri_masked)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

pickle.dump(ward,open('ward.pkl','wb'))
