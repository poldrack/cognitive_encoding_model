# cluster data for dimensionality reduction
# for use in encoding model

from nilearn import input_data

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
