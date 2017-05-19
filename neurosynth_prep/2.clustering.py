# cluster data for dimensionality reduction
# for use in encoding model
# based on: https://nilearn.github.io/auto_examples/03_connectivity/plot_rest_clustering.html#sphx-glr-auto-examples-03-connectivity-plot-rest-clustering-py

from nilearn import input_data
import pickle
import os
from sklearn.cluster import FeatureAgglomeration

nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)
n_clusters=2000

if os.path.exists('../data/neurosynth/ward.pkl'):
    print('loading saved ward clustering')
    ward=pickle.load(open('../data/neurosynth/ward.pkl','rb'))
else:
    print('runnign ward clustering')
    # The NiftiMasker will extract the data on a mask. We do not have a
    # mask, hence we need to compute one.
    #
    # This is resting-state data: the background has not been removed yet,
    # thus we need to use mask_strategy='epi' to compute the mask from the
    # EPI images

    func_filename = '../data/neurosynth/all_ns_data.nii.gz'
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
    # If you have scikit-learn older than 0.14, you need to import
    # WardAgglomeration instead of FeatureAgglomeration
    import time
    start = time.time()
    ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                                linkage='ward', memory='nilearn_cache')
    ward.fit(fmri_masked)
    print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

    pickle.dump(ward,open('../data/neurosynth/ward.pkl','wb'))

    fmri_reduced = ward.transform(fmri_masked)

    pickle.dump(fmri_reduced,open('../data/neurosynth/neurosynth_reduced.pkl','wb'))



# plot results
plot=False
if plot:
    from nilearn.plotting import plot_roi, plot_epi, show
    fmri_mean = nifti_masker.fit_transform('../data/neurosynth/mean_ns_data.nii.gz')

    # Unmask the labels

    # Avoid 0 label
    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)

    from nilearn.image import mean_img
    mean_func_img = 'mean_ns_data.nii.gz'


    first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation",
                          display_mode='xz')

    # common cut coordinates for all plots
    cut_coords = first_plot.cut_coords
    labels_img.to_filename('parcellation.nii.gz')

    # Display the original data
    plot_epi(nifti_masker.inverse_transform(fmri_mean),
             cut_coords=cut_coords,
             title='Original (%i voxels)' % fmri_mean.shape[1],
             vmax=fmri_mean.max(), vmin=fmri_mean.min(),
             display_mode='xz')

    # A reduced data can be create by taking the parcel-level average:
    # Note that, as many objects in the scikit-learn, the ward object exposes
    # a transform method that modifies input features. Here it reduces their
    # dimension
    fmri_reduced = ward.transform(fmri_mean)

    # Display the corresponding data compressed using the parcellation
    fmri_compressed = ward.inverse_transform(fmri_reduced)
    compressed_img = nifti_masker.inverse_transform(fmri_compressed)

    plot_epi(compressed_img, cut_coords=cut_coords,
             title='Compressed representation (2000 parcels)',
             vmax=fmri_mean.max(), vmin=fmri_mean.min(),
             display_mode='xz')
