# trksnapshot

## Installation
 `pip install trksnapshot` ([project page]([https://pypi.org/project/trksnapshot))

## Description
**trksnapshot** is a CLI for creating bundle visualizations (created using FURY and DIPY). It supports  
1. Visualizing 3D bundles in an interactive window with custom color options and camera view. These include per-bundle color and along-tract color using BUAN (for tractometry results).
2. Supports plotting bundle with glass brain and pial surfaces.
3. Rotate the camera however you like and save camera settings to produce consistent plots.
4. Save snapshot images for publication-ready figures!

All bundle visualizations in our recent work [*Microstructural mapping of neural pathways in Alzheimer's disease using macrostructure-informed normative tractometry*](https://doi.org/10.1002/alz.14371) were created with this script. Here's one of the figures:

<img src="assets/atlas_bundles.jpg" alt="bundle_atlas" width="800">

## Usage

[**RECOMMENDED**] To visualization bundle(s) with default pial surfaces (in MNI space)
```bash
trksnapshot \
      -i /path_to_tractogram/bundle*.trk \
      -glass default_pial \
      -show
```
You may also provide your own pial surface files
```bash
trksnapshot \
      -i /path_to_tractogram/bundle*.trk \
      -glass /path_to_surfaces/pial_left.nii /path_to_surfaces/pial_right.nii \
      -show
```

To visualization bundle(s) with default glass brain (must be in MNI space)
```bash
trksnapshot \
      -i /path_to_tractogram/bundle*.trk \
      -glass default_glass \
      -show
```

To visualization bundle with pial surfaces with BUAN output
```bash
trksnapshot \
      -i /path_to_tractogram/bundle.trk \
      -f /path_to_buan_results/pvals.npy \
      -ctitle pvalues -auto -cbar /path_to_output_folder/cbar.png \
      -glass /path_to_surfaces/pial_left.gii.gz /path_to_surfaces/pial_right.gii.gz
      -show \
      -o /path_to_output_folder/bundle.png
```

## Citations
1. [Microstructural mapping of neural pathways in Alzheimer's disease using macrostructure-informed normative tractometry
](https://doi.org/10.1002/alz.14371)
2. [FURY: advanced scientific visualization](10.21105/joss.03384)
