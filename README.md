# trksnapshot

## Installation
 `pip install trksnapshot`

## Description
**trksnapshot** is a CLI for creating bundle visualizations (created using FURY and DIPY). It supports  
1. Visualizing 3D bundles in an interactive window with custom color options, mask/ROIs and camera view. These include per-bundle color, per-mask colors, along-tract color for visualizing tractometry results.
2. Plotting bundle with glass brain and pial surfaces.
3. Saving custom camera settings to produce consistent plots.
4. Saving snapshot images for publication-ready figures!

All bundle visualizations in our recent work [*Microstructural mapping of neural pathways in Alzheimer's disease using macrostructure-informed normative tractometry*](https://doi.org/10.1002/alz.14371) were created with this script. Here's one of the figures:

<img src="examples/atlas_bundles.jpg" alt="bundle_atlas" width="800">

## Usage

To visualization bundle(s) with default glass brain (must be in MNI space)
```bash
trksnapshot \
      -i /path_to_tractogram/bundle*.trk \
      -glass default_glass \
      -show
```
To visualization bundle(s) with default pial surfaces (in MNI space)
```bash
trksnapshot \
      -i /path_to_tractogram/bundle*.trk \
      -glass default_pial \
      -show
```

To visualize bundle with glass brain with along-tract values saved in `.npy` or `.txt` files (such as those from BUAN), with a given color map, minimum and maxmium value
```bash
trksnapshot \
      -i AF_L.trk \
      -c /path_to_buan_results/pvals.npy \
      -ctitle pvalues -auto -cbar /path_to_output_folder/cbar.png \
      -glass default_glass \
      -cmap viridis \
      -vmax 1 \
      -vmin 0.0 \
      -show \
      -o /path_to_output_folder/bundle.png
```

To visualize multiple bundles with distinct colors ([named colors from matplotlib](https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors))
```bash
trksnapshot \
      -i AF_L.trk CST_L.trk IFOF_L.trk \
      -c red green blue \
      -glass default_glass \
      -show \
      -o /path_to_output_folder/bundle.png
```

To visualize bundle(s) and mask(s) with provided named colors and opacity
```bash
trksnapshot \
      -i AF_L.trk CST_L.trk IFOF_L.trk \
      -c red green blue \
      -mask roi1.nii.gz roi2.nii.gz \
      -mask_color red blue \
      -mask_opacity 0.1
      -show \
      -o /path_to_output_folder/bundle.png
```
>[!NOTE]
> - Remove `-show` and only specify `-o output.png` to save output image without showing interactive window  
> - Add `-crop` to remove white borders in the output image so you figures are publication-ready! 

## Citations
1. [Microstructural mapping of neural pathways in Alzheimer's disease using macrostructure-informed normative tractometry
](https://doi.org/10.1002/alz.14371)
2. [FURY: advanced scientific visualization](10.21105/joss.03384)
