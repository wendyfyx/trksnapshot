# trksnapshot

## Example Commands

To visualization bundle with glass brain (provided by DIPY)
```bash
python snapshot_bundle.py \
      -i /path_to_tractogram/bundle.trk \
      -glass ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii \
      --show_bundle
```

To visualization bundle with pial surfaces
```bash
python snapshot_bundle.py \
      -i /path_to_tractogram/bundle.trk \
      -glass /path_to_surfaces/pial_left.nii /path_to_surfaces/pial_right.nii \
      --show_bundle
```

To visualization bundle with pial surfaces with BUAN output
```bash
python snapshot_bundle.py \
      -i /path_to_tractogram/bundle.trk \
      -f /path_to_buan_results/pvals.npy \
      -ctitle pvalues -auto -cbar /path_to_output_folder/cbar.png \
      -glass /path_to_surfaces/pial_left.gii.gz /path_to_surfaces/pial_right.gii.gz
      --show_bundle \
      -o /path_to_output_folder/bundle.png
```
