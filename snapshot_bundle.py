import argparse
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.stats.analysis import assignment_map
from dipy.viz import actor, window


def bundle_colors_from_values(values, indx, cmap="YlOrBr_r", vmin=0, vmax=1, label='values', save_cmap=None):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = cmap.to_rgba(values)[:,:3]
    colors = [tuple(i) for i in list(colors)]
    disks_color = []
    for i in range(len(indx)):
        disks_color.append(tuple(colors[indx[i]]))
    if save_cmap is not None:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(cmap, cax=ax, orientation='horizontal')
        cb.set_label(label=label, size=14)
        fig.savefig(save_cmap)
    return disks_color



def run(args):

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    # Set camera given camera setting file
    if args.in_campath is not None:
        with open(args.in_campath, 'rb') as f:
            cam_settings = pickle.load(f)
        scene.set_camera(position=cam_settings['pos'], 
                        focal_point=cam_settings['foc'],
                        view_up=cam_settings['vup'])

    for i, bundle_path in enumerate(args.in_bpath):
        bundle = load_trk(bundle_path, "same", bbox_valid_check=False).streamlines


        # Make segment colors if values are supplied
        colors=None
        if args.value_file is not None:
            values = np.load(args.value_file[i])
            indx = assignment_map(bundle, bundle, len(values))
            indx = np.array(indx)
            vmin = values.min() if args.auto_range else args.vmin
            vmax = values.max() if args.auto_range else args.vmax
            colors = bundle_colors_from_values(values, indx, cmap=args.cmap, 
                                            vmin=vmin, vmax=vmax, label=args.cmap_title,
                                            save_cmap=args.out_cbarpath)
        
        if args.as_points:
            stream_actor = actor.point(bundle.get_data(), colors=window.colors.green, point_radius=0.3)
        else:
            stream_actor = actor.line(bundle, fake_tube=True, linewidth=6, colors=colors)

        scene.add(stream_actor)

    if args.glass_brain_path is not None:
        MASK, AFFINE = load_nifti(args.glass_brain_path)
        scene.add(actor.contour_from_roi(MASK, affine=AFFINE, color=np.array([0, 0, 0]), opacity=0.08))
  
    # Show bundle window
    if args.show_bundle:
        window.show(scene, size=(1200,1200))

    # Save camera settings
    if args.out_campath is not None:
        cam_settings = {}
        cam_settings['pos'], cam_settings['foc'], cam_settings['vup'] = scene.get_camera()
        with open(args.out_campath, 'wb') as f:
            print(f"Saving camera settings to {args.out_campath}")
            pickle.dump(cam_settings, f)
    
    # Save bundle snapshot
    if args.out_bpath is not None:
        print(f"Saving bundle snapshot to {args.out_bpath}")
        window.record(scene, size=(1200, 1200), out_path=args.out_bpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bpath', '-i', nargs="+", type=str, required=True,
                        help="Input bundle file, i.e. ./bundle.trk")
    parser.add_argument('--out_bpath', '-o', type=str, default=None, 
                        help="Output filepath to save bundle snapshot, i.e. ./bundle.png")
    parser.add_argument('--in_campath', '-si', type=str, default=None, 
                        help="Input camera setting file, i.e ./cam.pkl")
    parser.add_argument('--out_campath', '-so', type=str, default=None, 
                        help="Output camera setting file, i.e ./cam.pkl")
    
    # Segment arguments
    parser.add_argument('--value_file', '-f', nargs="*", type=str, required=None,
                        help="Input value path to, i.e. ./pval.npy")
    parser.add_argument('--cmap', '-c', type=str, default='YlOrBr_r', 
                        help="Colormap name from matplotlibm i.e. Blues")
    parser.add_argument('--cmap_title', '-ctitle', type=str, default='values', 
                        help="Colormap title")
    parser.add_argument('--out_cbarpath', '-cbar', type=str, default=None, 
                        help="Output filepath to save the colorbar")
    parser.add_argument('--vmin', '-vmin', type=float, default=0, 
                        help="Minimum value of colormap")
    parser.add_argument('--vmax', '-vmax', type=float, default=1, 
                        help="Maximum value of colormap")
    parser.add_argument('--auto_range', '-auto', action="store_true", 
                        help="Automatically define vmin and vmax, override specified values")
    
    # Visualization option
    parser.add_argument('--show_bundle', '-show', action="store_true", 
                        help="Show the interactive window")
    parser.add_argument('--glass_brain_path', '-glass', type=str, default=None,
                        help="Glass brain to plot")
    parser.add_argument('--as_points', '-points', action='store_true', 
                        help="Plots the bundle using points instead of streamtubes.")
    args = parser.parse_args()

    run(args)