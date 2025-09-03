import sys

import logging
from warnings import warn
import pickle
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2   
from nibabel.streamlines.array_sequence import ArraySequence

from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.io.surface import load_gifti
from dipy.stats.analysis import assignment_map
from dipy.viz import actor, window
from dipy.tracking.streamline import transform_streamlines
from fury.utils import map_coordinates_3d_4d

default_glass_brain="~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"

SCALE=1.6
CAM_SETTINGS = {
    'Axial': {'view_up': (0,1,0), 'focal_point': (0,0,0), 'position': (0,0,SCALE) },
    'Sagittal_L': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (-1*SCALE, 0, 0) },
    'Sagittal_R': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (SCALE, 0, 0) },
    'Coronal_A': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, SCALE, 0)},
    'Coronal_P': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, -1*SCALE, 0)}
    }

# Custom discrete colormap
DISCRETE_CMAP = list(plt.cm.tab10(np.arange(10))) + [np.array(mpl.colors.to_rgba('crimson')),
                                                     np.array(mpl.colors.to_rgba('indigo'))]

def autocrop(image, border=10):
    '''
        Automatically crop image to remove whole rows/columns with white space
        Adatped from https://stackoverflow.com/a/14211727
    '''
    image_data = np.asarray(image)
    image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0)<255)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1)<255)[0]
    cropBox = (min(non_empty_rows)-border, max(non_empty_rows)+border, 
               min(non_empty_columns)-border, max(non_empty_columns)+border)
    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return image_data_new


def colors_from_values(values, indx=None, cmap="YlOrBr_r", 
                       vmin=0, vmax=1, vcenter=None, label="values", save_cmap=None, ):
    # Get value range
    if vmin is None:
        vmin = np.percentile(values, 5) #min(values)
    if vmax is None:
        vmax = np.percentile(values, 95) #max(values)
    logging.info(f"Plotting values with using range [{vmin:.4f},{vmax:.4f}].")

    # Create colormap and map values to colors
    if vcenter is not None and vmin < vcenter:
        maxdist = max(abs(vmin - vcenter), abs(vmax - vcenter))
        vmin = vcenter - maxdist
        vmax = vcenter + maxdist
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = cmap.to_rgba(values)[:, :3]
    
    if indx is None:
        colors_mapped = colors
    else:
        # Creating segment color based on assignment map indx
        colors = [tuple(i) for i in list(colors)]
        colors_mapped = []
        for i in range(len(indx)):
            colors_mapped.append(tuple(colors[indx[i]]))

    # Plot 
    if save_cmap is not None:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(cmap, cax=ax, orientation="horizontal")
        cb.set_label(label=label, size=14)
        fig.savefig(save_cmap)

    return colors_mapped

def get_default_pial():
    from nilearn.datasets import load_fsaverage
    fsaverage_meshes = load_fsaverage()
    pial_l = fsaverage_meshes['pial'].parts['left']
    pial_r = fsaverage_meshes['pial'].parts['right']
    faces_r_shifted = pial_r.faces + len(pial_l.coordinates)
    vertices = np.vstack((pial_l.coordinates, pial_r.coordinates))
    faces = np.vstack((pial_l.faces, faces_r_shifted))
    return vertices, faces

def run(args):

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    # Set camera given camera setting file
    if args.in_campath is not None:
        with open(args.in_campath, "rb") as f:
            cam_settings = pickle.load(f)
        scene.set_camera(
            position=cam_settings["pos"],
            focal_point=cam_settings["foc"],
            view_up=cam_settings["vup"],
        )
    else:
        cam=CAM_SETTINGS[args.cam_view]
        logging.info(f'Camera is set to {args.cam_view}.')
        scene.set_camera(**cam)

    for i, bundle_path in enumerate(args.in_bpath):
        bundle = load_trk(
            bundle_path, "same", bbox_valid_check=False
        ).streamlines
        logging.info(f'Loaded {bundle_path}.')

        # Make segment colors if values are supplied
        colors = None
        if args.value_file is not None:
            if args.value_file[0].endswith('.npy'): # Along-tract segment color
                values = np.load(args.value_file[i])
                indx = assignment_map(bundle, bundle, len(values))
                indx = np.array(indx)

            elif args.value_file[0].endswith('nii.gz'): # Volume color
                img_data, affine = load_nifti(args.value_file[0])
                bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))
                bundle_native = ArraySequence(bundle_native).get_data()
                values = map_coordinates_3d_4d(img_data, bundle_native).T
                indx = None

            colors = colors_from_values(
                values,
                indx=indx,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                label=args.cmap_title,
                save_cmap=args.out_cbarpath,
            )

        elif args.per_bundle_color is not None:
            colors = np.array(mpl.colors.to_rgba(args.per_bundle_color[i]))

        if args.as_points:
            stream_actor = actor.point(
                bundle.get_data(), colors=window.colors.green, point_radius=0.3
            )
        else:
            stream_actor = actor.line(
                bundle, 
                fake_tube=args.fake_tube, 
                linewidth=args.linewidth, 
                colors=colors, 
                opacity=args.opacity
            )

        scene.add(stream_actor)

    if args.glass_brain_path is not None:
        for i, surf_path in enumerate(args.glass_brain_path):

            # NIFTI image
            if 'default_glass' in surf_path:
                surf_path = default_glass_brain
                logging.info(f"Using default glass brain path: {surf_path}.")

            fl = surf_path
            ends = fl.endswith

            if ends(".nii.gz") or ends(".nii"):
                MASK, AFFINE = load_nifti(surf_path)
                scene.add(
                    actor.contour_from_roi(
                        MASK,
                        affine=AFFINE,
                        color=np.array([0, 0, 0]),
                        opacity=0.08,
                    )
                )

            # GIFTI image
            if 'default_pial' in fl or ends(".gii.gz") or ends(".gii"):
                if 'default_pial' in fl:
                    vertices, faces = get_default_pial()
                    logging.info('Using default pial surface...')
                else:
                    surface = load_gifti(surf_path)
                    vertices, faces = surface
                    if len(vertices) and len(faces):
                        vertices, faces = surface
                    else:
                        warn(
                            "{} does not have any surface geometry.".format(
                                args.glass_brain_path
                            )
                        )

                colors = np.zeros((vertices.shape[0], 3))

                surf_actor = actor.surface(
                    vertices, faces=faces, colors=colors
                )
                surf_actor.GetProperty().SetOpacity(0.06)

                scene.add(surf_actor)

    # Show bundle window
    if args.show_bundle:
        window.show(scene, size=(1200, 1200))

    # Save camera settings
    if args.out_campath is not None:
        cam_settings = {}
        cam_settings["pos"], cam_settings["foc"], cam_settings["vup"] = (
            scene.get_camera()
        )
        with open(args.out_campath, "wb") as f:
            logging.info(f"Saving camera settings to {args.out_campath}")
            pickle.dump(cam_settings, f)

    # Save bundle snapshot
    if args.out_bpath is not None:
        if args.auto_crop:
            arr = window.snapshot(scene, size=(1200, 1200))
            arr = autocrop(arr, border=10)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(args.out_bpath, arr)

            logging.info(f"Saved cropped bundle snapshot to {args.out_bpath}")
        else:
            logging.info(f"Saved bundle snapshot to {args.out_bpath}")
            window.record(scene, size=(1200, 1200), out_path=args.out_bpath)


def main():
    logging.basicConfig(stream=sys.stdout,
                format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                datefmt='%Y-%m-%d %H:%M:%S',
                encoding='utf-8', level=logging.INFO, force=True)
    

    parser = argparse.ArgumentParser()
    parser.add_argument( "--in_bpath", "-i", nargs="+", type=str, required=True, 
                        help="Input bundle file(s), i.e. ./bundle.trk", )
    parser.add_argument( "--out_bpath", "-o", type=str, default=None, 
                        help="Output filepath to save bundle snapshot, i.e. ./bundle.png", )
    
    # Cam settings args
    parser.add_argument( "--cam_view", "-cam", type=str, default="Axial", 
                        help="Select a preset cam view, from Axial (default), Saggital_L/R, Coronal_A/P", )
    parser.add_argument( "--in_campath", "-si", type=str, default=None, 
                        help="Input camera setting file, i.e ./cam.pkl", )
    parser.add_argument( "--out_campath", "-so", type=str, default=None, 
                        help="Output camera setting file, i.e ./cam.pkl", )
    
    # Along-tract color args
    parser.add_argument( "--value_file", "-f", nargs="*", type=str, default=None, 
                        help="Input value path, can be nifti or .npy for along-tract colors", )
    parser.add_argument( "--cmap", "-cmap", type=str, default="YlOrBr_r", 
                        help="Colormap name from matplotlib or cmasher i.e. Blues", )
    parser.add_argument( "--cmap_title", "-ctitle", type=str, default="values", 
                        help="Colormap title", )
    parser.add_argument( "--out_cbarpath", "-cbar", type=str, default=None, 
                        help="Output filepath to save the colorbar", )
    parser.add_argument( "--vmin", "-vmin", type=float, default=None, 
                        help="Minimum value of colormap", )
    parser.add_argument( "--vmax", "-vmax", type=float, default=None, 
                        help="Maximum value of colormap", )
    parser.add_argument( "--vcenter", "-vcenter", type=float, default=None, 
                        help="Center value of colormap", )
    
    # Glass brain args
    parser.add_argument( "--glass_brain_path", "-glass", nargs="+", type=str, default=None, 
                        help="Glass brain to plot, specify file paths, or default_glass, or default pial", )

    # Plot option args
    parser.add_argument( "--auto_crop", "-crop", action="store_true", 
                        help="Automatically crop image when saving", )
    parser.add_argument( "--per_bundle_color", "-per_bundle_color", nargs="*", type=str, default=None, 
                        help="List of color names (available in mpl) to use for each bundle", )
    parser.add_argument( "--show_bundle", "-show", action="store_true", 
                        help="Show the interactive window", )
    parser.add_argument( "--as_points", "-points", action="store_true", 
                        help="Plots the bundle using points instead of streamtubes.", )
    parser.add_argument( "--linewidth", "-linewidth", type=float, default=4, 
                        help="Thickness of streamlines", )
    parser.add_argument( "--opacity", "-a", type=float, default=1, 
                        help="Opacity", )
    parser.add_argument( "--fake_tube", "-fake_tube", action="store_true", default=True, 
                        help="Show streamline as fake tube", )
    args = parser.parse_args()
    run(args)
