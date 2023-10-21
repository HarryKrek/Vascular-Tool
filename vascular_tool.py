import matplotlib.pyplot as plt
from skimage import data, io
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, isotropic_dilation
from skimage.filters import gaussian, threshold_local
import numpy as np
from dsepruning import skel_pruning_DSE
from scipy.ndimage import distance_transform_edt
from alive_progress import alive_bar
import pandas as pd
from pathlib import Path
import skan.csr
import argparse

def find_images_in_path(pathdir):
    path = Path(pathdir)
    images = list(path.glob('*.tif'))
    print(f"{len(images)} images found in directory")
    images.sort()
    return images

def get_running_approval():
    pass

def import_and_blur_image(imgPath, sigma = 3.5):
    img = io.imread(imgPath)
    imgGrey = rgb2gray(img)
    blurred = gaussian(imgGrey, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    return blurred

def segment_image(blurred):
    thresh = threshold_local(blurred , block_size = 301) 
    segmentation = blurred > thresh
    return segmentation

def remove_holes_and_small_items(segmentation, min_object_size = 100, min_hole_size = 100):
    ensmallend = remove_small_objects(segmentation, min_size = 160, connectivity=8)
    unholed = remove_small_holes(ensmallend, area_threshold = 100)
    return unholed

def create_skeleton(segmentation, area_min = 100):
    skel = skeletonize(segmentation)
    skel_dist = distance_transform_edt(segmentation,return_indices=False, return_distances=True)
    pruned_skel = skel_pruning_DSE(skel,skel_dist, area_min)
    return pruned_skel

def find_branch_and_end_points(skel):
    degree = skan.csr.make_degree_image(skel) 
    branches = degree > 2
    ends = np.nonzero(degree == 1)
    branch_pts = np.nonzero(branches)
    end_pts = np.nonzero(ends)
    return (branch_pts, end_pts)


def process_image_results(segmentation, skeleton, branch_pts, end_pts):
    results = {}
    # Get number of cells
    results["numPoints"] = len(branch_pts[0])
    # Size of image
    imgSize = np.size(segmentation)
    # Area of Vessels
    results["vesselArea"] = np.count_nonzero(segmentation)
    # Percentage Area
    results["percentArea"] = results["vesselArea"]/imgSize * 100
    results["vesselLength"] = np.count_nonzero(skeleton)
    return results
    
def save_results_to_csv(savename,data):
    df=pd.DataFrame(data)
    df.to_csv(savename)


def main(path: str, savename: str):
    path = "\\\\shares01.rdm.uq.edu.au\\HKUG2023-A10939\\20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE\\Wellc2\\F2"
    savename = 'test.csv'
    images = find_images_in_path(path)
    results = [] #list of dictionaries
    with alive_bar(len(images)) as bar:
        for image in images:
            blurred = import_and_blur_image(image)
            segmentation = segment_image(blurred)
            cleaned_segmentation = remove_holes_and_small_items(segmentation)
            skel = create_skeleton(cleaned_segmentation)
            branch_points, end_points = find_branch_and_end_points(skel)
            img_results = process_image_results(cleaned_segmentation, skel, 
                                                branch_points, end_points)
            print(img_results)
            results.append(img_results)
            bar()
    save_results_to_csv(savename, results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Vascular Tool for Microscopy Analysis')
    parser.add_argument('-c', '--config')
    parser.add_argument('-p', '--path')
    parser.add_argument('-s', '--savename')
    args = parser.parse_args()
    main(args.path, args.savename)