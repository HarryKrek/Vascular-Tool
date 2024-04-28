# TODO clean up whatever tf is going on with my imports, this is getting excessive
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes,
    isotropic_dilation,
    isotropic_erosion,
    disk,
)
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_local, median
from skimage.exposure import equalize_adapthist, rescale_intensity
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import pixel_graph
from alive_progress import alive_bar
import pandas as pd
from pathlib import Path
import skan.csr
import argparse
import sknw
import networkx as nx
import skimage
import kmeans1d
import plantcv.plantcv as pcv
import os
from multiprocessing import Pool, cpu_count, set_start_method
import yaml


def find_images_in_path(pathdir):
    path = Path(pathdir)
    images = list(path.glob("*.tif"))
    print(f"{len(images)} images found in directory")
    images.sort()
    return images


# def get_global_threshold(images):
#     #preallocate zeros for all the bins
#     histoTotal = np.zeros(255)
#     binVals = np.linspace(0,255,num=256)
#     for i,imageLoc in enumerate(images):
#         if i % 10 == 0:
#             img = skimage.io.imread(imageLoc)
#             imgGrey = img[:,:,1] #Take the green channel
#             imgAdapted1 = rescale_intensity(imgGrey)
#             imgAdapted = (skimage.exposure.equalize_adapthist(imgAdapted1,nbins=256)*255).astype(np.uint8)

#             #get the values in bin form
#             imgArray = imgAdapted.ravel()
#             imageHistoN, _ = np.histogram(imgArray, bins=binVals)
#             histoTotal = histoTotal + imageHistoN
#     #Use k-means clustering to determine the global threshold
#     # yen_thresh = threshold_yen(histoTotal)
#     return None


def get_running_approval():
    pass


def import_and_blur_image(imgPath, config):
    sigma = config.get("Blur Sigma")
    img = io.imread(imgPath)
    imgGrey = img[:, :, 1]  # Take the green channel
    rescaled_grey = rescale_intensity(imgGrey)
    adapted_hist = equalize_adapthist(rescaled_grey)
    blurred = gaussian(
        adapted_hist, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1
    )
    return img, blurred


def segment_image(blurred):
    # Convert to uint8
    uint8Image = (blurred * 255).astype(np.uint8)
    thresh = threshold_local(uint8Image, block_size=101)
    # Apply segmentation threshold
    segmentation = uint8Image > thresh
    return segmentation


def remove_holes_and_small_items(segmentation, config):
    min_object_size = config.get("Min Object Size")
    min_hole_size = config.get("Min Hole Size")
    # erode a bit
    eroded = isotropic_erosion(segmentation, 1)
    ensmallend = remove_small_objects(eroded, min_size=min_object_size, connectivity=8)
    dilated = isotropic_dilation(ensmallend, 1)
    unholed = remove_small_holes(dilated, area_threshold=min_hole_size)
    return unholed


def create_skeleton(segmentation, config):
    line_min = config.get("Min Line Length")
    # Skeletonisation
    skel = pcv.morphology.skeletonize(segmentation)
    pruned_skeleton, img, objects = pcv.morphology.prune(skel_img=skel, size=line_min)
    return pruned_skeleton


def draw_and_save_images(image, segmentation, bp, ep, skel, name, config):
    save = config.get("Save Image")
    show = config.get("Show Image")
    if not save and not show:
        pass

    # Make an image
    masked = label2rgb(
        label(segmentation * 255),
        image=image,
        colors=["red"],
        kind="overlay",
        alpha=0.6,
        bg_label=0,
        bg_color=None,
        saturation=1,
    )
    # Mask skeleton
    maskedSkel = label2rgb(
        label(skel * 255),
        image=masked,
        colors=["white"],
        kind="overlay",
        alpha=1,
        bg_label=0,
        bg_color=None,
        saturation=1,
    )
    adjusted = (maskedSkel * 255).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(adjusted)
    plt.scatter(
        [point[1] for point in bp], [point[0] for point in bp], color="blue", s=2
    )
    plt.scatter(
        [point[1] for point in ep], [point[0] for point in ep], color="green", s=2
    )
    # plt.imshow(skel > 0, alpha=0.8)
    # Save image, not working too well at the moment+
    # skimage.io.imsave(name,adjusted)
    ax.axis("off")
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, transparent=True)
    if show:
        ax.set_title(name)
        plt.show()
    else:
        plt.close()


def obtain_branch_and_end_points(skel):
    branchPoints = []
    endPoints = []

    degree_img = skan.csr.make_degree_image(skel)
    endPntProp = regionprops(label((degree_img == 1) * 255))
    branchPntProp = regionprops(label((degree_img > 2) * 255))

    for pnt in endPntProp:
        x0, y0 = pnt.centroid
        endPoints.append([x0, y0])

    for pnt in branchPntProp:
        x0, y0 = pnt.centroid
        branchPoints.append([x0, y0])

    return (branchPoints, endPoints)


def vessel_statistics_from_graph(graph):
    totalLen = 0
    for s, e in graph.edges():
        ps = graph[s][e][0]["weight"]
        totalLen += ps

    return totalLen, totalLen / len(graph.edges())


def process_image_results(i, segmentation, graph, skel):
    try:
        branchPoints, endPoints = obtain_branch_and_end_points(skel)
        results = {}
        # Get number of cells
        results["Branch Points"] = len(branchPoints)
        results["End Points"] = len(endPoints)
        # Size of image
        imgSize = np.size(segmentation)
        # Area of Vessels
        results["vesselArea"] = np.count_nonzero(segmentation)
        # Percentage Area
        results["percentArea"] = results["vesselArea"] / imgSize * 100
        results["totalVesselLength"], results["avgVesselLength"] = (
            vessel_statistics_from_graph(graph)
        )
        results["Errors"] = ""
    except Exception as e:
        results["Errors"] = str(e)
    return results, branchPoints, endPoints


def save_results_to_csv(savename, data):
    df = pd.DataFrame(data)
    df.to_csv(savename)


def worker_process(args):
    try:
        i, image, resultsPath, config = args

        rgbimg, blurred = import_and_blur_image(image, config)
        segmentation = segment_image(blurred)
        cleaned_segmentation = remove_holes_and_small_items(segmentation, config)
        skel = create_skeleton(cleaned_segmentation, config)
        graph = sknw.build_sknw(skel, multi=True, iso=False, ring=True, full=True)
        img_results, branchPoints, endPoints = process_image_results(
            i, cleaned_segmentation, graph, skel
        )
        print(img_results)
        draw_and_save_images(
            rgbimg,
            cleaned_segmentation,
            branchPoints,
            endPoints,
            skel,
            os.path.abspath(resultsPath + str(i) + ".png"),
            config,
        )
        return img_results
    except Exception as e:
        print(f"EXCEPTION: {e}")


def main(path: str, savename: str, configPath: str):
    path = "F://20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE//Wellc8//F2"
    savename = ".\\24_04_well_c8_ASH_speckle_and_holes.csv"
    resultsPath = "./Results/"
    configPath = ".\\config.yml"
    with open(configPath, "r") as file:
        config = yaml.safe_load(file)

    images = find_images_in_path(path)
    results = []  # list of dictionaries
    args = [(i, image, resultsPath, config) for i, image in enumerate(images)]
    set_start_method("spawn")
    with Pool(cpu_count()) as p:
        results = p.map(worker_process, args)
        p.close()
    # print(savename,results)
    save_results_to_csv(savename, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vascular Tool for Microscopy Analysis"
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--path")
    parser.add_argument("-s", "--savename")
    args = parser.parse_args()
    main(args.path, args.savename, None)
