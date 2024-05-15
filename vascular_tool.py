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
import networkx
from time import time


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
    if len(np.shape(img)) == 3:
        imgGrey = img[:, :, 1]  # Take the green channel
    else:
        imgGrey = img
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
    ensmallend = remove_small_objects(
        segmentation, min_size=min_object_size, connectivity=8
    )
    dilated = isotropic_dilation(ensmallend, 1)
    unholed = remove_small_holes(dilated, area_threshold=min_hole_size)
    return unholed


def vessel_width_and_prune(skel, segmentation, config: dict):
    # Compute Distance Image using SCIPI
    distance_img = distance_transform_edt(segmentation)
    # Use skel as a mask to get distance_skel:
    distance_skel = distance_img * (skel > 0)
    # Prune skeleton if below required width
    new_skel = 2 * distance_skel > config.get("Minimum Vessel Width")
    skel_width = new_skel * distance_img * 2
    return np.uint8(new_skel * 255), skel_width


def prune_skeleton_spurs_with_graph(
    graph: networkx.MultiGraph, config: dict
) -> networkx.MultiGraph:
    line_min = config.get("Min Spur Line Length")
    nodeRemovalCandidates = []

    edges = graph.edges
    for edgeLoc in edges:
        edge = edges[edgeLoc]
        if edge.get("weight") < line_min:
            # Check if nodes are on the edge
            for node in edgeLoc:
                neighbours = sum(1 for _ in graph.neighbors(node))
                if neighbours == 1:
                    nodeRemovalCandidates.append(node)

    # Remove edges from the graph
    graph.remove_nodes_from(nodeRemovalCandidates)
    return graph


def generate_skeleton_from_graph(shape: tuple, graph: networkx.MultiGraph):
    skel = np.zeros(shape)
    # Plot every point on the graph on the new skeleton
    for u, v, l in graph.edges:
        ps = graph[u][v][l].get("pts")
        X = ps[:, 1]
        Y = ps[:, 0]
        for i in range(len(X)):
            skel[Y[i], X[i]] = 255

    return skel


def create_skeleton(segmentation, config):
    # Skeletonisation
    skel = pcv.morphology.skeletonize(segmentation)
    skelWidthPruned, skel_width = vessel_width_and_prune(skel, segmentation, config)
    graph = sknw.build_sknw(
        skelWidthPruned, multi=True, iso=False, ring=True, full=True
    )
    graph = prune_skeleton_spurs_with_graph(graph, config)
    skelLengthPruned = generate_skeleton_from_graph(np.shape(skel), graph)
    return skelLengthPruned, skel_width, graph


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
        label(isotropic_dilation(skel * 255, 3)),
        image=masked,
        colors=["yellow"],
        kind="overlay",
        alpha=1,
        bg_label=0,
        bg_color=None,
        saturation=1,
    )
    adjusted = (maskedSkel * 255).astype(np.uint8)
    fig, ax = plt.subplots()
    fig.set_dpi(100)
    fig.set_size_inches(16, 16)

    ax.imshow(adjusted)
    plt.scatter(
        [point[1] for point in bp], [point[0] for point in bp], color="blue", s=10
    )
    plt.scatter(
        [point[1] for point in ep], [point[0] for point in ep], color="green", s=10
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


def obtain_branch_and_end_points(graph: networkx.MultiGraph):
    branchPoints = []
    endPoints = []

    nodes = graph.nodes
    nodePs = np.array([nodes[i]["o"] for i in nodes])
    for i, node in enumerate(nodes):
        neighbours = sum(1 for _ in graph.neighbors(node))
        if neighbours == 1:
            # End Point
            endPoints.append([nodePs[i, 0], nodePs[i, 1]])

        elif neighbours > 2:
            # Branch Point
            branchPoints.append([nodePs[i, 0], nodePs[i, 1]])

    return (branchPoints, endPoints)


def vessel_statistics_from_graph(graph: networkx.MultiGraph):
    totalLen = 0
    for s, e in graph.edges():
        ps = graph[s][e][0]["weight"]
        totalLen += ps

    return totalLen, totalLen / len(graph.edges())


def process_image_results(i, segmentation, graph, skel):
    try:
        results = {}
        results["Num"] = i
        branchPoints, endPoints = obtain_branch_and_end_points(graph)
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
        eroded = isotropic_erosion(segmentation, 1)
        cleaned_segmentation = remove_holes_and_small_items(eroded, config)
        skel, width_im = create_skeleton(cleaned_segmentation, config)
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
    startTime = time()
    configPath = ".\\config.yml"
    with open(configPath, "r") as file:
        config = yaml.safe_load(file)

    path = config.get("Img Path")
    savename = config.get("Spreadsheet Output Path")
    resultsPath = config.get("Results Path")

    images = find_images_in_path(path)
    results = []  # list of dictionaries
    args = [
        (i, image, resultsPath, config) for i, image in enumerate(images) if i % 10 == 0
    ]
    set_start_method("spawn")
    with Pool(cpu_count()) as p:
        results = p.map(worker_process, args)
        p.close()
    # print(savename,results)
    save_results_to_csv(savename, results)
    elapsed = time() - startTime
    print(f"Completed Processing in {elapsed} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vascular Tool for Microscopy Analysis"
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--path")
    parser.add_argument("-s", "--savename")
    args = parser.parse_args()
    main(args.path, args.savename, None)
