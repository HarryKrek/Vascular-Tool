#TODO clean up whatever tf is going on with my imports, this is getting excessive
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, isotropic_dilation
from skimage.filters import gaussian, threshold_local
import numpy as np
from scipy.ndimage import distance_transform_edt
from alive_progress import alive_bar
import pandas as pd
from pathlib import Path
import skan.csr
import argparse
import sknw
import networkx as nx
import tifffile
from skimage.exposure import adjust_gamma
import plantcv.plantcv as pcv


def find_images_in_path(pathdir):
    path = Path(pathdir)
    images = list(path.glob('*.tif'))
    print(f"{len(images)} images found in directory")
    images.sort()
    return images


def get_running_approval():
    pass

def import_and_blur_image(imgPath, sigma = 2.5):
    img = io.imread(imgPath)
    imgGrey = img[:,:,1] #Take the green channel
    adapted_hist = adjust_gamma(imgGrey)
    blurred = gaussian(adapted_hist, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    return img, blurred

def segment_image(blurred):
    thresh = threshold_local(blurred , block_size = 301)
    #Apply segmentation threshold
    segmentation = blurred > thresh
    return segmentation

def remove_holes_and_small_items(segmentation, min_object_size = 1000, min_hole_size = 300):
    ensmallend = remove_small_objects(segmentation, min_size = min_object_size, connectivity=8)
    unholed = remove_small_holes(ensmallend, area_threshold = min_hole_size)
    return unholed

def create_skeleton(segmentation, line_min = 30):
    #Skeletonisation
    skel = pcv.morphology.skeletonize(segmentation)
    pruned_skeleton, img, objects = pcv.morphology.prune(skel_img=skel, size=line_min)
    return pruned_skeleton

def draw_and_save_images(image, segmentation,name):
    #Make an image
    masked = label2rgb(segmentation,image=image, colors = ['red'], alpha=0.5, saturation = 1)
    #Save image, not working too well at the moment
    tifffile.imsave(name, masked)


def obtain_branch_and_end_points(graph):
    bp = []
    ep = []

    nodes = graph.nodes()
    for node in nodes:
        if graph.degree[node] == 1:
            ep.append(node)
        elif graph.degree[node] > 2:
            bp.append(node)
    branchPoints = np.array([nodes[i]['o'] for i in bp])
    endPoints = np.array([nodes[i]['o'] for i in ep])

    return (branchPoints, endPoints)


def vessel_statistics_from_graph(graph):
    totalLen = 0
    for (s,e) in graph.edges():
        ps = graph[s][e]['weight']
        totalLen += ps

    return totalLen, totalLen/len(graph.edges())

def process_image_results(segmentation, graph):
    try:
        branchPoints, endPoints = obtain_branch_and_end_points(graph)
        results = {}
        # Get number of cells
        results["Branch Points"] = len(branchPoints)
        results["End Points"] = len(endPoints)
        # Size of image
        imgSize = np.size(segmentation)
        # Area of Vessels
        results["vesselArea"] = np.count_nonzero(segmentation)
        # Percentage Area
        results["percentArea"] = results["vesselArea"]/imgSize * 100
        results["totalVesselLength"], results["avgVesselLength"] = vessel_statistics_from_graph(graph)
        results["Errors"] = ""
    except Exception as e:
        results["Errors"] = str(e)
    return results

def save_results_to_csv(savename,data):
    df=pd.DataFrame(data)
    df.to_csv(savename)


def main(path: str, savename: str):
    path = "F://F2//"
    savename = ".\\test.csv"
    resultsPath = '.\\Results\\'
    images = find_images_in_path(path)
    results = [] #list of dictionaries
    try:
        with alive_bar(len(images)) as bar:
            for i, image in enumerate(images):
                rgbimg, blurred = import_and_blur_image(image)
                segmentation = segment_image(blurred)
                cleaned_segmentation = remove_holes_and_small_items(segmentation)
                skel = create_skeleton(cleaned_segmentation)
                graph = sknw.build_sknw(skel)
                img_results = process_image_results(segmentation, graph)
                print(img_results)
                results.append(img_results)
                #draw_and_save_images((rgbimg * 255).astype(np.uint8),
                #    segmentation, resultsPath + str(i)+'.tiff')
                bar()
    except Exception as e:
        print(e)

    save_results_to_csv(savename, results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Vascular Tool for Microscopy Analysis')
    parser.add_argument('-c', '--config')
    parser.add_argument('-p', '--path')
    parser.add_argument('-s', '--savename')
    args = parser.parse_args()
    main(args.path, args.savename)