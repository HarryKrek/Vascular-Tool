from glob import glob
import os
import sknw
import yaml
import matplotlib.pyplot as plt

from vascular_tool import (
    import_and_blur_image,
    segment_image,
    remove_holes_and_small_items,
    create_skeleton,
    draw_and_save_images,
    process_image_results,
)
from skimage.morphology import isotropic_erosion, remove_small_objects

# Setup Variables
image = glob("C:\\Users\\uqhkrek\\Desktop\\reference image\\diaphragm_60x_3.tif")[0]
configLoc = glob("C:\\Users\\uqhkrek\\Desktop\\reference image\\config.yml")[0]
saveLoc = glob("C:\\Users\\uqhkrek\\Desktop\\reference image")[0]

print(image, configLoc, saveLoc)

with open(configLoc, "r") as file:
    config = yaml.safe_load(file)

rgbimg, blurred = import_and_blur_image(image, config)
segmentation = segment_image(blurred)
eroded = isotropic_erosion(segmentation, 1)
cleaned_segmentation = remove_holes_and_small_items(eroded, config)
skel, width_im, graph = create_skeleton(cleaned_segmentation, config)
img_results, branchPoints, endPoints = process_image_results(
    0, cleaned_segmentation, graph, skel
)

print(img_results)
draw_and_save_images(
    rgbimg,
    cleaned_segmentation,
    branchPoints,
    endPoints,
    skel,
    os.path.abspath(f"{saveLoc}\\final.png"),
    config,
)
fig, ax = plt.subplots()
fig.set_dpi(100)
fig.set_size_inches(16, 16)

ax.imshow(cleaned_segmentation)
ax.axis("off")
plt.savefig(
    f"{saveLoc}\\segmentation.png", bbox_inches="tight", pad_inches=0, transparent=True
)
