"""'
A TOOL TO ASSESS THE SEGMENTATION PERFORMACNE OF VASCULAR TOOL
SET VASCULAR TOOL TO ONLY SAVE THE SEGMENTATION -> NO BRANCH OR END POINTS
"""

import glob
import pandas as pd
import skimage
import numpy as np
from pathlib import Path


VascularImagePath = "./Results/*.tif"
TruthImagePath = "C:/Users/uqhkrek/Downloads/REAVER_Vascular_Networks_Image_Dataset/ImageJ_Auto/*.tif"

# Import images
MyImages = glob.glob(VascularImagePath)
truthImages = glob.glob(TruthImagePath)

# Do a quick check to make sure that both have the same number of images
print(f"{len(MyImages)} mine, {len(truthImages)} ground truth")
if len(MyImages) != len(truthImages):
    raise ValueError("Incorrect number of Images")

metrics = []
# Go through each image
for i in range(len(MyImages)):
    # Import images
    myImg = skimage.io.imread(MyImages[i], as_gray=True)
    truthImg = skimage.io.imread(truthImages[i], as_gray=True)
    # Convert both images to Binary
    myImgBin = myImg > 0
    truthImgBin = truthImg > 0
    myImgNegated = np.logical_not(myImgBin)
    truthImgNegated = np.logical_not(truthImgBin)
    # Metrics Generation
    truePositive = np.logical_and(myImgBin, truthImgBin)
    trueNegative = np.logical_and(myImgNegated, truthImgNegated)
    falsePositive = np.logical_and(myImgBin, truthImgNegated)
    falseNegative = np.logical_and(myImgNegated, truthImgBin)
    # Count
    TP = np.count_nonzero(truePositive)
    TN = np.count_nonzero(trueNegative)
    FP = np.count_nonzero(falsePositive)
    FN = np.count_nonzero(falseNegative)
    # Accuracy, sensitivity, specificity
    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / (TP + FP)
    spec = TN / (TN + FP)
    # Metrics into dict
    result = {
        "name": Path(MyImages[i]).stem,
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
    }
    print(result)
    metrics.append(result)
avgAcc = np.mean([r["accuracy"] for r in metrics])
avgSens = np.mean([r["sensitivity"] for r in metrics])
avgSpec = np.mean([r["specificity"] for r in metrics])
print(f"Mean Acc {avgAcc}, Mean Sens {avgSens}, Mean Spec = {avgSpec}")


# Convert metrics to pd Dataframe then csv
df = pd.DataFrame.from_dict(metrics)
df.to_csv("./SegmentationAsssessment.csv")
