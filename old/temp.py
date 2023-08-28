#from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.color import rgb2gray
from skimage.morphology import skeletonize
from skan.pre import threshold
import numpy as np
#from skimage.transform import 

PATH = "//shares01.rdm.uq.edu.au/HKUG2023-A10939/20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE/WellC6/F2/MyExperiment_Wellc6_F2_000885.tif"


img = io.imread(PATH)
#TODO check data type and fix
imgGrey = rgb2gray(img)


fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])


ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')



smooth_radius = 3  # float OK
threshold_radius = int(np.ceil(10))
binary = threshold(imgGrey, sigma=smooth_radius,radius=threshold_radius)
fig, ax = plt.subplots()
ax.imshow(binary)

skel = skeletonize(binary,method='lee')

plt.figure(2)
plt.imshow(img)
plt.figure(3)
plt.imshow(binary, cmap=plt.cm.gray)
plt.figure(4)
ax = plt.imshow(skel)



plt.show()
