import cv2
import skimage
import os
import matplotlib.pyplot as plt
import numpy as np
import kmeans1d

from vascular_tool import find_images_in_path

video_Output = "wellc8.avi"
originalPath = "F://20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE//Wellc8//F2"
resultsPath = "./Results/"
size = (1000, 1000)


def figure_to_array(fig):
    # Function from stackOVerflow
    # https://stackoverflow.com/questions/72399929/how-can-i-write-video-using-matplotlib-plot-images-to-opencv-library
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


# cV2 Video Writer
video = cv2.VideoWriter(video_Output, cv2.VideoWriter_fourcc(*"MJPG"), 60, size)

images = find_images_in_path(originalPath)


for i, imageLoc in enumerate(images):
    # Get image and result image
    img = skimage.io.imread(imageLoc)
    resultImgPath = os.path.abspath(resultsPath + str(i) + ".png")
    resultImg = skimage.io.imread(resultImgPath)

    # Generate the histogram
    # Convert to greyscale
    imgGrey = img[:, :, 1]  # Take the green channel
    imgGrey = skimage.exposure.equalize_hist(imgGrey, nbins=256)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    dpi = 100
    fig.set_dpi(dpi)
    fig.set_size_inches(600 / float(dpi), 600 / float(dpi))

    ax1.imshow(img)
    ax1.set_axis_off()

    # ax2.hist(imgGrey.ravel(), bins = 256, histtype='step', color='black')
    # #Get the otsu and put on just for fun
    # ax2.axvline(thresh, color = 'r')

    ax2.imshow(resultImg)
    ax2.set_axis_off()

    fig.tight_layout()
    plt.axis("equal")
    # fig.show()
    plt.close()
    fig_arr = figure_to_array(fig)
    f_arr = cv2.resize(fig_arr, (1000, 1000))
    bgr = cv2.cvtColor(f_arr, cv2.COLOR_RGBA2BGR)
    video.write(bgr)
    print(f"{i} image completed")

video.release()
