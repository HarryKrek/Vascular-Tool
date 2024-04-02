import cv2
import skimage
import os
import matplotlib.pyplot as plt
import numpy as np

from vascular_tool import find_images_in_path

video_Output = "wellc6.avi"
originalPath = "F://20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE//Wellc2//F2"
resultsPath = './Results/'
size = (1920,1080)


#cV2 Video Writer
video = cv2.VideoWriter(video_Output, cv2.VideoWriter_fourcc(*'MJPG'), 2, size)

images = find_images_in_path(originalPath)

def figure_to_array(fig):
    #Function from stackOVerflow
    #https://stackoverflow.com/questions/72399929/how-can-i-write-video-using-matplotlib-plot-images-to-opencv-library
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


for i,imageLoc in enumerate(images):
    if i % 5 == 0:
        #Get image and result image
        img = skimage.io.imread(imageLoc)
        resultImgPath = os.path.abspath(resultsPath + str(i)+'.tiff')
        resultImg = skimage.io.imread(resultImgPath)

        #Generate the histogram
        #Convert to greyscale
        imgGrey = img[:,:,1] #Take the green channel
        hist, bins = np.histogram(imgGrey.flatten(), bins=256, range=[0,1])
    
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        dpi = fig.get_dpi()
        fig.set_size_inches(1920/float(dpi), 1080/float(dpi))
        
        ax1.imshow(img)
        ax1.set_axis_off()
        
        ax2.hist(imgGrey.ravel(), bins = 256, histtype='step', color='black')
        #Get the otsu and put on just for fun
        thresh= skimage.filters.threshold_otsu(imgGrey)
        ax2.axvline(thresh, color = 'r')

        ax3.imshow(resultImg)
        ax3.set_axis_off()

        fig.tight_layout()
        plt.axis('equal')
        # fig.show()
        plt.close()
        fig_arr = figure_to_array(fig)
        f_arr  = cv2.resize(fig_arr,(1920,1080))
        bgr = cv2.cvtColor(f_arr, cv2.COLOR_RGBA2BGR)
        video.write(bgr)
        print(f"{i} image completed")

video.release()
