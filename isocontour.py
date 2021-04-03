import snowy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

from skimage.measure import approximate_polygon, find_contours

import cv2

square = snowy.reshape(snowy.load('oval.png')[:,:,0])
sdf = snowy.unitize(snowy.generate_sdf(square != 0.0))

max_dist = 1
num_levels = 10

step = max_dist/num_levels

levels = np.ones((sdf.shape[0],sdf.shape[1]))
plt.figure()

for k in range(num_levels):
    levels = np.ones((sdf.shape[0],sdf.shape[1]))
    print("level ", k)
    for i in range(sdf.shape[0]):
        for j in range(sdf.shape[1]):
            if sdf[i,j,0] < step*k:
                levels[i,j] = 0
    cv2.imwrite('contours.png', levels*255)
    img = cv2.imread('contours.png', 0)

    # the '[:-1]' is used to skip the contour at the outer border of the image
    contours = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0][:-1]


    for contour in contours:
        x = contour[:,0,0]
        x = np.append(x, contour[0,0,0])
        y = contour[:,0,1]
        y = np.append(y, contour[0,0,1])
        plt.plot(x,y)

plt.show()
