import numpy as np
from skimage.measure import approximate_polygon, find_contours

import cv2

img = cv2.imread('loadtest.png', 0)
contours = find_contours(img, 0)

result_contour = np.zeros(img.shape + (3, ), np.uint8)
result_polygon1 = np.zeros(img.shape + (3, ), np.uint8)
result_polygon2 = np.zeros(img.shape + (3, ), np.uint8)

for contour in contours:
    print('Contour shape:', contour.shape)

    # reduce the number of lines by approximating polygons
    polygon1 = approximate_polygon(contour, tolerance=2.5)
    print('Polygon 1 shape:', polygon1.shape)

    # increase tolerance to further reduce number of lines
    polygon2 = approximate_polygon(contour, tolerance=15)
    print('Polygon 2 shape:', polygon2.shape)

    contour = contour.astype(np.int).tolist()
    polygon1 = polygon1.astype(np.int).tolist()
    polygon2 = polygon2.astype(np.int).tolist()

    # draw contour lines
    for idx, coords in enumerate(contour[:-1]):
        y1, x1, y2, x2 = coords + contour[idx + 1]
        result_contour = cv2.line(result_contour, (x1, y1), (x2, y2),
                                  (0, 255, 0), 1)
    # draw polygon 1 lines
    for idx, coords in enumerate(polygon1[:-1]):
        y1, x1, y2, x2 = coords + polygon1[idx + 1]
        result_polygon1 = cv2.line(result_polygon1, (x1, y1), (x2, y2),
                                   (0, 255, 0), 1)
    # draw polygon 2 lines
    for idx, coords in enumerate(polygon2[:-1]):
        y1, x1, y2, x2 = coords + polygon2[idx + 1]
        result_polygon2 = cv2.line(result_polygon2, (x1, y1), (x2, y2),
                                   (0, 255, 0), 1)

cv2.imwrite('contour_lines.png', result_contour)
cv2.imwrite('polygon1_lines.png', result_polygon1)
cv2.imwrite('polygon2_lines.png', result_polygon2)
