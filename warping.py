import cv2
import numpy as np
import easygui, os
import matplotlib.pyplot as plt
from PIL import Image
# 将img2变换到img1
#name1 = easygui.fileopenbox("Choose file to warp")
#name2 = easygui.fileopenbox("Choose background")

#print(repr(name1) + ', ' + str(os.path.isfile(name1)) + '\n' + repr(name2) + ', ' + str(os.path.isfile(name2)))

#img1 = cv2.imread(name1.replace('\\', '\\\\'))
#img2 = cv2.imread(name2.replace('\\', '\\\\'))
# img2 = cv2.imread(r"westermannsatlas00stie_0106.jpg")[..., ::-1].copy()
img2 = np.array(Image.open(easygui.fileopenbox("Choose file to warp")), dtype=np.uint8)[..., :3][..., ::-1].copy()
img1 = np.array(Image.open(easygui.fileopenbox("Choose background")), dtype=np.uint8)[..., :3][..., ::-1].copy()
print(img1.dtype)

# img1 = cv2.resize(img1, None, fx=0.1, fy=0.1)
# print(img1.shape)
# print(img2.shape)
h, w, _ = img1.shape
img2 = cv2.resize(img2, (w, h))

img1_show = img1.copy()
img2_show = img1.copy()

img_warpped = np.zeros_like(img1)

points1 = []
points2 = []

output = img1

def update_tps():
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(np.array(points1)[None].astype(np.int32), np.array(points2)[None].astype(np.int32), [cv2.DMatch(i, i, 0) for i in range(len(points1))])
    global img_warpped, output
    output = tps.warpImage(img2)
    img_warpped = (output * 0.5 + img1 * 0.5).astype(np.uint8)
    #cv2.imwrite("aligned.png", output)

def draw_points(img, points, color):
    for i, point in enumerate(points):
        cv2.putText(img, str(i), point, 0, 1, (255, 255, 255), 2)
        cv2.circle(img, point, 3, (0, 0, 255), -1)
def click_old(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        print("R click.")
        for i in range(len(points1)):
            point = points1[i]
            dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
            print(dist)
            if dist <= 4:
                del points1[i]
                break
    global img1_show
    img1_show = img1.copy()
    draw_points(img1_show, points1, (0, 255, 0))

    if len(points1) >= 4 and len(points2) == len(points1):
        update_tps()

def click_new(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        print("R click.")
        for i in range(len(points2)):
            point = points2[i]
            dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
            print(dist)
            if dist <= 5:
                del points2[i]
                break
    global img2_show
    img2_show = img2.copy()
    draw_points(img2_show, points2, (0, 0, 255))

    if len(points2) >= 4 and len(points2) == len(points1):
        update_tps()

cv2.imshow("old", img1_show)
cv2.imshow("new", img2_show)
cv2.setMouseCallback('old', click_old)
cv2.setMouseCallback('new', click_new)

while True:
    cv2.imshow("old", img1_show)
    cv2.imshow("new", img2_show)
    cv2.imshow("warped", img_warpped)
    cv2.waitKey(10)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.imwrite(easygui.filesavebox("Save as"), output)
        break
cv2.destroyAllWindows()