import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Loading the image
image = cv2.imread("Assets/screen.PNG")

# Creating empty image to draw parking spaces
parking_spaces_image = image.copy()


parking_spaces = [
    #front 3
    [(30, 380), (80, 470)],
    [(80, 380), (130, 470)],
    [(130, 380), (180, 470)],

    # scattered in the front
    [(0, 160), (50, 250)],
    [(50, 160), (100, 250)],
    [(310, 110), (350, 190)],
    [(370, 30), (410, 110)],

    # the first two
    [(161, 157), (209, 245)],
    [(210, 158), (258, 246)],

    #  next 7
    [(231, 442), (335, 488)],
    [(231, 488), (335, 533)],
    [(231, 533), (335, 577)],
    [(231, 577), (335, 621)],
    [(231, 621), (335, 665)],
    [(231, 665), (335, 709)],
    [(231, 709), (335, 753)],

    # Tribune 15
    [(1227, 629), (1272, 717)],
    [(549, 629), (593, 717)],
    [(593, 629), (654, 717)],
    [(657, 629), (701, 717)],
    [(701, 629), (745, 717)],
    [(745, 629), (789, 717)],
    [(789, 629), (833, 717)],
    [(833, 629), (875, 717)],
    [(875, 629), (922, 717)],
    [(922, 629), (962, 717)],
    [(962, 629), (1006, 717)],
    [(1006, 629), (1050, 717)],
    [(1050, 629), (1094, 717)],
    [(1094, 629), (1138, 717)],
    [(1138, 629), (1182, 717)],
    [(1182, 629), (1222, 717)],
    # Enterance
    [(420, 134), (523, 179)],
    [(420, 179), (523, 224)],
    [(420, 224), (523, 269)],
    # Gate
    [(415, 310), (522, 355)],
    [(415, 355), (522, 400)],
    [(415, 400), (522, 445)],
    [(415, 445), (522, 490)],
    [(415, 490), (522, 531)],
    [(415, 531), (525, 575)],
    [(415, 575), (525, 619)],
    [(415, 619), (525, 663)],
    [(415, 663), (525, 708)],
    [(415, 708), (525, 753)],
    [(415, 753), (525, 798)],

    #Middle
    [(540, 360), (590, 450)],
    [(590, 360), (640, 450)],
    [(640, 360), (690, 450)],
    [(690, 360), (740, 450)],
    [(740, 360), (790, 450)],
    [(790, 360), (840, 450)],
    [(840, 360), (890, 450)],
    [(890, 360), (940, 450)],
    [(940, 360), (990, 450)],
    [(990, 360), (1040, 450)],
    [(1040, 360), (1090, 450)],
    [(1090, 360), (1140, 450)],

    # inner
    [(580, 470), (630, 550)],
    [(630, 470), (680, 550)],
    [(680, 470), (730, 550)],
    [(730, 470), (780, 550)],
    [(780, 470), (830, 550)],
    [(830, 470), (880, 550)],
    [(880, 470), (930, 550)],
    [(930, 470), (980, 550)],
    [(980, 470), (1030, 550)],
    [(1030, 470), (1080, 550)],
    [(1080, 470), (1130, 550)],

    # upper
    [(525, 160), (570, 260)],
    [(570, 160), (620, 260)],
    [(620, 160), (670, 260)],
    [(670, 160), (720, 260)],
    [(720, 160), (770, 260)],
    [(770, 160), (820, 260)],
    [(820, 160), (870, 260)],
    [(870, 160), (920, 260)],
    [(920, 160), (970, 260)],
    [(970, 160), (1020, 260)],
    [(1020, 160), (1070, 260)],
    [(1070, 160), (1130, 260)],
    [(1130, 160), (1190, 260)],
    [(1190, 160), (1250, 260)],
    [(1250, 160), (1310, 260)],
    [(1310, 160), (1370, 260)],
    [(1370, 160), (1430, 260)],
    [(1430, 160), (1490, 260)],

        # near the goal
    [(1270, 420), (1360, 460)],
    [(1270, 460), (1360, 500)],
    [(1270, 500), (1360, 540)],

]

# Drawing parking spaces on the image
for space in parking_spaces:
    cv2.rectangle(parking_spaces_image, space[0], space[1], (255, 255, 255), 2)

# Displaying the result
# cv2.imshow("Parking Spaces", parking_spaces_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the result
# cv2.imwrite("./Assets/result_with_parking_spaces.png", parking_spaces_image)
# ___________________________________________________________________________________________
def featureMatchingHomography():
    root = os.getcwd()
    img1path = os.path.join(root, './Assets/result_with_parking_spaces.png')
    img2path = os.path.join(root, './Assets/frame.png')
    img1 = cv2.imread(img1path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    nNeighbors = 2
    matches = flann.knnMatch(descriptor1, descriptor2, k=nNeighbors)

    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    minGoodMatches = 100

    if len(goodMatches) > minGoodMatches:
        # reshape(-1,1,2) -> reshapes to (nKeypoints, 1, 2)
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        errorThreshold = 5
        M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, errorThreshold)
        matchesMask = mask.ravel().tolist()
        M_inverse = np.linalg.inv(M)

        # Apply the perspective transformation to img2
        warped_img2 = cv2.warpPerspective(img2, M_inverse, (img1.shape[1], img1.shape[0]))

        # Combine the overlayed image (img1) and the transformed img2
        result_image = cv2.addWeighted(img1, 0.5, warped_img2, 0.5, 0)

        h, w = img2.shape
        imgBorder = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        warpedImgBorder = cv2.perspectiveTransform(imgBorder, M)
        img1 = cv2.polylines(img1, [np.int32(warpedImgBorder)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches")
        matchesMask = None

    green = (0, 255, 0)
    drawParams = dict(matchColor=green, singlePointColor=None, matchesMask=matchesMask,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, None, **drawParams)

    plt.figure()
    plt.imshow(result_image, 'gray')
    plt.show()

    cv2.imwrite("./Assets/shouldhave.png", result_image)

    return M, result_image


if __name__ == '__main__':
    M, result_image = featureMatchingHomography()

    M_inverse = np.linalg.inv(M)

    # Function to apply inverse homography to a list of points
    def apply_inverse_homography(points, inverse_matrix):
        return [cv2.perspectiveTransform(np.float32([pt]), inverse_matrix)[0][0] for pt in points]
# ___________________________________________________________________________________________
# Printing coordinates of parking spaces
for i, space in enumerate(parking_spaces, 1):
    print(f"Parking Space {i} Coordinates: {space}")
