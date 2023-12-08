import cv2
import numpy as np

# Load images
result_img = cv2.imread('./Assets/result_with_parking_spaces.png')
shouldhave_img = cv2.imread('./Assets/shouldhave.png')

parking_spaces = [
    # front 3
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
    [(420, 224), (523, 260)],
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

    # Middle
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
    [(730, 470), (772, 550)],
    [(776, 470), (830, 550)],
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
    [(924, 160), (970, 260)],
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

# List to store parking space templates
templates = []

# Extracting content from each parking space in result_img
for space in parking_spaces:
    x1, y1 = space[0]
    x2, y2 = space[1]
    template = result_img[y1:y2, x1:x2]
    templates.append(template)


threshold = 0.92
occupied_spaces = []

# Extracting the region of interest
for i, space in enumerate(parking_spaces):
    x1, y1 = space[0]
    x2, y2 = space[1]
    roi = shouldhave_img[y1:y2, x1:x2]

    # Matching templates
    res = cv2.matchTemplate(roi, templates[i], cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # Checking if there is a match
    if loc[0].size == 0:
        occupied_spaces.append(i)


# Drawing rectangles on shouldhave_img
for i in range(len(parking_spaces)):
    color = (0, 0, 255) if i in occupied_spaces else (0, 255, 0)
    cv2.rectangle(shouldhave_img, parking_spaces[i][0], parking_spaces[i][1], color, 2)


for i in range(len(parking_spaces)):
    # calculating and displaying available free spots
    free_parking_spots = len(parking_spaces) - len(occupied_spaces)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 255, 0)

    cv2.putText(shouldhave_img, f'Free Spots: {free_parking_spots}', (490, 31), font, font_scale, font_color, font_thickness)


# Displaying the result image
cv2.imshow('Occupancy Detection', shouldhave_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
