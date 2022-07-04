import cv2 as cv
import numpy as np
import math
import os

bin_th = 70
h_d = 20
s_d = 100
v_d = 20
h_u = 30
s_u = 200
v_u = 255
c_d = 7
c_u = 25

kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)

average_moments1 = []
average_moments2 = []
average_moments3 = []
average_moments4 = []
average_moments5 = []


# create average moments list for every type of shape
def create_models():

    average_moments1.append(
        0.28955189)
    average_moments1.append(
        0.05704769)
    average_moments1.append(
        2.6946414e-05)
    average_moments1.append(
        6.46731589e-06)

    average_moments2.append(
        0.19821147)
    average_moments2.append(
        0.00283369)
    average_moments2.append(
        0.00289843)
    average_moments2.append(
        1.15707598e-05)

    average_moments3.append(
        0.20568786)
    average_moments3.append(
        0.01250939)
    average_moments3.append(
        4.00198852e-06)
    average_moments3.append(
        5.67530963e-07)

    average_moments4.append(
        0.21956682)
    average_moments4.append(
        0.01081098)
    average_moments4.append(
        0.00294242)
    average_moments4.append(
        0.00020091)

    average_moments5.append(
        0.1689841)
    average_moments5.append(
        0.00118559)
    average_moments5.append(
        7.12332656e-06)
    average_moments5.append(
        3.45635017e-07)


def detect(image):
    # read image
    img_color = cv.imread(image)
    img_color = cv.resize(img_color, (0, 0), fx=0.4, fy=0.4)
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

    # separate BGR spaces
    img_blue = img_color[:, :, 0]
    img_green = img_color[:, :, 1]
    img_red = img_color[:, :, 2]

    # median blur for later use
    blur_gray = cv.medianBlur(img_gray, 11)
    blur_blue = cv.medianBlur(img_blue, 11)
    blur_green = cv.medianBlur(img_green, 11)
    blur_red = cv.medianBlur(img_red, 11)

    # sobel edge detection
    # https://stackoverflow.com/questions/51167768/sobel-edge-detection-using-opencv
    grad_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    _, ths = cv.threshold(grad, 15, 255, cv.THRESH_BINARY)
    sobel_dilated = cv.dilate(ths, kernel3)
    sobel_eroded = cv.erode(sobel_dilated, kernel3)

    # binary thresholds on BGR color and yellow
    ret, binary_blue = cv.threshold(blur_blue, bin_th, 255, cv.THRESH_BINARY_INV)
    ret, binary_green = cv.threshold(blur_green, bin_th, 255, cv.THRESH_BINARY_INV)
    ret, binary_red = cv.threshold(blur_red, bin_th, 255, cv.THRESH_BINARY_INV)
    binary_yellow = cv.inRange(img_hsv, np.array([h_d, s_d, v_d]), np.array([h_u, s_u, v_u]))

    # adaptive binary thresholds on BGR space for edge detecion
    th_blue = cv.adaptiveThreshold(blur_blue, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 2)
    th_green = cv.adaptiveThreshold(blur_green, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 2)
    th_red = cv.adaptiveThreshold(blur_red, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 2)

    # canny edge detection with dilation
    canny = cv.Canny(blur_gray, c_d, c_u)
    canny = cv.dilate(canny, kernel3)

    # sum of all binary images
    bitwise1 = cv.bitwise_or(binary_blue, binary_green)
    bitwise2 = cv.bitwise_or(bitwise1, binary_red)
    bitwise3 = cv.bitwise_or(bitwise2, binary_yellow)

    bitwise_bg = cv.bitwise_or(th_blue, th_green)
    bitwise_bgr = cv.bitwise_or(bitwise_bg, th_red)
    bitwise_lines = cv.bitwise_or(bitwise_bgr, canny)
    bitwise_final = cv.bitwise_or(bitwise_lines, bitwise3)

    # morphology to get rid of artefacts and close up gaps
    img_eroded = cv.erode(bitwise_final, kernel3)
    img_dilated = cv.dilate(img_eroded, kernel3)

    bitwise_sobel = cv.bitwise_or(img_dilated, sobel_eroded)

    # contour detection and filled drawing in order to fully enclose shapes
    contours, hierarchy = cv.findContours(bitwise_sobel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(bitwise_sobel, contours, -1, (255, 255, 255), cv.FILLED)

    # final erosion and additional erosion with smaller shapes used for color detection
    final_erosion = cv.erode(bitwise_sobel, kernel9)
    color_erosion = cv.erode(final_erosion, kernel9)

    # final contour detection
    contours_final, hierarchy_final = cv.findContours(final_erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    copy = img_color.copy()

    # creating a mask for color detection
    masked = cv.bitwise_and(copy, copy, mask=color_erosion)
    first_type = {}
    first_max = 0
    second_type = {}
    second_max = 0
    third_type = {}
    third_max = 0
    fourth_type = {}
    fourth_max = 0
    fifth_type = {}
    fifth_max = 0

    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # checking every contour on the image
    for c in contours_final:

        # get area and all moments
        area = cv.contourArea(c)
        moments = cv.moments(c)
        # get hu_moments
        hu_moments = cv.HuMoments(moments)

        # calculate a difference to every shape model
        one_first_hu = math.sqrt(pow((hu_moments[0] - average_moments1[0]), 2))
        one_second_hu = math.sqrt(pow((hu_moments[1] - average_moments1[1]), 2))
        one_third_hu = math.sqrt(pow((hu_moments[2] - average_moments1[2]), 2))
        one_fourth_hu = math.sqrt(pow((hu_moments[3] - average_moments1[3]), 2))

        two_first_hu = math.sqrt(pow((hu_moments[0] - average_moments2[0]), 2))
        two_second_hu = math.sqrt(pow((hu_moments[1] - average_moments2[1]), 2))
        two_third_hu = math.sqrt(pow((hu_moments[2] - average_moments2[2]), 2))
        two_fourth_hu = math.sqrt(pow((hu_moments[3] - average_moments2[3]), 2))

        three_first_hu = math.sqrt(pow((hu_moments[0] - average_moments4[0]), 2))
        three_second_hu = math.sqrt(pow((hu_moments[1] - average_moments4[1]), 2))
        three_third_hu = math.sqrt(pow((hu_moments[2] - average_moments4[2]), 2))
        three_fourth_hu = math.sqrt(pow((hu_moments[3] - average_moments4[3]), 2))

        four_first_hu = math.sqrt(pow((hu_moments[0] - average_moments5[0]), 2))
        four_second_hu = math.sqrt(pow((hu_moments[1] - average_moments5[1]), 2))
        four_third_hu = math.sqrt(pow((hu_moments[2] - average_moments5[2]), 2))
        four_fourth_hu = math.sqrt(pow((hu_moments[3] - average_moments5[3]), 2))

        five_first_hu = math.sqrt(pow((hu_moments[0] - average_moments3[0]), 2))
        five_second_hu = math.sqrt(pow((hu_moments[1] - average_moments3[1]), 2))
        five_third_hu = math.sqrt(pow((hu_moments[2] - average_moments3[2]), 2))
        five_fourth_hu = math.sqrt(pow((hu_moments[3] - average_moments3[3]), 2))

        # check only shapes of area greater than 2000 to eliminate small objects
        # all numeric variables chosen to suit training images and leave some room for possible differences
        if area > 2000:
            # creating a dictionary of all shapes detected
            # additionally maximal sizes of each shape on current image are saved for comparison
            if one_first_hu < 0.062 and one_second_hu < 0.035 and one_third_hu < 0.0001 and one_fourth_hu < 0.0001 and area > 5000:
                if area > first_max:
                    first_max = area
                first_type[area] = c
            elif four_first_hu < 0.018 and four_second_hu < 0.006 and four_third_hu < 0.0004 and four_fourth_hu < 0.00001 and area > 4500:
                if area > fourth_max:
                    fourth_max = area
                fourth_type[area] = c
            elif two_first_hu < 0.019 and two_second_hu < 0.003 and two_third_hu < 0.0028 and two_fourth_hu < 0.0001 and area > 6000:
                if area > second_max:
                    second_max = area
                second_type[area] = c
            elif three_first_hu < 0.025 and three_second_hu < 0.01 and three_third_hu < 0.002 and three_fourth_hu < 0.0004 and area > 5000:
                if area > third_max:
                    third_max = area
                third_type[area] = c
            elif five_first_hu < 0.023 and five_second_hu < 0.02 and five_third_hu < 0.004 and five_fourth_hu < 0.00002 and area > 6000:
                if area > fifth_max:
                    fifth_max = area
                fifth_type[area] = c
            else:
                # drawing contours that do not fit any other class
                cv.drawContours(copy, c, -1, (0, 0, 0), 2)

    # choosing the most common class as reference class for more checks
    max_number = max([len(first_type), len(second_type), len(third_type), len(fourth_type), len(fifth_type)])
    reference_class = 0

    if max_number == len(first_type):
        reference_class = 1
    elif max_number == len(second_type):
        reference_class = 2
    elif max_number == len(third_type):
        reference_class = 3
    elif max_number == len(fourth_type):
        reference_class = 4
    elif max_number == len(fifth_type):
        reference_class = 5

    # checking every shape saved in first type dictionary
    # all other dictionaries are treated in a similar manner (only numerical constants change)
    for area in first_type:
        blue = False
        green = False
        red = False
        white = False
        yellow = False

        masked_copy = masked.copy()
        # creating a minimal rectangle of each shape and rotating it in order to use it correctly
        # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
        shape = (masked_copy.shape[1], masked_copy.shape[0])  # cv2.warpAffine expects shape in (length, height)

        rect = cv.minAreaRect(first_type[area])
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))
        if width < height:
            theta -= 90
            width, height = height, width

        matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
        masked_copy = cv.warpAffine(src=masked_copy, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        single_piece = masked_copy[y:y + int(height), x:x + int(width)]
        # final image of current shape is being converted to HSV to check its color
        single_piece = cv.cvtColor(single_piece, cv.COLOR_BGR2HSV)

        # if there is only one instance of this type its size is checked in relation to reference class
        unmatched_scale = False
        scaler = 0.75
        if len(first_type) == 1:
            if reference_class == 2:
                if area < (scaler * 1.23 * second_max):
                    unmatched_scale = True
            elif reference_class == 3:
                if area < (scaler * 0.9435 * third_max):
                    unmatched_scale = True
            elif reference_class == 4:
                if area < (scaler * 1.038 * fourth_max):
                    unmatched_scale = True
            elif reference_class == 5:
                if area < (scaler * 0.97 * fifth_max):
                    unmatched_scale = True

        # each shape has to be bigger than 70% of the biggest instance or be checked with reference class
        if area > 0.7*first_max and not unmatched_scale:
            # output of selected class is changed
            output[0] += 1

            # contour drawn on image
            cv.drawContours(copy, first_type[area], -1, (255, 0, 0), 2)
            copy = cv.putText(copy, 'Type 1', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2, cv.LINE_AA)

            # creating binary images of each possible color
            inrange_blue = cv.inRange(single_piece, np.array([100, 100, 100]), np.array([110, 255, 150]))
            inrange_green = cv.inRange(single_piece, np.array([70, 90, 60]), np.array([85, 255, 150]))
            inrange_red_1 = cv.inRange(single_piece, np.array([0, 120, 100]), np.array([10, 255, 160]))
            inrange_red_2 = cv.inRange(single_piece, np.array([170, 130, 120]), np.array([180, 255, 150]))
            inrange_white = cv.inRange(single_piece, np.array([0, 10, 155]), np.array([180, 40, 255]))
            inrange_yellow = cv.inRange(single_piece, np.array([20, 120, 140]), np.array([30, 255, 200]))
            inrange_red = cv.bitwise_or(inrange_red_1, inrange_red_2)

            # checking if binary images of colors are empty
            if inrange_blue.any() > 0:
                blue = True
            if inrange_green.any() > 0:
                green = True
            if inrange_red.any() > 0:
                red = True
            if inrange_white.any() > 0:
                white = True
            if inrange_yellow.any() > 0:
                yellow = True

            # final decisions of shape color
            # for output singular colors are needed, otherwise shape is "mixed")
            if blue and not green and not red and not white and not yellow:
                copy = cv.putText(copy, 'BLUE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 0, 0), 2, cv.LINE_AA)
                output[5] += 1
            elif not blue and green and not red and not white and not yellow:
                copy = cv.putText(copy, 'GREEN', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2, cv.LINE_AA)
                output[6] += 1
            elif not blue and not green and red and not white and not yellow:
                copy = cv.putText(copy, 'RED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 255), 2, cv.LINE_AA)
                output[7] += 1
            elif not blue and not green and not red and white and not yellow:
                copy = cv.putText(copy, 'WHITE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2, cv.LINE_AA)
                output[8] += 1
            elif not blue and not green and not red and not white and yellow:
                copy = cv.putText(copy, 'YELLOW', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 255), 2, cv.LINE_AA)
                output[9] += 1
            else:
                copy = cv.putText(copy, 'MIXED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 0), 2, cv.LINE_AA)
                output[10] += 1
        else:
            # shapes that do not match additional checks are marked on image
            cv.drawContours(copy, first_type[area], -1, (0, 0, 0), 10)
            copy = cv.putText(copy, 'NOPE', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv.LINE_AA)

    for area in second_type:
        blue = False
        green = False
        red = False
        white = False
        yellow = False

        masked_copy = masked.copy()
        # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
        shape = (masked_copy.shape[1], masked_copy.shape[0])  # cv2.warpAffine expects shape in (length, height)

        rect = cv.minAreaRect(second_type[area])
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))
        if width < height:
            theta -= 90
            width, height = height, width

        matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
        masked_copy = cv.warpAffine(src=masked_copy, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        single_piece = masked_copy[y:y + int(height), x:x + int(width)]
        single_piece = cv.cvtColor(single_piece, cv.COLOR_BGR2HSV)

        unmatched_scale = False
        scaler = 0.75
        if len(second_type) == 1:
            if reference_class == 1:
                if area < (scaler * 0.859 * first_max):
                    unmatched_scale = True
            elif reference_class == 3:
                if area < (scaler * 0.975 * third_max):
                    unmatched_scale = True
            elif reference_class == 4:
                if area < (scaler * 0.891 * fourth_max):
                    unmatched_scale = True
            elif reference_class == 5:
                if area < (scaler * 1.015 * fifth_max):
                    unmatched_scale = True

        if area > 0.7*second_max and not unmatched_scale:
            output[1] += 1
            cv.drawContours(copy, second_type[area], -1, (0, 255, 0), 2)
            copy = cv.putText(copy, 'Type 2', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2, cv.LINE_AA)

            inrange_blue = cv.inRange(single_piece, np.array([100, 100, 100]), np.array([110, 255, 150]))
            inrange_green = cv.inRange(single_piece, np.array([70, 90, 60]), np.array([85, 255, 150]))
            inrange_red_1 = cv.inRange(single_piece, np.array([0, 120, 100]), np.array([10, 255, 160]))
            inrange_red_2 = cv.inRange(single_piece, np.array([170, 130, 120]), np.array([180, 255, 150]))
            inrange_white = cv.inRange(single_piece, np.array([0, 10, 155]), np.array([180, 40, 255]))
            inrange_yellow = cv.inRange(single_piece, np.array([20, 120, 140]), np.array([30, 255, 200]))
            inrange_red = cv.bitwise_or(inrange_red_1, inrange_red_2)

            if inrange_blue.any() > 0:
                blue = True
            if inrange_green.any() > 0:
                green = True
            if inrange_red.any() > 0:
                red = True
            if inrange_white.any() > 0:
                white = True
            if inrange_yellow.any() > 0:
                yellow = True

            if blue and not green and not red and not white and not yellow:
                copy = cv.putText(copy, 'BLUE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 0, 0), 2, cv.LINE_AA)
                output[5] += 1
            elif not blue and green and not red and not white and not yellow:
                copy = cv.putText(copy, 'GREEN', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2, cv.LINE_AA)
                output[6] += 1
            elif not blue and not green and red and not white and not yellow:
                copy = cv.putText(copy, 'RED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 255), 2, cv.LINE_AA)
                output[7] += 1
            elif not blue and not green and not red and white and not yellow:
                copy = cv.putText(copy, 'WHITE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2, cv.LINE_AA)
                output[8] += 1
            elif not blue and not green and not red and not white and yellow:
                copy = cv.putText(copy, 'YELLOW', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 255), 2, cv.LINE_AA)
                output[9] += 1
            else:
                copy = cv.putText(copy, 'MIXED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 0), 2, cv.LINE_AA)
                output[10] += 1
        else:
            cv.drawContours(copy, second_type[area], -1, (0, 0, 0), 10)
            copy = cv.putText(copy, 'NOPE', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv.LINE_AA)
    for area in third_type:
        blue = False
        green = False
        red = False
        white = False
        yellow = False

        masked_copy = masked.copy()
        # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
        shape = (masked_copy.shape[1], masked_copy.shape[0])

        rect = cv.minAreaRect(third_type[area])
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))
        if width < height:
            theta -= 90
            width, height = height, width

        matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
        masked_copy = cv.warpAffine(src=masked_copy, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        single_piece = masked_copy[y:y + int(height), x:x + int(width)]
        single_piece = cv.cvtColor(single_piece, cv.COLOR_BGR2HSV)

        unmatched_scale = False
        scaler = 0.75
        if len(third_type) == 1:
            if reference_class == 1:
                if area < (scaler * 0.986 * first_max):
                    unmatched_scale = True
            elif reference_class == 2:
                if area < (scaler * 0.85 * second_max):
                    unmatched_scale = True
            elif reference_class == 4:
                if area < (scaler * 0.996 * fourth_max):
                    unmatched_scale = True
            elif reference_class == 5:
                if area < (scaler * 0.973 * fifth_max):
                    unmatched_scale = True

        if area > 0.7*third_max and not unmatched_scale:
            output[2] += 1
            cv.drawContours(copy, third_type[area], -1, (0, 0, 255), 2)
            copy = cv.putText(copy, 'Type 3', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2, cv.LINE_AA)

            inrange_blue = cv.inRange(single_piece, np.array([100, 100, 100]), np.array([110, 255, 150]))
            inrange_green = cv.inRange(single_piece, np.array([70, 90, 60]), np.array([85, 255, 150]))
            inrange_red_1 = cv.inRange(single_piece, np.array([0, 120, 100]), np.array([10, 255, 160]))
            inrange_red_2 = cv.inRange(single_piece, np.array([170, 130, 120]), np.array([180, 255, 150]))
            inrange_white = cv.inRange(single_piece, np.array([0, 10, 155]), np.array([180, 40, 255]))
            inrange_yellow = cv.inRange(single_piece, np.array([20, 120, 140]), np.array([30, 255, 200]))
            inrange_red = cv.bitwise_or(inrange_red_1, inrange_red_2)

            if inrange_blue.any() > 0:
                blue = True
            if inrange_green.any() > 0:
                green = True
            if inrange_red.any() > 0:
                red = True
            if inrange_white.any() > 0:
                white = True
            if inrange_yellow.any() > 0:
                yellow = True

            if blue and not green and not red and not white and not yellow:
                copy = cv.putText(copy, 'BLUE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 0, 0), 2, cv.LINE_AA)
                output[5] += 1
            elif not blue and green and not red and not white and not yellow:
                copy = cv.putText(copy, 'GREEN', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2, cv.LINE_AA)
                output[6] += 1
            elif not blue and not green and red and not white and not yellow:
                copy = cv.putText(copy, 'RED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 255), 2, cv.LINE_AA)
                output[7] += 1
            elif not blue and not green and not red and white and not yellow:
                copy = cv.putText(copy, 'WHITE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2, cv.LINE_AA)
                output[8] += 1
            elif not blue and not green and not red and not white and yellow:
                copy = cv.putText(copy, 'YELLOW', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 255), 2, cv.LINE_AA)
                output[9] += 1
            else:
                copy = cv.putText(copy, 'MIXED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 0), 2, cv.LINE_AA)
                output[10] += 1
        else:
            cv.drawContours(copy, third_type[area], -1, (0, 0, 0), 10)
            copy = cv.putText(copy, 'NOPE', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv.LINE_AA)
    for area in fourth_type:
        blue = False
        green = False
        red = False
        white = False
        yellow = False

        masked_copy = masked.copy()
        # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
        shape = (masked_copy.shape[1], masked_copy.shape[0])

        rect = cv.minAreaRect(fourth_type[area])
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))
        if width < height:
            theta -= 90
            width, height = height, width

        matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
        masked_copy = cv.warpAffine(src=masked_copy, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        single_piece = masked_copy[y:y + int(height), x:x + int(width)]
        single_piece = cv.cvtColor(single_piece, cv.COLOR_BGR2HSV)

        unmatched_scale = False
        scaler = 0.75
        if len(fourth_type) == 1:
            if reference_class == 1:
                if area < (scaler * 0.935 * first_max):
                    unmatched_scale = True
            elif reference_class == 2:
                if area < (scaler * 0.803 * second_max):
                    unmatched_scale = True
            elif reference_class == 3:
                if area < (scaler * 0.928 * third_max):
                    unmatched_scale = True
            elif reference_class == 5:
                if area < (scaler * 0.903 * fifth_max):
                    unmatched_scale = True

        if area > 0.7*fourth_max and not unmatched_scale:
            output[3] += 1
            cv.drawContours(copy, fourth_type[area], -1, (255, 0, 255), 2)
            copy = cv.putText(copy, 'Type 4', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2, cv.LINE_AA)

            inrange_blue = cv.inRange(single_piece, np.array([100, 100, 100]), np.array([110, 255, 150]))
            inrange_green = cv.inRange(single_piece, np.array([70, 90, 60]), np.array([85, 255, 150]))
            inrange_red_1 = cv.inRange(single_piece, np.array([0, 120, 100]), np.array([10, 255, 160]))
            inrange_red_2 = cv.inRange(single_piece, np.array([170, 130, 120]), np.array([180, 255, 150]))
            inrange_white = cv.inRange(single_piece, np.array([0, 10, 155]), np.array([180, 40, 255]))
            inrange_yellow = cv.inRange(single_piece, np.array([20, 120, 140]), np.array([30, 255, 200]))
            inrange_red = cv.bitwise_or(inrange_red_1, inrange_red_2)

            if inrange_blue.any() > 0:
                blue = True
            if inrange_green.any() > 0:
                green = True
            if inrange_red.any() > 0:
                red = True
            if inrange_white.any() > 0:
                white = True
            if inrange_yellow.any() > 0:
                yellow = True

            if blue and not green and not red and not white and not yellow:
                copy = cv.putText(copy, 'BLUE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 0, 0), 2, cv.LINE_AA)
                output[5] += 1
            elif not blue and green and not red and not white and not yellow:
                copy = cv.putText(copy, 'GREEN', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2, cv.LINE_AA)
                output[6] += 1
            elif not blue and not green and red and not white and not yellow:
                copy = cv.putText(copy, 'RED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 255), 2, cv.LINE_AA)
                output[7] += 1
            elif not blue and not green and not red and white and not yellow:
                copy = cv.putText(copy, 'WHITE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2, cv.LINE_AA)
                output[8] += 1
            elif not blue and not green and not red and not white and yellow:
                copy = cv.putText(copy, 'YELLOW', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 255), 2, cv.LINE_AA)
                output[9] += 1
            else:
                copy = cv.putText(copy, 'MIXED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 0), 2, cv.LINE_AA)
                output[10] += 1
        else:
            cv.drawContours(copy, fourth_type[area], -1, (0, 0, 0), 10)
            copy = cv.putText(copy, 'NOPE', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv.LINE_AA)
    for area in fifth_type:
        blue = False
        green = False
        red = False
        white = False
        yellow = False

        masked_copy = masked.copy()
        # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
        shape = (masked_copy.shape[1], masked_copy.shape[0])

        rect = cv.minAreaRect(fifth_type[area])
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))
        if width < height:
            theta -= 90
            width, height = height, width

        matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
        masked_copy = cv.warpAffine(src=masked_copy, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        single_piece = masked_copy[y:y + int(height), x:x + int(width)]
        single_piece = cv.cvtColor(single_piece, cv.COLOR_BGR2HSV)

        unmatched_scale = False
        scaler = 0.75
        if len(fifth_type) == 1:
            if reference_class == 1:
                if area < (scaler * 0.986 * first_max):
                    unmatched_scale = True
            elif reference_class == 2:
                if area < (scaler * 0.831 * second_max):
                    unmatched_scale = True
            elif reference_class == 3:
                if area < (scaler * 0.974 * third_max):
                    unmatched_scale = True
            elif reference_class == 4:
                if area < (scaler * 1.021 * fourth_max):
                    unmatched_scale = True

        if area > 0.72*fifth_max and not unmatched_scale:
            output[4] += 1
            cv.drawContours(copy, fifth_type[area], -1, (0, 255, 255), 2)
            copy = cv.putText(copy, 'Type 5', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2, cv.LINE_AA)

            inrange_blue = cv.inRange(single_piece, np.array([100, 100, 100]), np.array([110, 255, 150]))
            inrange_green = cv.inRange(single_piece, np.array([70, 90, 60]), np.array([85, 255, 150]))
            inrange_red_1 = cv.inRange(single_piece, np.array([0, 120, 100]), np.array([10, 255, 160]))
            inrange_red_2 = cv.inRange(single_piece, np.array([170, 130, 120]), np.array([180, 255, 150]))
            inrange_white = cv.inRange(single_piece, np.array([0, 10, 155]), np.array([180, 40, 255]))
            inrange_yellow = cv.inRange(single_piece, np.array([20, 120, 140]), np.array([30, 255, 200]))
            inrange_red = cv.bitwise_or(inrange_red_1, inrange_red_2)

            if inrange_blue.any() > 0:
                blue = True
            if inrange_green.any() > 0:
                green = True
            if inrange_red.any() > 0:
                red = True
            if inrange_white.any() > 0:
                white = True
            if inrange_yellow.any() > 0:
                yellow = True

            if blue and not green and not red and not white and not yellow:
                copy = cv.putText(copy, 'BLUE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 0, 0), 2, cv.LINE_AA)
                output[5] += 1
            elif not blue and green and not red and not white and not yellow:
                copy = cv.putText(copy, 'GREEN', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2, cv.LINE_AA)
                output[6] += 1
            elif not blue and not green and red and not white and not yellow:
                copy = cv.putText(copy, 'RED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 255), 2, cv.LINE_AA)
                output[7] += 1
            elif not blue and not green and not red and white and not yellow:
                copy = cv.putText(copy, 'WHITE', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2, cv.LINE_AA)
                output[8] += 1
            elif not blue and not green and not red and not white and yellow:
                copy = cv.putText(copy, 'YELLOW', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 255), 2, cv.LINE_AA)
                output[9] += 1
            else:
                copy = cv.putText(copy, 'MIXED', (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 0, 0), 2, cv.LINE_AA)
                output[10] += 1
        else:
            cv.drawContours(copy, fifth_type[area], -1, (0, 0, 0), 10)
            copy = cv.putText(copy, 'NOPE', (x, y), cv.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow('contours', copy)
    return output


if __name__ == '__main__':
    create_models()
    directory = r'train_images'

    # to launch directory of image folder is needed
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            output = detect(path)
            print(filename, ":")
            print(output)
            cv.waitKey(0)
            cv.destroyAllWindows()
            continue
        else:
            continue
