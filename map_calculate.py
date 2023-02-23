import math

import cv2
import numpy as np
from mse import mse
from calculateDegreeToDestination import calculateDegreeToDestination

box_points = []
button_down = False


# Function for rotation image based on angle
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Function for scaling image based on scale percent
def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent

def invariantMatchTemplate(rgbimage, rbgDroneImage, method, matched_thresh, rgbdiff_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax, bestX, bestY):
    # image of the drone that is resized to desire dimension
    rbgDroneImage = cv2.resize(rbgDroneImage, (bestX, bestY))
    # for better matching graying out the images
    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rbgDroneImage, cv2.COLOR_RGB2GRAY)
    # max width that image can be scaled
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    # points for path of the drone
    all_points = []
    # error between matched image and original one
    error=0
    # actual finding the match
    if minmax == False:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                else:
                    raise MethodError("There's no such comparison method for template matching.")
                for pt in zip(*satisfied_points[::-1]):
                    all_points.append([pt, next_angle, actual_scale])
    else:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                    img_gray_rotated = img_gray
                else:
                    rotated_template = scaled_template_gray
                    img_gray_rotated = rotate_image(img_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCOEFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCORR)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCORR_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        top_left = (int(top_left[0] * (next_scale / 100)), int(top_left[1] * (next_scale / 100)))
                        predictionImage = img_gray_rotated[top_left[1]:top_left[1]+rotated_template.shape[0],top_left[0]:top_left[0]+rotated_template.shape[1]]
                        try:
                            error = mse(predictionImage, rotated_template)
                        except:
                            error=0
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_SQDIFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)

                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)

                else:
                    raise MethodError("There's no such comparison method for template matching.")
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[3])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[3])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    color_filtered_list = []
    template_channels = cv2.mean(rbgDroneImage)
    template_channels = np.array([template_channels[0], template_channels[1], template_channels[2]])
    for point_info in points_list:
        point = point_info[0]
        cropped_img = rgbimage[point[1]:point[1]+height, point[0]:point[0]+width]
        cropped_channels = cv2.mean(cropped_img)
        cropped_channels = np.array([cropped_channels[0], cropped_channels[1], cropped_channels[2]])
        diff_observation = cropped_channels - template_channels
        total_diff = np.sum(np.absolute(diff_observation))
        if total_diff < rgbdiff_thresh:
            color_filtered_list.append([point_info[0],point_info[1],point_info[2],
                                        [(point_info[0][0],point_info[0][1]),
                                         (int(point_info[0][0]+int((width)*(point_info[2]/100))),point_info[0][1]),
                                         (int(point_info[0][0]+int((width)*(point_info[2]/100))), int(point_info[0][1]+int((height)*(point_info[2]/100)))),
                                         (point_info[0][0],int(point_info[0][1]+int((height)*(point_info[2]/100))))],error])
    return color_filtered_list

# Similar method as before but this one is with sequecne images
def invariantMatchTemplateSeq(rgbimage, rbgDroneImage, method, matched_thresh, rgbdiff_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax):
    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rbgDroneImage, cv2.COLOR_RGB2GRAY)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []
    error=0
    if minmax == False:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                else:
                    raise MethodError("There's no such comparison method for template matching.")
                for pt in zip(*satisfied_points[::-1]):
                    all_points.append([pt, next_angle, actual_scale])
    else:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                    img_gray_rotated = img_gray
                else:
                    rotated_template = scaled_template_gray
                    img_gray_rotated = rotate_image(img_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCOEFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCORR)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_CCORR_NORMED)
                    img_gray_rotatedCopyy=img_gray_rotated.copy()
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        top_left = (int(top_left[0] * (next_scale / 100)), int(top_left[1] * (next_scale / 100)))
                        bottom_right = (int((top_left[0] + rotated_template.shape[0])) , int((top_left[1] + rotated_template.shape[1])))
                        img_gray_rotatedtt=rotate_image(img_gray_rotated, next_angle)
                        predictionImage = img_gray_rotated[top_left[1]:top_left[1]+rotated_template.shape[0],top_left[0]:top_left[0]+rotated_template.shape[1]]

                        try:
                            error = mse(predictionImage, rotated_template)
                        except:
                            error=0
                            # print('not found')
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_SQDIFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)

                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray_rotated,rotated_template,cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + width, top_left[1] + height)
                        predictionImage = img_gray_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        error = mse(predictionImage, rotated_template)

                else:
                    raise MethodError("There's no such comparison method for template matching.")
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[3])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[3])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    color_filtered_list = []
    template_channels = cv2.mean(rbgDroneImage)
    template_channels = np.array([template_channels[0], template_channels[1], template_channels[2]])
    for point_info in points_list:
        point = point_info[0]
        cropped_img = rgbimage[point[1]:point[1]+height, point[0]:point[0]+width]
        cropped_channels = cv2.mean(cropped_img)
        cropped_channels = np.array([cropped_channels[0], cropped_channels[1], cropped_channels[2]])
        diff_observation = cropped_channels - template_channels
        total_diff = np.sum(np.absolute(diff_observation))
        if total_diff < rgbdiff_thresh:
            color_filtered_list.append([point_info[0],point_info[1],point_info[2],
                                        [(point_info[0][0],point_info[0][1]),
                                         (int(point_info[0][0]+int((width)*(point_info[2]/100))),point_info[0][1]),
                                         (int(point_info[0][0]+int((width)*(point_info[2]/100))), int(point_info[0][1]+int((height)*(point_info[2]/100)))),
                                         (point_info[0][0],int(point_info[0][1]+int((height)*(point_info[2]/100))))],error])
    return color_filtered_list

def calculateForVideoOnlySeq(video_path,map_sector,startX,startY):
    cap = cv2.VideoCapture(video_path)
    img_rgb = None
    cropped_template_rgb = None
    largeMapImageCopy = None
    # image view height on the map sector
    h=260
    # image view width on the map sector
    w=100
    lastcenter=None
    heightcamera=100
    entireImg= cv2.imread(map_sector)
    points=[]
    # degrees of the drone based on north
    yaw=[-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-10,-10,-10,0,0,0,10,10,10,20,20,20,20,20,20,20,20,20,20,20,20,30,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,30,40,40,40,40,40,40,40,40,30,40,40,40,40,40,40,40,40,30,40,40,40,40,40,40,50,50,50,50,50,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80]
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if i>len(yaw)-1:
            break
        nowyaw=yaw[i]
        i+=1
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if img_rgb is None:
            img_rgb = frame
        else:
            img_rgb=largeMapImageCopy
        cropped_template_rgb=frame
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        if cropped_template_rgb is  not None:
            # cropping image that we are looking for in the next image
            cropped_template_rgb = cropped_template_rgb[400:int(height - 400), 600:int(width-600)]
            cropped_template_rgb = np.array(cropped_template_rgb)
            cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
            height, width = cropped_template_gray.shape
            points_list = invariantMatchTemplateSeq(img_rgb, cropped_template_rgb, "TM_CCOEFF_NORMED", 0.5, 500, [-20,20], 10, [100,110], 10, True, True)
            # all points that were found in matching started with best one with index 0
            for point_info in points_list:
                point = point_info[0]
                scale = point_info[2]
                canterx=(point[0]+int(point[0]+width*scale/100))/2
                cantery=(point[1]+int(point[1]+height*scale/100))/2
                if lastcenter is not None:
                    nowyaw=-nowyaw
                    #calculating the point of drone camera to original map sector
                    y1=(cantery-(img_rgb.shape[0]/2))*(h/entireImg.shape[1])
                    x1=(canterx-(img_rgb.shape[1]/2))*(w/entireImg.shape[0])
                    if y1!=0:
                        nowyaw+=math.asin(x1/y1)* (180/math.pi )
                    angleInRadians = (nowyaw) * (math.pi / 180)
                    sin = math.sin(angleInRadians)
                    cos = math.cos(angleInRadians)
                    x1After = int(x1+points[len(points)-1][0])
                    y1After = int(y1+points[len(points)-1][1])
                    x1After2 = int(
                        ((x1After - points[len(points) - 1][0]) * cos - (y1After - points[len(points) - 1][1]) * sin)) + \
                               points[len(points) - 1][0]
                    y1After2 = int(
                        ((x1After - points[len(points) - 1][0]) * sin + (y1After - points[len(points) - 1][1]) * cos)) + \
                               points[len(points) - 1][1]
                    points.append([x1After2, y1After2])

                else:
                    points.append([startX,startY])
                lastcenter=(canterx,cantery)
                # drawing the computations
                print(points)
                img_rgb=cv2.resize(img_rgb, (600,300))
                entireImg = cv2.circle(entireImg,
                                     (startX, startY),
                                     radius=3,
                                     color=(0, 0, 255),
                                     thickness=3)
                centers = np.array(points, np.int32)
                centers = centers.reshape((-1, 1, 2))
                cv2.polylines(entireImg, [centers], False, (0, 255, 255), 3)
                cv2.imshow('image2', entireImg)
                cv2.imshow('image', img_rgb)
                largeMapImageCopy = frame.copy()
                break;
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cv2.waitKey(0)
def calculate(frameskipnumber, path_to_video, map_sector_path):
    # image view width on the map sector
    bestX = 228
    # image view height on the map sector
    bestY = 124
    cap = cv2.VideoCapture(path_to_video)
    frameskip=0
    centers=[]
    overrallAngle=0
    needToRecalculateScale=False
    notFind=0
    errorTreshold=0.1
    noGPS = False
    turned = False
    landed = False
    while cap.isOpened():
        ret, frame = cap.read()
        if frameskip==0:
            frameskip=frameskipnumber
        else:
            frameskip-=1
            continue
        if ret:
            img_rgb = cv2.imread(map_sector_path)
            cropped_template_rgb=frame
            image_center = tuple(np.array(img_rgb.shape[1::-1]) / 2)
            img_rgbcopy = img_rgb.copy()
            img_rgb_copy = rotate_image(img_rgb, overrallAngle)
            if needToRecalculateScale:
                points_list = invariantMatchTemplate(img_rgb_copy, cropped_template_rgb, "TM_CCORR_NORMED", 0.6, 600,
                                                     [-30, 30], 10, [50, 300], 10, True, True, bestX, bestY)
                needToRecalculateScale = False
            else:
                points_list = invariantMatchTemplate(img_rgb_copy, cropped_template_rgb, "TM_CCORR_NORMED", 0.6, 600,
                                                     [-30, 30], 10, [100, 100+10], 10, True, True, bestX, bestY)

            for point_info in points_list:
                scale=point_info[2]
                angle = point_info[1]
                overrallAngle+=angle
                # calculating the points after rotations to be well drawn on the original map
                if overrallAngle>360:
                    overrallAngle-=360
                elif overrallAngle<-360:
                    overrallAngle += 360
                angleInRadians = (360-overrallAngle) * (math.pi / 180)
                sin = math.sin(angleInRadians)
                cos = math.cos(angleInRadians)
                x1 = point_info[3][0][0] - image_center[0]
                y1 = point_info[3][0][1] - image_center[1]
                x2 = point_info[3][1][0] - image_center[0]
                y2 = point_info[3][1][1] - image_center[1]
                x3 = point_info[3][2][0] - image_center[0]
                y3 = point_info[3][2][1] - image_center[1]
                x4 = point_info[3][3][0] - image_center[0]
                y4 = point_info[3][3][1] - image_center[1]
                img_rgb=img_rgbcopy.copy()
                x1After = int((x1 * cos - y2 * sin) + image_center[0])
                y1After = int((x1 * sin + y1 * cos) + image_center[1])
                x2After = int((x2 * cos - y2 * sin) + image_center[0])
                y2After = int((x2 * sin + y2 * cos) + image_center[1])
                x3After = int((x3 * cos - y3 * sin) + image_center[0])
                y3After = int((x3 * sin + y3 * cos) + image_center[1])
                x4After = int((x4 * cos - y4 * sin) + image_center[0])
                y4After = int((x4 * sin + y4 * cos) + image_center[1])
                center_dronex=int((x1After+x3After)/2)
                center_droney=int((y1After+y3After)/2)
                # calculating degree between our mission coordinates and drone coordinates
                degree, centerFront, centerBack =calculateDegreeToDestination((x1After,y1After), (x2After,y2After), (x3After, y3After), (x4After, y4After), 380, 733)
                # taking to account only points in certain treshold
                if len(centers)>0 and not((centers[len(centers)-1][0]<center_dronex*(1+errorTreshold) and \
                        centers[len(centers)-1][0]>center_dronex*(1-errorTreshold)) and (centers[len(centers)-1][1]<center_droney*(1+errorTreshold)) \
                        and centers[len(centers)-1][1]>center_droney*(1-errorTreshold)):
                    overrallAngle -= angle
                    notFind+=1
                    if notFind>5:
                        errorTreshold=min(errorTreshold+0.01,0.15)
                        if errorTreshold==1.2:
                            needToRecalculateScale=False
                        notFind=0
                    Draw=False
                    continue
                else:
                    Draw = True
                    errorTreshold=0.1
                #     actual drawing of the map sector and drone camera
                if Draw:
                    centers.append([center_dronex,center_droney])

                    img_rgb=cv2.circle(img_rgb,
                                       (center_dronex,center_droney),
                                       radius=3,
                                       color=(0, 0, 255),
                                       thickness=3)
                    pts = np.array([[x1After, y1After], [x2After, y2After], [x3After, y3After], [x4After, y4After]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img_rgb, [pts], True, (0, 255, 255), 3)
                    cv2.rectangle(img_rgb, (633, 457), (643, 467), (0, 0, 255), 20)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 100)
                    fontScale = 1
                    color = (0, 255, 0)
                    thickness = 3
                    text = "turn  %.2f degree" % degree
                    if degree < 0.05:
                        turned = True
                    if centerFront == (616, 513):
                        landed = True
                    if centerFront == (765, 121):
                        noGPS = True
                    if noGPS:
                        print("NO GPS")
                        if landed == False:
                            if turned == False:
                                cv2.putText(img_rgb, text, org, font,
                                                fontScale, color, thickness, cv2.LINE_AA)
                            else:
                                cv2.putText(img_rgb, "FLY STRAIGHT " + str(513 - centerFront[1]) + "m", org, font,
                                            fontScale, color, thickness, cv2.LINE_AA)
                        else:
                            print("LANDED")
                            cv2.putText(img_rgb, "LAND", org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                            cv2.putText(img_rgb, "MISSION COMPLETE", (400, 400), font,
                                        3, (255, 255, 255), 5, cv2.LINE_AA)
                        cv2.putText(img_rgb, "NO GPS MODE", (50,50), font,
                                            fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
                    else:
                        print("GPS MODE")
                        cv2.putText(img_rgb, "GPS MODE", (50, 50), font,
                                    fontScale, (0, 255, 0), thickness, cv2.LINE_AA)


                    centers2 = np.array(centers, np.int32)
                    centers2 = centers2.reshape((-1, 1, 2))
                    cv2.polylines(img_rgb, [centers2], False, (0, 255, 255), 3)
                    img_rgb=cv2.resize(img_rgb,(1000,600))
                    cropped_template_rgb=cv2.resize(cropped_template_rgb,(300,200))


                    cv2.imshow('Sector Map', img_rgb)

                    cv2.imshow('Camera from Drone', cropped_template_rgb)
                    if landed == True:
                        cv2.waitKey(0)
                        break
                    break
                Draw=True
            if landed == True:
                break
        else:
            break
        # cv2.imshow('image', img_rgbcopy)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
