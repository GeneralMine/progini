import cv2
import datetime
import mss
import numpy as np
import os
import pyautogui
import sys
import time
from camera import CameraFactory
import argparse

# Screen resolution
SCREEN_WIDTH=1280
SCREEN_HEIGHT=720

# Definition of area of interest
# This is set to the area of the projection
X_OFFSET=280
Y_OFFSET=0
WIDTH=680
HEIGHT=340

# Thresholds
REFLECTIONS_THRESHOLD=30
MOVEMENT_THRESH=100000

# Image processing iteration (for debugging only)
iteration = 0


def to_greenchannel(image):
    result = image[:][:][1]
    return result


def clip_projection_area(image):
    result = image[int(Y_OFFSET) : (int(Y_OFFSET) + int(HEIGHT)), int(X_OFFSET) : (int(X_OFFSET) + int(WIDTH))]
    return result


def preprocess_image(image):

    clipped = clip_projection_area(image)
    greenImage = to_greenchannel(clipped)

    return greenImage


def remove_background(camera):
    global args

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    _, backgroundRemoved = cv2.threshold(camera, 100, 255, cv2.THRESH_BINARY)

    if args.debug: write_debug_image('background_removed', backgroundRemoved)

    eroded = cv2.erode(camera, kernel, 1)

    if args.debug: write_debug_image('background_removed_eroded', eroded)

    return eroded


def calc_threshold_image(old_image, new_image):
    global args

    img_neg = cv2.subtract(old_image, new_image)
    drawable_img = cv2.bitwise_not(img_neg)

    # Filter reflections with treshhold
    _, filtered_neg = cv2.threshold(img_neg, REFLECTIONS_THRESHOLD, 255, cv2.THRESH_TOZERO)

    if args.debug: write_debug_image('filtered_neg', filtered_neg)

    return filtered_neg

def determine_contour_maximum_point(image):
    # Get the biggest contour on the screen
    _img, contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_contour = max(contours, key = cv2.contourArea)

    # Get the extrem points of the biggest contour.
    left = tuple(biggest_contour[biggest_contour[:, :, 0].argmin()][0])
    right = tuple(biggest_contour[biggest_contour[:, :, 0].argmax()][0])
    top = tuple(biggest_contour[biggest_contour[:, :, 1].argmin()][0])
    bot = tuple(biggest_contour[biggest_contour[:, :, 1].argmax()][0])

    # Assume that the fingertip is on the opposite of the side the hand enters the area of interest.
    fingertip = None
    if (bot[1] == HEIGHT - 1):
        return top
    elif (left[0] == 0):
        return right
    elif (right[0] == WIDTH - 1):
        return left
    elif (top[1] == 0):
        return bot


def scale_position_to_screen(cam_position):
    x = cam_position[0] / WIDTH * SCREEN_WIDTH
    y = cam_position[1] / HEIGHT * SCREEN_HEIGHT

    return (int(x), int(y))


def move_mouse_to(position):
    pyautogui.moveTo(position[0], position[1])


def click_at(position):
    pyautogui.click(position[0], position[1])


def write_debug_image(name, image):
    global folder_debug_images

    write_image(folder_debug_images, name, image)


def write_output_image(name, image):
    global folder_output_images

    write_image(folder_output_images, name, image)


def write_image(folder, name, image):
    global iteration

    cv2.imwrite('{}/{}-{}.png'.format(folder, name, iteration), image)


def main():
    global args, iteration

    img_old = preprocess_image("data/camera-29.png")
    
    img_new = remove_background("data/camera-29.png")

    threshold_img = calc_threshold_image(img_old, img_new)
    img_delta = np.sum(threshold_img)
    print(img_delta)

main()
