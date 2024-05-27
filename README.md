# Detection_of_Road_Line
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import math
# Initialize cache and first_frame variables
cache = None
first_frame = True
# Function to mask the region of interest in the image
def interested_region(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        mask_color_ignore = (255,) * img.shape[2]
    else:
        mask_color_ignore = 255
        
    cv2.fillPoly(mask, vertices, mask_color_ignore)
    return cv2.bitwise_and(img, mask)
# Function to apply Hough Transform and draw lane lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_drawn(line_img, lines)
    return line_img
# Function to draw the detected lines on the image
def lines_drawn(img, lines, color=[255, 0, 0], thickness=6):
    global cache
    global first_frame

    slope_l, slope_r = [], []
    lane_l, lane_r = [], []
    α = 0.2
# Classify lines based on their slopes
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.4:
                slope_r.append(slope)
                lane_r.append(line)
            elif slope < -0.4:
                slope_l.append(slope)
                lane_l.append(line)

    if ((len(lane_l) == 0) or (len(lane_r) == 0)):
        print('no lane detected')
        return
# Calculate mean slopes and intercepts for left and right lanes
    slope_mean_l = np.mean(slope_l, axis=0)
    slope_mean_r = np.mean(slope_r, axis=0)
    mean_l = np.mean(np.array(lane_l), axis=0)
    mean_r = np.mean(np.array(lane_r), axis=0)

    if ((slope_mean_r == 0) or (slope_mean_l == 0)):
        print('dividing by zero')
        return
# Define y-coordinates for lane lines
    y1 = img.shape[0]
    y2 = int(img.shape[0] * 0.6)
# Calculate x-coordinates for left and right lanes
    x1_l = int((y1 - mean_l[0][1]) / slope_mean_l + mean_l[0][0])
    x2_l = int((y2 - mean_l[0][1]) / slope_mean_l + mean_l[0][0])
    x1_r = int((y1 - mean_r[0][1]) / slope_mean_r + mean_r[0][0])
    x2_r = int((y2 - mean_r[0][1]) / slope_mean_r + mean_r[0][0])
# Store current frame's coordinates
    present_frame = np.array([x1_l, y1, x2_l, y2, x1_r, y1, x2_r, y2], dtype="float32")
# Smooth the lane lines using exponential moving average
    if first_frame:
        next_frame = present_frame
        first_frame = False
    else:
        prev_frame = cache
        next_frame = (1 - α) * prev_frame + α * present_frame
# Draw the smoothed lane lines on the image
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)
# Update cache with current frame
    cache = next_frame
# Function to process each frame of the video
def process_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blur, 50, 150)
    # Define region of interest
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = interested_region(edges, vertices)
    # Hough Transform parameters 
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20
    # Detect and draw lines
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # Overlay the line image on the original image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result
# Define the input and output video paths
white_output = r"C:\Users\HP\OneDrive\.ipynb_checkpoints\image.mp4" 
clip1 = VideoFileClip(r"C:\Users\HP\OneDrive\.ipynb_checkpoints\image.mp4")
# Process the video and save the output
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
