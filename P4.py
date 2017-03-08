"""
The goals / steps of this project are the following:

    ! Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    ! Apply a distortion correction to raw images.
    Use color transforms, gradients, etc., to create a thresholded binary image.
    Apply a perspective transform to rectify binary image ("birds-eye view").
    Detect lane pixels and fit to find the lane boundary (masking and
    thresholding).
    Determine the curvature of the lane and vehicle position with respect to center.
    Warp the detected lane boundaries back onto the original image.
    Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    Camera calibration source images in ../camera_cal
    Images for testing pipeline are in ../test_images
    Save examples of the output from EACH stage of the pipeline in
    ../output_images and include a description in your writeup
    
"""
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys


#################################################################
# CAMERA CALIBRATION DOES NOT NEED TO RUN EVERYTIME. LOAD DATA INSTEAD.
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
#dist_pickle = pickle.load( open( "camera_cal/P4_calibration.pickle", "rb" ) )
#mtx = dist_pickle["mtx"]
#dist = dist_pickle["dist"]
#################################################################


# Camera Calibration
def calibrate_camera():
    ''' 
    This function takes no arguments and returns a dictionary containing the 
    camera calibration matrix and distortion coefficients. The function is 
    ONLY called if the main python script is run with the command line 
    argument "cal". For example,
    >>> python3 P4.py "cal"
    Camera calibration images are assumed to reside in a sub-directory, relative 
    to the parent script, called "camera_cal". Calibration images are also assumed
    to be named calibration*.jpg where * is an integer > 0. 
    Calibration results are saved to disk via pickle in the same directory as
    the calibration images. 
    ''' 

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3D points in real world space, NOTE: z is assumed = 0
    imgpoints = [] # 2D poionts in image plane 
    
    # nx & ny corresponds to number of INTERIOR corner points on
        # chessboard calibration images. Alter as necessary if 
        # different calibration images are used.
    nx = 9 # rows
    ny = 6 # columns
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... , (nx,ny,0)
    objp = np.zeros( (nx * ny, 3), np.float32) 

    # Inserts x, y coordinates into first two columns of objp
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Read in calibration images from disk
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        #print(fname)
        # Read in each image
        img = mpimg.imread(fname)
    
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If corners are found, add object points, image points to initialized arrays
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            #plt.show()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Store camera calibration matrix (mtx) and distortion coefficients (dist) in
    # a dictionary
    camera_dict = {"mtx": mtx, "dist": dist}
    # Write data to disk using pickle
    with open('camera_cal/P4_calibration.pickle', 'wb') as f:
        pickle.dump(camera_dict, f)

    # Read in and undistort a test images
    test_images = glob.glob('test_images/*.jpg')
    for image in test_images:
       test = mpimg.imread(str(image)) 
       dst = cv2.undistort(test, mtx, dist, None, mtx)
       plt.imshow(dst)
       # Cleanup string representation of filename
       f1 = str(image.rstrip('.jpg')) # strip off .jpg 
       f = str(f1.split('/')[1])      # remove test_images/
       plt.suptitle('Undistorted ' + f)
       plt.savefig('output_images/undistorted_' + f + '.png')

    #test_img = mpimg.imread('test_images/test4.jpg')
    #dst = cv2.undistort(test_img, mtx, dist, None, mtx)
    #plt.imshow(dst)
    #plt.suptitle('Undistorted Test Image 4')
    #plt.savefig('output_images/undistorted_test4.png')
    ##plt.show()
    return camera_dict

if len(sys.argv) > 1 and sys.argv[1] == "cal":
    camera_dict = calibrate_camera()
    mtx = camera_dict["mtx"]
    dist =camera_dict["dist"]
else:
    dist_pickle = pickle.load( open("camera_cal/P4_calibration.pickle","rb") )
    mtx  = dist_pickle["mtx"]
    dist = dist_pickle["dist"]


# Returns the undistorted image
def cal_undistort(img, mtx, dist):
    undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_image

undistorted = cal_undistort(img, objpoints, imgpoints)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(undistorted)
#ax2.set_title('Undistorted Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)






"""
A note on image shape

The shape of the image, which is passed into the calibrateCamera function, is just the height and width of the image. One way to retrieve these values is by retrieving them from the grayscale image shape array gray.shape[::-1]. This returns the image height and width in pixel values like (960, 1280).

Another way to retrieve the image shape, is to get them directly from the color image by retrieving the first two values in the color image shape array using img.shape[0:2]. This code snippet asks for just the first two values in the shape array.

It's important to use an entire grayscale image shape or the first two values of a color image shape.
"""
























