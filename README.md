# Object-Tracking-Using-Kalman-Filter

In this we perform automatic detection and motion-based tracking of moving objects in a video file. The tracking method was divided into 
2 parts: first, detecting moving objects in each frame, second, associating the detections corresponding to the same object over time. 

The detection of moving objects is done by background subtraction algorithm based on Gaussian mixture models. Morphological operations
are then applied to the resulting foreground mask to eliminate noise. Finally, blob analysis is used to detect groups of connected pixels,
which are likely to correspond to moving objects. Here, the association of detections to the same object is based solely on motion. 

The motion of each track is estimated by a Kalman filter. The filter is used to predict the track's location in each frame, and determine the likelihood of each detection being assigned to each track. 
So we first create a system object that is used for reading the video frames, detecting the moving objects by using the foreground detector
, and displaying the results using the blob analysis. All these were done using the computer vision toolbox in MATLAB. 

After this, we initialize the tracks by creating an array of tracks, where each track is a structure representing a moving object in the 
video. We then read the video frames and detect the objects using a function that performs motion segmentation using the foreground 
detector, morphological operations on the resulting binary mask to remove noisy pixels, and fill the holes in the remaining blobs. 

After this we need to predict the new locations of the existing tracks for which we use the Kalman filter to predict the centroid of 
each track in the current frame, and update its bounding box accordingly. The major task after this was to assign the detections to 
the tracks. This is done by minimizing cost.

The algorithm for this involves two steps: first, we compute the cost of assigning every 
detection to each track using the distance method of the vision.KalmanFilter System object, second, Solve the assignment problem
represented by the cost matrix using the assignDetectionsToTracks function which uses the Munkres' version of the Hungarian algorithm 
to compute an assignment which minimizes the total cost. 

Finally, the tracking results are displayed by draws a bounding box and label ID for each track on the video frame and the
foreground mask.

The MATLAB code can be found by the name ObjectTracker.m.
The video file used for detectiona and tracking objects was one in which the pedestrains were crossing a raod. The video was taken from
shutterstock.com. 
The detection and tracking of people can be found in the images pedestrian_detection_1.bmp, pedestrian_detection_2.bmp and Pedestrian_Tracking.bmp.
The laplacian of gradient can be seen in LoG_pedestrian.bmp.
