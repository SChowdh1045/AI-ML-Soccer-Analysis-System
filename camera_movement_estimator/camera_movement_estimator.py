import pickle
import cv2
import numpy as np
import os
from utils import measure_distance_between_points, measure_xy_distance

class CameraMovementEstimator:

    # Setting up the parameters for the Lucas-Kanade optical flow method and the Shi-Tomasi corner detection method, and creating a mask to specify where in the frame to detect features.
    def __init__(self, frame):
        # This sets the minimum distance that a feature must move to be considered as part of the camera movement.
        # If a feature moves less than this distance, it's not considered.
        self.minimum_distance = 5

        # This sets the parameters for the Lucas-Kanade optical flow method, which is used to track the movement of features between frames.
        self.lk_params = dict(
            winSize = (15, 15), # The size of the search window at each pyramid level.
            maxLevel = 2, # The maximum pyramid level number. If set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on.
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # The termination criteria of the iterative search algorithm. The algorithm stops either after 10 iterations or when the search window moves by less than 0.03.
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # This converts the first frame to grayscale. The Lucas-Kanade method works on grayscale images.
        mask_features = np.zeros_like(first_frame_grayscale) # This creates a mask with the same size as the first frame, filled with zeros. This mask will be used to specify the regions of the frame where features should be detected.
        
        # These lines set the values in the first 20 columns and the columns from 900 to 1050 of the mask to 1.
        # This means that features will be detected in these regions of the frame.
        # So these regions are vertical strips along the left and right sides of the frame. The : in the row position means that these regions extend over all rows, from top to bottom of the frame.
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        # This sets the parameters for the Shi-Tomasi corner detection method, which is used to detect the features to be tracked.
        self.features = dict(
            maxCorners = 100, # The maximum number of corners to return. If there are more than 100 corners in the image, it returns the 100 strongest ones.
            qualityLevel = 0.3, # The minimal accepted quality of image corners. The parameter characterizes the minimal accepted quality of image corners; the parameter value is multiplied by the best corner quality measure (smallest eigenvalue). The corners with the quality measure less than the product are rejected.
            minDistance =3, # The minimum possible Euclidean distance between the returned corners.
            blockSize = 7, # The size of an average block for computing derivative covariation matrix over each pixel neighborhood.
            mask = mask_features # The mask defining where to look for corners.
        )
                    

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # This initializes a 2D list to store the camera movement in the x and y directions for each frame.
        # The outer list has a length equal to the number of frames in the video.
        # Each element in the outer is another list: [0, 0]
        camera_movement = [[0,0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) # This converts the first frame to grayscale.
        
        # This detects the features in the first frame using the Shi-Tomasi corner detection method.
        # The double asterisks ** in **self.features is a syntax in Python called "dictionary unpacking".
        # When you use ** before a dictionary in a function call, it unpacks the dictionary and passes its key-value pairs as named arguments to the function.
        # This is equivalent to calling the function like this:
        # cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask_features)
        #
        # The cv2.goodFeaturesToTrack() function returns an array of detected corners. Each corner is represented as a 2D point (x, y coordinates) in the image.
        # The old_features variable, after the call to cv2.goodFeaturesToTrack(), is a 3D numpy array of shape (num_corners, 1, 2). Here, num_corners is the number of corners detected in the image.
        # Here's an example of what 'old_features' might look like:
        # array([[[ 50.,  50.]],
        #       [[100., 100.]],
        #       [[150., 150.]],
        #       ...,
        #       [[300., 300.]]], dtype=float32)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # This loop iterates over the rest of the frames in the video (excluding the first frame)
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY) # This converts the current frame to grayscale.

            # This calculates the optical flow between the previous frame and the current frame using the Lucas-Kanade method. Here's what the function arguments mean:
            # old_gray: The first 8-bit single-channel input image.
            # frame_gray: The second input image of the same size and the same type as old_gray.
            # old_features: Vector of 2D points for which the flow needs to be found.
            # **self.lk_params: Additional parameters for the Lucas-Kanade method.
            #
            # The function returns three values:
            # new_features: The calculated new positions of input features in the second image. This is a vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image.
            # _: The status vector (not used here, hence the underscore). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.
            # _: The error vector (not used here, hence the underscore). Each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
            #
            # The new_features variable, after the call to cv2.calcOpticalFlowPyrLK(), is a 3D numpy array of shape (num_features, 1, 2). Here, num_features is the number of feature points for which the optical flow was successfully calculated.
            # Each element of the new_features array is a 2D array of shape (1, 2), which represents a single feature point. The two elements of this 2D array are the x and y coordinates of the feature point in the second image (frame_gray).
            # Here's an example of what 'new_features' might look like:
            # array([[[ 51.,  51.]],
            #       [[101., 101.]],
            #       [[151., 151.]],
            #       ...,
            #       [[301., 301.]]], dtype=float32)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0 # This initializes the maximum distance moved by a feature to 0.
            camera_movement_x, camera_movement_y = 0,0 # This initializes the camera movement in the x and y directions to 0.

            # The purpose of this 2nd for-loop is to find the feature point that moved the farthest between the previous and current frames.
            # zip(new_features, old_features) is creating pairs of corresponding feature points from 'new_features' and 'old_features'. Each pair contains one feature point from 'new_features' and one from 'old_features'.
            # Here's an example of what zip(new_features, old_features) might look like:
            #   [
            #     (array([[51., 51.]], dtype=float32), array([[50., 50.]], dtype=float32)),
            #     (array([[101., 101.]], dtype=float32), array([[100., 100.]], dtype=float32)),
            #     ...
            #   ]
            for i, (new, old) in enumerate(zip(new_features, old_features)):

                # ".ravel()" is a numpy function that flattens an array into a 1D array.
                # So, if 'new' is array([[51., 51.]], dtype=float32), then new.ravel() would be array([51., 51.], dtype=float32)
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # This calculates the Euclidean distance between the feature points in the previous and current frames.
                distance = measure_distance_between_points(old_features_point, new_features_point)
                
                # This checks if the current distance is greater than the maximum distance.
                if distance > max_distance:
                    max_distance = distance # If the current distance is greater, it updates the maximum distance.

                    # This measures the distance in the x and y directions between the old and new feature points.
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point) 
            
            # This checks if the maximum distance moved by a feature is greater than the minimum distance threshold.
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y] # Sets the camera movement for the current frame to the measured movement.
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features) # Detects new features in the current frame using the Shi-Tomasi corner detection method. Will be used in the next iteration as 'old_features'.

            old_gray = frame_gray.copy() # This sets the current frame as the 'previous' frame for the next iteration.
        
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        # 'camera_movement' is returns a 2D list of camera movement in the x and y directions for each frame.
        return camera_movement
    

    # This function is adjusting the positions of game objects (like players, referees, or the ball) in each frame to account for camera movement.
    def add_adjusted_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for game_object, object_frames_list in tracks.items():
            for frame_num, frame_track in enumerate(object_frames_list):
                for track_id, track_info in frame_track.items():
                    
                    position = track_info["position"] # This gets the position of the game object in the current frame.
                    camera_movement = camera_movement_per_frame[frame_num] # This gets the camera movement in the current frame.
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1]) # This line calculates the adjusted position of the game object by subtracting the camera movement from the game object's position.
                    tracks[game_object][frame_num][track_id]["position_adjusted"] = position_adjusted # This line adds the adjusted position to the track dictionary under a new key named "position_adjusted". This allows the adjusted position to be accessed later.
    


    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame) 

        return output_frames