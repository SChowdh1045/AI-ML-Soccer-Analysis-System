import numpy as np 
import cv2

# The ViewTransformer class is used to transform the perspective of the video frames from the camera's perspective to a top-down view of the soccer field.
# This is done to convert the pixel dimensions to real-world dimensions, which allows for more accurate calculations of player speed and distance covered.
# THIS ONLY COVERS THE MIDDLE PART OF THE SOCCER FIELD. THE SIDES ARE NOT COVERED.
class ViewTransformer():
    def __init__(self):
        # These are the real-world dimensions of the soccer court in meters. Soccer field is around 105m x 68m.
        court_width = 68

        # Soccer fields have stripes. I need only 4 of them for the trapezoid. 
        # Half a soccer field would be 105/2 = 52.5m
        # Half a soccer field usually has 9 stripes, so 52.5/9 = 5.83m. This means the width of 1 stripe is 5.83m.
        # In need 4 stripes, so 5.83*4 = 23.32m
        court_length = 23.32

        # These are the pixel coordinates of the 4 corners of the trapezoid in the first frame of the video. This is a 2D numpy array.
        # These are manually set through trial & error (by the tutorial). The first frame of the video was used to get these pixel coordinates.
        # They are in this order: bottom-left, top-left, top-right, bottom-right
        self.pixel_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 915]
        ])
        
        # These are the real-world coordinates of the 4 corners of the rectangle (contained within trapezoid) that I want to transform the soccer court to. This is a 2D numpy array.
        # They are in this order: bottom-left, top-left, top-right, bottom-right
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Convert the pixel_vertices and target_vertices to float32 data type because that's what the cv2.getPerspectiveTransform() function expects.
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # The cv2.getPerspectiveTransform() function is used to calculate the 3x3 perspective transformation matrix that can be used to transform the four vertices of a source image to the corresponding points in a destination image.
        # The perspective transformation matrix is always a 3x3 matrix. This is due to the mathematical principles behind projective transformations, also known as homographies, which are used in perspective transformations.
        # In this case, self.pixel_vertices are the coordinates of the four vertices in the source image (the trapezoid in the first frame of the video), and self.target_vertices are the coordinates of the four vertices in the destination image (the rectangle in the real-world soccer field).
        # The output, self.perspective_transformer, is a 3x3 2D numpy array (matrix) of type float64. This matrix can be used to apply the calculated perspective transformation to any point in the source image to find its corresponding point in the destination image.
        # Here's an example of what the output might look like:
        # array([   
        #           [ 1.43841336e+00, -1.12316841e-01, -1.57958984e+02],
        #           [ 3.33066907e-16,  1.43841336e+00, -3.55271368e-15],
        #           [ 1.11022302e-16, -1.11022302e-16,  1.00000000e+00]
        #       ])
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)



    def add_transformed_positions_to_tracks(self, tracks):
        for game_object, object_frames_list in tracks.items():
            for frame_num, track in enumerate(object_frames_list):
                for track_id, track_info in track.items():
                    
                    # "position_adjusted" is the adjusted position of the object in the frame, taking into account the camera movement.
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    
                    if position_transformed is not None:
                        # "position_transformed" is a 2D numpy array with shape (1, 2). I need to convert it to a list of two elements.
                        # The squeeze() function in numpy is used to remove single-dimensional entries from the shape of an array.
                        # When squeeze() is applied, it becomes a 1D array with a shape of (2,), which is equivalent to a list with two elements.
                        # The tolist() function then converts this numpy array to a regular Python list. This is done to make the data easier to work with in the rest of the code.
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    # Creating a new key "position_transformed" in the 'tracks' dictionary to store the transformed positions of the objects in the frame.
                    tracks[game_object][frame_num][track_id]["position_transformed"] = position_transformed

    
    # The transform_point() function takes a point in the camera's perspective, checks if it's inside the trapezoid, and if it is, transforms it to the top-down view and returns it. If the point is outside the trapezoid, it returns None.
    def transform_point(self, point):
        # Converts the coordinates of the point to integers. This is done because the cv2.pointPolygonTest() function, which is used in the next line, requires the point to be an integer tuple.
        p = (int(point[0]), int(point[1]))
        
        # This line checks if the point is inside the trapezoid defined by self.pixel_vertices.
        # The cv2.pointPolygonTest() function returns a non-negative value if the point is inside the polygon, and a negative value if it's outside. So, is_inside will be True if the point is inside the trapezoid, and False otherwise.
        # The cv2.pointPolygonTest() function takes three arguments:
        # contour: This is a 2D numpy array of shape (n, 1, 2) where n is the number of vertices in the polygon. Each element of this array is a 2D point (x, y). In this case, 'self.pixel_vertices' is the contour.
        # pt: This is the point that you want to check. It should be a 2D point (x, y). In this case, 'p' is the point.
        # measureDist: This is a boolean value that specifies whether the function should calculate the shortest distance from the point to the polygon edges. If measureDist is True, the function returns the signed distance. If it's False, the function only checks whether the point is inside the polygon or not. In your case, measureDist is False, so the function will return 1.0 if the point is inside the polygon, -1.0 if it's outside, and 0.0 if it's on an edge.
        #
        # So, cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 checks if the point p is inside the polygon defined by self.pixel_vertices. If the point is inside the polygon or on an edge, the function will return True, otherwise it will return False.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        
        # If the point is outside the trapezoid, return None.
        if not is_inside:
            return None

        # This line reshapes the point to a 3D array with a shape of (1, 1, 2), and converts it to float32.
        # This is done because the cv2.perspectiveTransform() function, which is used in the next line, requires the input to be a 3D float32 array.
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)

        # This line applies the perspective transformation to the reshaped point using the transformation matrix 'self.perspective_transformer'
        # The output is a 3D numpy array with shape (1, 1, 2) containing the transformed point.
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        # I need to return the transformed point as a 2D numpy array with shape (1, 2).
        return tranform_point.reshape(-1,2)                    