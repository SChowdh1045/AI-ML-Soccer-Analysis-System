from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd

from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    ############################## TRACKING FUNCTIONS ##############################
    def detect_frames(self, video_frames):
        batch_size = 15
        detections = []
        
        for i in range(0, len(video_frames), batch_size):
            batch = video_frames[i : i+batch_size]

            # 'conf=0.1' is the confidence threshold. It's the minimum confidence score required for a detection to be considered valid. If the confidence score of a detection is below this threshold, it will be ignored.
            batch_detections = self.model.predict(batch, conf=0.1)
            detections.extend(batch_detections)
        
        return detections
    
    def track_objects(self, video_frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            
            return tracks

        detections = self.detect_frames(video_frames)

        # 'tracks' is a dictionary with the keys being 'players', 'referees', and 'ball'.
        # Each key holds a list of dictionaries, where each dictionary represents a video frame.
        # The dictionaries that represent video frames hold key,value pairs where the key is the 'tracker_id' and the value is another dictionary with the key named 'bbox'.
        # The 'bbox' key holds a list of four integers that represent the bounding box coordinates in the form of [x1, y1, x2, y2].
        # So for the 'players' key for example, the structure is as follows: 
        
        # tracks["players"] = [
        #                       {0: {"bbox": [0,0,10,20]}, 1: {"bbox": [5,10,20,30]}, 17: {"bbox": [120,0,133,40]}, ...},
        #                       {0: {"bbox": [5,10,15,22]}, 1: {"bbox": [10,20,26,29]}, 17: {"bbox": [105,10,128,56]}, ...}, 
        #                       ... 
        #                     ]
        #
        # But actually, along with "bbox", the dictionaries for the players would also have "team" and "team_color" keys. I just didn't include them here for simplicity.
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            class_names = detection.names # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            class_inverted = {v:k for k,v in class_names.items()} # {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}

            # Convert the detections to 'supervision detection' format
            # Output is first a 2D array of the bounding boxes in the form of [x1, y1, x2, y2], along with other information about the class (e.g. 'ball', 'goalkeepers', 'players', 'referees')
            detections_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeepers to players. Not hardcoding the ids because I want to make the code more flexible.
            # 'detections_supervision.class_id' is a list of class IDs (integers) for each bounding box in the 'detections_supervision' object.
            for obj_index, class_id in enumerate(detections_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detections_supervision.class_id[obj_index] = class_inverted['player']
            
            # To track objects
            # Updates 'detections_supervision' with an integer array named "tracker_id" with the same length as the number of bounding boxes in the 'detection_supervision' object and holds unique IDs for each object.
            detections_with_tracking = self.tracker.update_with_detections(detections_supervision)

            # Creating a new dictionary for the current frame in the 'tracks' dictionary for each of the players, referees, and ball.
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})


            # The 'detections_with_tracking' object is an instance of the Detections class from the supervisely_lib library. When you iterate over an instance of this class using a for loop, it doesn't return the entire object for each iteration. Instead, it returns a tuple for each detected object in the frame.
            # The Detections class has an __iter__ method defined, which is a special method in Python classes that allows the instances of the class to be iterable. When you use a for loop on an instance of a class that has this method, it will return the values yielded by this method.
            # In the case of the Detections class, the __iter__ method is defined to yield a tuple for each detected object, which includes the bounding box coordinates, mask, the confidence score, the class ID, the tracker ID, and some additional data. Like this:
            # (array([851.93, 634.9, 902.23, 721.19], dtype=float32), None, 0.9111449, 2, 1, {'class_name': 'player'})
            # So, when you do 'for frame_detection in detections_with_tracking:', 'frame_detection' is a tuple representing a single detected object, not the entire 'detections_with_tracking' object. This is why 'frame_detection' seems to get updated to the next element in each of the arrays in each iteration. It's actually moving to the next detected object.
            # This design allows you to easily process each detected object individually in the for loop. 
            for frame_detection in detections_with_tracking:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_inverted['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                elif class_id == class_inverted['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
                elif class_id == class_inverted['ball']:
                    tracks["ball"][frame_num][0] = {"bbox": bbox}
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        # 'ball_positions' is a list of dictionaries, where each dictionary represents a video frame. There is only 1 ball in each frame, so each dictionary only has 1 key-value pair.
        # It would be of this format: [
        #                               { 0: {"bbox": [0,0,10,20]} }, 
        #                               { 0: {"bbox": [5,10,15,22]} }, 
        #                               ...
        #                             ]
        #
        # In this line, 'ball_positions' becomes a 2D list. Each inner list represents the bounding box coordinates for the ball in a single frame, and is structured as [x1, y1, x2, y2].
        # If a frame doesn't contain a ball, the corresponding inner list will be empty ([]), indicating that the ball's position is missing for that frame.
        # So the structure of ball_positions would now look something like this:
        #  [
        #     [x1, y1, x2, y2],  # bounding box for ball in frame 1
        #     [],                # ball missing in frame 2
        #     [x1, y1, x2, y2],  # bounding box for ball in frame 3
        #     ...
        #  ]
        ball_positions = [x.get(0, {}).get("bbox", []) for x in ball_positions]
        
        # Converting the 2D list ball_positions into a pandas DataFrame.
        # A pandas DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. It is similar to a spreadsheet or SQL table, or a dictionary of Series objects.
        # The pd.DataFrame() function is used to create a DataFrame. The first argument to this function is the data that you want to convert into a DataFrame. In this case, it's the 'ball_positions' list.
        # The columns parameter is used to specify the column labels for the DataFrame. In this case, the labels are "x1", "y1", "x2", and "y2", which represent the coordinates of the bounding box for the ball in each frame.
        # So, after this line of code, 'df_ball_positions' is a DataFrame that looks something like this:
        #   x1	y1	x2	y2
        # 0	x1	y1	x2	y2
        # 1	NaN	NaN	NaN	NaN
        # 2	x1	y1	x2	y2
        # .	.	.	.	.
        #
        # Each row in the DataFrame corresponds to a frame, and the columns represent the bounding box coordinates for the ball in that frame. If the ball's position is missing in a frame, the corresponding row in the DataFrame will contain NaN values.
        # The interpolate() and bfill() functions are built-in pandas functions that make it easy to fill in missing values in a DataFrame. While numpy also has functions for handling missing data, they are not as straightforward to use as pandas' functions.
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # The interpolate() and bfill() functions fill missing values based on the values in other rows, but they don't have any knowledge of the physical context of the data. They don't know that the data represents the position of a bouncing soccer ball, and they don't try to predict the ball's trajectory based on physics. They simply fill missing values based on the values in neighboring rows.
        # The interpolate() function, by default, uses linear interpolation. This means that it fills a missing value with a value that lies on the straight line between the nearest non-missing values.
        # For example, if the x1 values for frame 1 and frame 3 are 10 and 30, and the x1 value for frame 2 is missing, interpolate() will fill the missing value with 20, which is the midpoint of 10 and 30.
        #
        # The bfill() function, short for 'backward fill', fills missing values in the DataFrame with the next (downwards in the DataFrame) valid value along the column. I mainly used it here to fill the missing values at the beginning of the DataFrame.
        # For example, if the first few frames are missing values, bfill() will fill them with the next valid value downwards in the DataFrame. For example, if the x1 values for the first three frames are missing and the x1 value for the fourth frame is 10, bfill() will fill the missing values in the first three frames with 10.
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # df_ball_positions.to_numpy(): This converts the DataFrame df_ball_positions into a numpy array. This is done because pandas DataFrames have a lot of additional functionality that isn't needed here, and working with numpy arrays can be faster and more memory-efficient.
        # .tolist(): This converts the numpy array into a list of lists. Each inner list represents a row in the DataFrame and contains the values of the columns in that row (i.e. the bounding box coordinates for the ball in that frame in the format: [x1, y1, x2, y2]).
        # {0: {"bbox": x}} for x in ...: This is a list comprehension that creates a new list of dictionaries. For each list 'x' in the list of lists, it creates a dictionary {0: {"bbox": x}}. The key 0 represents the track ID of the ball, and the value {"bbox": x} is another dictionary that contains the updated bounding box of the ball.
        ball_positions = [{0: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def add_positions_to_tracks(self, tracks):
        for game_object, object_frames_list in tracks.items():
            for frame_num, frame_track in enumerate(object_frames_list):
                for track_id, track_info in frame_track.items():
                    bbox = track_info["bbox"]
                    
                    if game_object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    # Creating new key 'position' in the 'tracks' dictionary to store the position of each player, referee, and the ball in each frame.
                    tracks[game_object][frame_num][track_id]["position"] = position

    ############################## DRAWING FUNCTIONS ##############################
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Making a copy of the frame so that the original frame is not modified

            # These would be the dictionaries for the players, referees, and ball for the current frame, and of this format (except ball, which has only 1 key-value pair per frame):
            # {0: {"bbox": [0,0,10,20]}, 1: {"bbox": [5,10,20,30]}, 17: {"bbox": [120,0,133,40]}, ...}
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Looping over the keys (track IDs) and values (bounding boxes) in the dictionaries.
            # These would be of this format: 0: {"bbox": [0,0,10,20]}
            # Where '0' is 'track_id' and '{"bbox": [0,0,10,20]}' is 'player'/'referee'/'ball'.
            for track_id, player_track in player_dict.items():
                color = player_track.get("team_color", (0,0,255)) # If the 'team_color' key doesn't exist, it returns the default color of (0,0,255) which is red.
                frame = self.draw_ellipses_and_rects(frame, player_track["bbox"], color, track_id)
                
                # If the player has the ball, draw a red triangle above their head.
                if player_track.get("has_ball", False):
                    frame = self.draw_triangle(frame, player_track["bbox"], (0,0,255))
            
            # Don't want to give the referees track IDs.
            for _, referee in referee_dict.items():
                frame = self.draw_ellipses_and_rects(frame, referee["bbox"], (0,255,255))
            
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
        

        return output_video_frames
    
    def draw_ellipses_and_rects(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # bottom of the bounding box

        x_center, _ = get_center_of_bbox(bbox)
        bbox_width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(bbox_width), int(bbox_width*0.35)),
            angle=0,
            startAngle= -45,
            endAngle= 235,
            color=color,
            thickness=int(2),
            lineType=cv2.LINE_4
            )
        

        if track_id is not None:
            # Drawing the rectangles on the bottom of the players with their track ID
            rect_width = 40
            rect_height = 20
            
            x1_rect = x_center - rect_width//2
            y1_rect = y2 + 12
            
            x2_rect = x_center + rect_width//2
            y2_rect = y1_rect + rect_height

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            # Drawing the track_ID TEXT
            x_text = x1_rect + 10
            y_text = y1_rect + 18

            if track_id < 10:
                x_text += 4
            elif track_id > 99:
                x_text -= 5
            
            cv2.putText(frame, str(track_id), (int(x_text), int(y_text)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        

        # The returned frame has the ellipse(s) drawn on it for the players, referees, and ball.
        # Also the rectangles with the track IDs for the players.
        return frame

    def draw_triangle(self, frame, bbox, color):
        y1 = int(bbox[1]) # top of the ball's bounding box
        x_center, _ = get_center_of_bbox(bbox)

        # numpy arrays are more efficient for numerical operations than Python lists.
        triangle_points = np.array([
            [x_center, y1],
            [x_center - 10, y1 - 20],
            [x_center + 10, y1 - 20]
        ])

        # The cv2.drawContours() function expects a list of contours, where each contour is a 2D array of points.
        # Even if you have only one contour (like here), you still need to put it in a list, hence the square brackets around 'triangle_points'.
        # If you have multiple contours, you would put each contour in the list, like this: [contour1, contour2, contour3, ...]
        # The '0' is the index of the contour in the list. Since we only have one contour, the index is 0. If I wanted to draw all the contours in the list, I would put -1 instead of 0.
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # The border of the triangle

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Drawing a semi-transparent rectangle at the bottom-right corner of the frame to display the team that has control of the ball.
        overlay = frame.copy()

        # 'frame.shape[1]' is the width of the frame, and 'frame.shape[0]' is the height of the frame.
        cv2.rectangle(overlay, (frame.shape[1] - 700, frame.shape[0] - 170), (frame.shape[1] - 50, frame.shape[0] - 50), (255, 255, 255), cv2.FILLED)

        alpha = 0.4
        # The cv2.addWeighted() function is a part of OpenCV, a library used for image processing tasks. This function calculates the weighted sum of two arrays (in this case, images), which can be used to blend or mix these images. The function's signature is: cv2.addWeighted(src1, alpha, src2, beta, gamma, dst)
        # src1: The first source array. In thia case, this is 'overlay', which is a copy of the original frame with a semi-transparent rectangle drawn on it.
        # alpha: The weight of the first array elements. In this case, this is 'alpha', which is set to 0.4. This means that the elements of overlay will contribute 40% to the final image.
        # src2: The second source array. In this case, this is 'frame', which is the original frame.
        # beta: The weight of the second array elements. In this case, this is '1 - alpha', which is 0.6. This means that the elements of frame will contribute 60% to the final image.
        # gamma: A scalar added to each sum. In this case, this is 0, so it doesn't affect the result.
        # dst: The destination array, where the result will be stored. In this case, this is 'frame', so the result will be stored in the original frame.
        #
        # So, cv2.addWeighted() is blending the overlay image (which has a semi-transparent rectangle) and the original frame image. The result is an image that looks like the original frame, but with a semi-transparent rectangle. The rectangle appears semi-transparent because it's only contributing 40% to the final image, while the original frame is contributing 60%.
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Python slicing is exclusive of the end index, so I have to do 'frame_num+1' to include the current frame.
        # Since 'team_ball_control' is a numpy array, 'teams_ball_control_until_now' will also be a numpy array.
        teams_ball_control_until_now = team_ball_control[:frame_num+1]
        
        # Counting the number of times each team has had control of the ball so far up to the current frame.
        # This is way of counting the number of times each team has had control of the ball is called 'boolean indexing'. We can only use this in numpy arrays.
        # When you do 'teams_ball_control_until_now == 1', it returns a Boolean array of the same shape as 'teams_ball_control_until_now', with True where the condition is met and False where it isn't.
        # Then, when you use this Boolean array to index 'teams_ball_control_until_now' like this: teams_ball_control_until_now[teams_ball_control_until_now == 1], it returns a new array that includes only the elements where the Boolean array is True.
        # Finally, .shape[0] returns the size of the first dimension of this new array, which is the number of True values in the Boolean array, or equivalently, the number of 1s/2s in the original 'teams_ball_control_until_now' array.
        team1_ball_control = teams_ball_control_until_now[teams_ball_control_until_now == 1].shape[0]
        team2_ball_control = teams_ball_control_until_now[teams_ball_control_until_now == 2].shape[0]

        # Calculating the percentage of time each team has had control of the ball so far up to the current frame.
        team1_percentage = (team1_ball_control / len(teams_ball_control_until_now)) * 100
        team2_percentage = (team2_ball_control / len(teams_ball_control_until_now)) * 100

        # Drawing the text on the frame to display the team that has control of the ball.
        cv2.putText(frame, f"Team 1 Ball Control: {team1_percentage:.2f}%", (frame.shape[1] - 290, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team2_percentage:.2f}%", (frame.shape[1] - 290, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
