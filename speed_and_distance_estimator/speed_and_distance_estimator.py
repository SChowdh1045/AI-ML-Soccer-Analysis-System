import cv2
from utils import measure_distance_between_points, get_foot_position

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5 # Calculate speed and distance every 5 frames
        self.frame_rate = 24
    

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for game_object, object_frames_list in tracks.items():
            if game_object == "ball" or game_object == "referees":
                continue 
            
            number_of_frames = len(object_frames_list)
            
            # Jumps every 5 frames (i.e. 0, 5, 10, 15, etc...)
            for frame_num in range(0, number_of_frames, self.frame_window):
                next_frame = min(frame_num + self.frame_window, number_of_frames - 1) # Ensures that the next frame is not out of bounds

                for track_id, _ in object_frames_list[frame_num].items():
                    # If the track_id doesn't exist in the next frame, skip to the next iteration. This is because we need to calculate the speed and distance between two frames.
                    if track_id not in object_frames_list[next_frame]:
                        continue

                    # Get the start and end positions of the player in the current frame and the next frame. "position_transformed" is the position of the player in meters.
                    start_position = object_frames_list[frame_num][track_id]["position_transformed"]
                    end_position = object_frames_list[next_frame][track_id]["position_transformed"]

                    # If the start or end position is None, that means the player is out of the trapezoid area from the view transformer. Skip to the next iteration.
                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance_between_points(start_position, end_position)
                    time_elapsed = (next_frame - frame_num)/self.frame_rate # The logic here is that the frame rate is 24 frames per second. So, the time elapsed between two frames is 1/24 seconds.
                    speed_meters_per_second = distance_covered/time_elapsed # The formula for speed is distance/time
                    speed_km_per_hour = speed_meters_per_second * 3.6 # Convert speed from m/s to km/h

                    if game_object not in total_distance:
                        total_distance[game_object] = {}
                    
                    if track_id not in total_distance[game_object]:
                        total_distance[game_object][track_id] = 0
                    
                    total_distance[game_object][track_id] += distance_covered

                    # Loop through all the frames between the current frame and the next frame to store the speed and distance of the player in each frame (because we're jumping every 5 frames)
                    for frame_num_batch in range(frame_num, next_frame):
                        # If the track_id doesn't exist in the current frame, skip to the next frame.
                        if track_id not in tracks[game_object][frame_num_batch]:
                            continue
                        
                        # Creating 2 new keys in the 'tracks' dictionary named 'speed' and 'distance' to store the speed and distance of each player in each frame.
                        tracks[game_object][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                        tracks[game_object][frame_num_batch][track_id]["distance"] = total_distance[game_object][track_id]
    


    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            for game_object, object_frames_list in tracks.items():
                if game_object == "ball" or game_object == "referees":
                    continue 
                
                for _, track_info in object_frames_list[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40 # Shifting the position of the text downwards

                        # The map() function in Python applies a given function to each item of an iterable (like a list or a tuple) and returns a list of the results.
                        # In this case, map(int, position) is applying the int function to each element of position. This will convert each element of position to an integer.
                        # The tuple() function then converts the result back to a tuple.
                        # So, position = tuple(map(int, position)) is converting each element of position to an integer and storing the result as a tuple.
                        # For example, if position was [12.3, 45.6], after this line, position would be (12, 46). 
                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            
            output_frames.append(frame)
        

        return output_frames