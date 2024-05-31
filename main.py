# When you execute from "utils import read_video, save_video" in main.py, Python does the following:
# 1) It looks for a package or module named utils. In this case, utils is a package because there's a directory named utils with an __init__.py file in it.
# 2) Python then executes the __init__.py file inside the utils package. In this case, that file contains the line "from .video_utils import read_video, save_video", which imports the read_video and save_video functions from the video_utils module in the same package.
# 3) After executing __init__.py, the read_video and save_video functions are now attributes of the utils package.
# 4) Finally, the "from utils import read_video, save_video" statement in this file imports these functions from the utils package into main.py, so you can use them directly as read_video and save_video in my main.py code.
from utils import read_video, save_video
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import cv2
import numpy as np

def main():
    # Read the input video.
    video_frames = read_video('input_videos\\08fd33_4.mp4')

    tracker = Tracker('models\\XL\\best.pt')
    tracks = tracker.track_objects(video_frames, read_from_stub=True, stub_path='stubs\\tracks_stubs.pkl')

    # Setting player/referee/ball positions in the 'tracks' dictionary
    tracker.add_positions_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs\\camera_movement_stub.pkl')
    camera_movement_estimator.add_adjusted_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_positions_to_tracks(tracks)

    # Interpolating ball positions. This is done to fill in the gaps in the ball's trajectory.
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assigning team colors to players and saving it in the 'tracks' dictionary (NOT DRAWING YET).
    # The purpose of the 'assign_team_color()' method is to assign each team an official color. Uses only the first frame of the video. Only needs to be called once.
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # 1st loop iterates over each frame of the video.
    # 2nd loop iterates over each player in the current frame and assigns a team ID and team color to each player.
    for frame_num, player_frame in enumerate(tracks["players"]):
        for player_id, track in player_frame.items():
            team_id = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            
            # creating new keys in the 'tracks' dictionary (along with 'bbox') to store the team and team color of each player.
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team_id]


    # Assigning the ball to a player in each frame, and also storing the team that has control of the ball in each frame.
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = [] # List to store the team that has control of the ball in each frame.
    
    for frame_num, player_frame in enumerate(tracks['players']):
        # '[0]' here is a dictionary key, not an index. When I interpolated the ball positions, every frame is guaranteed to have a ball position.
        ball_bbox = tracks["ball"][frame_num][0]["bbox"]
        assigned_player = player_ball_assigner.assign_ball_to_player(player_frame, ball_bbox)

        if assigned_player != -1:
            # Creating a new key 'has_ball' in the 'tracks' dictionary to store the player who has the ball in each frame.
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            # If the ball is not assigned to any player in the current frame, the team that had control of the ball last will still be considered to have control of the ball.
            team_ball_control.append(team_ball_control[-1])
    

    # Drawing all the annotations on the video frames.
    output_video_frames = tracker.draw_annotations(video_frames, tracks, np.array(team_ball_control))

    # Drawing the camera movement on the video frames.
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw the speed and distance of each player on the video frames.
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Saving the video to computer with the annotations on each frame.
    save_video(output_video_frames, 'output_videos\\output_video.avi')


if __name__ == '__main__':
    main()
