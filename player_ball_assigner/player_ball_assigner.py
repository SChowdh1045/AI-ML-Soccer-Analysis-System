from utils import get_center_of_bbox, measure_distance_between_points

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
    

    def assign_ball_to_player(self, player_frame, ball_bbox):
        center_of_ball = get_center_of_bbox(ball_bbox)

        min_distance = 999999
        assigned_player = -1

        for player_id, player_track in player_frame.items():
            player_bbox = player_track["bbox"]

            # The reason for calculating the distance from both feet to the ball, rather than just the center of the player to the ball, is because the distance from the ball to a player's foot is a more accurate measure of how close the player is to being able to interact with the ball
            distance_left_foot = measure_distance_between_points((player_bbox[0], player_bbox[-1]), center_of_ball)
            distance_right_foot = measure_distance_between_points((player_bbox[2], player_bbox[-1]), center_of_ball)
            distance = min(distance_left_foot, distance_right_foot) # The distance from the ball to the closest foot of the player

            # If the distance is less than the maximum allowed distance (70), and it's the smallest distance so far, assign the ball to this player
            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        
        return assigned_player