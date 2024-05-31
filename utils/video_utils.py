import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release() # Release the VideoCapture object to free up the resources.
    
    return frames


def save_video(output_video_frames, output_video_path):
    # 'fourcc' means 'four character code' and is used to specify the video codec.
    # The asterisk (*) in cv2.VideoWriter_fourcc(*'XVID') is Python's argument unpacking operator. It takes a collection of items and unpacks them into individual elements.
    # In this case, 'XVID' is a string, which in Python is an iterable of its characters. When you use the * operator on 'XVID', it unpacks the string into individual characters: 'X', 'V', 'I', 'D'.
    # The cv2.VideoWriter_fourcc() function expects four arguments (each a single character) representing the four-character code of the codec. By using *'XVID', you're passing the four characters of 'XVID' as separate arguments to the function, as if you had written cv2.VideoWriter_fourcc('X', 'V', 'I', 'D').
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # output_video_frames[0].shape[1] is the width of the frame, and output_video_frames[0].shape[0] is the height of the frame.
    
    for frame in output_video_frames:
        out.write(frame) # Write the frames to the video file one by one.
    
    out.release() # Release the VideoWriter object to free up the resources.