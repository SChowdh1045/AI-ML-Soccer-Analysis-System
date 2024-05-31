# The __init__.py file is a special file in Python. It's often used in Python packages.
# When you create a directory (say, 'utils') and put an __init__.py file in it, you're telling Python that 'utils' should be treated as a package.
# A package is simply a way of organizing related modules into a directory hierarchy.
# The __init__.py file is executed when the package is imported

# Python import statements do not use file paths, but rather they use module names.
# The dot in "from .video_utils import read_video, save_video" is not a directory indicator like it is in file paths. Instead, it's a part of Python's relative import syntax, where a single dot represents the current package.
# So, from ".video_utils import read_video, save_video" is not the same as from ".\video_utils import read_video, save_video". The first one is a correct relative import statement, while the second one would result in a syntax error.
from .video_utils import read_video, save_video
from .bbox_utils import get_center_of_bbox, get_bbox_width, measure_distance_between_points, measure_xy_distance, get_foot_position