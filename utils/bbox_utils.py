def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox

    # The x and y values are packed into a tuple and returned.
    # When you return multiple values separated by commas, Python automatically packs them into a tuple.
    return int((x1+x2)/2), int((y1+y2)/2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def measure_distance_between_points(p1, p2):
    # p1 and p2 are tuples of (x, y) coordinates.
    x1, y1 = p1
    x2, y2 = p2

    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5


def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]


# Gets the bottom-center of the bounding box.
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)