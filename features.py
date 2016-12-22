import numpy as np
from scipy.ndimage.morphology import binary_erosion

class Feature:
    def __init__(self, curvature, x_position, y_position, orientation, size):
        self.shape = Shape(curvature)

        self.orientation = Orientation(orientation)
        self.x_position = XPosition(x_position)
        self.y_position = YPosition(y_position)
        self.size = Size(size)

    def feature_vector(self):
        vector = [0] * 16
        vector[0:3] = self.shape.feature_vector()
        vector[3:7] = self.orientation.feature_vector()
        vector[7:10] = self.x_position.feature_vector()
        vector[10:13] = self.y_position.feature_vector()
        vector[13:16] = self.size.feature_vector()

        return np.array(vector, dtype=np.float64)

    @staticmethod
    def from_vector(vector):
        f = Feature(0, 0, 0, 0, 0)
        f.shape = Shape.from_vector(vector[0:3])
        f.orientation = Orientation.from_vector(vector[3:7])
        f.x_position = XPosition.from_vector(vector[7:10])
        f.y_position = YPosition.from_vector(vector[10:13])
        f.size = Size.from_vector(vector[13:16])

        return f

    def __eq__(self, other):
        return (other.shape == self.shape and other.orientation == self.orientation and
                other.x_position == self.x_position and other.y_position == self.y_position and
                other.size == self.size)

    def __hash__(self):
        h1 = hash(self.shape)
        h2 = h1 * 31 + hash(self.orientation)
        h3 = h2 * 31 + hash(self.x_position)
        h4 = h3 * 31 + hash(self.y_position)
        h5 = h4 * 31 + hash(self.size)
        return h5

    def __str__(self):
        return "Feature({0}, {1}, {2}, {3}, {4})".format(self.shape, self.orientation, self.x_position, self.y_position, self.size)

    @staticmethod
    def recognize_features(image, with_mapping=False):
        if (len(image.shape) <> 2):
            return []

        features = set()

        # We apply the follow_shape function to all the nonzero pixels of the image.
        points = zip(*image.nonzero())

        point_count = float(len(points))

        for point in points:
            feature = None
            if with_mapping <> True:
                (length, distance, (start, end), position) = follow_shape(point, image, with_mapping)
                curvature = float(distance) / length
                orientation = np.arctan2(start[0] - end[0], start[1] - end[1])
                feature = Feature(curvature, position[0] / image.shape[0], position[1] / image.shape[1], orientation + np.pi, length / point_count)
            else:
                (length, distance, (start, end), position, mask) = follow_shape(point, image, with_mapping)
                curvature = float(distance) / length
                orientation = np.arctan2(start[0] - end[0], start[1] - end[1])
                feature = Feature(curvature, position[0] / image.shape[0], position[1] / image.shape[1], orientation + np.pi, length / point_count)
                feature.mask = mask
                
            
            if feature not in features:
                features.add(feature)

        return list(features)
            

class Shape:
    def __init__(self, curvature):
        self.loop = trapezoid(-1, -1, 0.15, 0.25, curvature)
        self.semiloop = trapezoid(0.15, 0.25, 0.85, 0.95, curvature)
        self.line = trapezoid(0.85, 0.95, 2, 2, curvature)

    def __eq__(self, other):
        return self.loop == other.loop and self.semiloop == other.semiloop and self.line == other.line

    def __hash__(self):
        h1 = hash(self.loop)
        h2 = h1 * 31 + hash(self.semiloop)
        h3 = h2 * 31 + hash(self.line)

        return h3

    def feature_vector(self):
        return [self.loop, self.semiloop, self.line]

    def __str__(self):
        return "Shape(loop={0}, semiloop={1}, line={2})".format(self.loop, self.semiloop, self.line)

    @staticmethod
    def from_vector(vector):
        s = Shape(0)
        s.loop = vector[0]
        s.semiloop = vector[1]
        s.line = vector[2]

        return s

class Orientation:
    def __init__(self, orientation):
        self.right = max(trapezoid(-1, -1, 0.4, 1.25, orientation),
                         trapezoid(5, 6, 7, 7, orientation))
        self.left = trapezoid(1.8, 2.8, 3.3, 4.3, orientation)
        self.up = trapezoid(0.3, 1.3, 1.9, 2.9, orientation)
        self.down = trapezoid(3.2, 4.4, 5.1, 6.3, orientation)

    def __eq__(self, other):
        return self.right == other.right and self.left == other.left and self.up == other.up and self.down == other.down

    def __hash__(self):
        h1 = hash(self.right)
        h2 = h1 * 31 + hash(self.left)
        h3 = h2 * 31 + hash(self.up)
        h4 = h3 * 31 + hash(self.down)

        return h4

    def feature_vector(self):
        return [self.right, self.left, self.up, self.down]

    def __str__(self):
        return "Orientation(right={0}, left={1}, up={2}, down={3})".format(self.right, self.left, self.up, self.down)

    @staticmethod
    def from_vector(vector):
        o = Orientation(0)
        o.right = vector[0]
        o.left = vector[1]
        o.up = vector[2]
        o.down = vector[3]

        return o

class YPosition:
    def __init__(self, position):
        self.high = trapezoid(0.65, 0.75, 2, 2, position)
        self.middle = trapezoid(0.25, 0.35, 0.65, 0.75, position)
        self.low = trapezoid(-1, -1, 0.25, 0.35, position)

    def __eq__(self, other):
        return (self.high == other.high and self.middle == other.middle and self.low == other.low)

    def __hash__(self):
        h1 = hash(self.high)
        h2 = h1 * 31 + hash(self.middle)
        h3 = h2 * 31 + hash(self.low)

        return h3

    def feature_vector(self):
        return [self.high, self.middle, self.low]

    def __str__(self):
        return "YPosition(high={0}, middle={1}, low={2})".format(self.high, self.middle, self.low)

    @staticmethod
    def from_vector(vector):
        y = YPosition(0)
        y.high = vector[0]
        y.middle = vector[1]
        y.low = vector[2]

        return y

class XPosition:
    def __init__(self, position):
        self.left = trapezoid(-1, -1, 0.25, 0.35, position)
        self.middle = trapezoid(0.25, 0.35, 0.65, 0.75, position)
        self.right = trapezoid(0.65, 0.75, 2, 2, position)

    def __eq__(self, other):
        return (self.left == other.left and self.middle == other.middle and self.right == other.right)

    def __hash__(self):
        h1 = hash(self.left)
        h2 = h1 * 31 + hash(self.middle)
        h3 = h2 * 31 + hash(self.right)

        return h3

    def feature_vector(self):
        return [self.left, self.middle, self.right]

    def __str__(self):
        return "XPosition(left={0}, middle={1}, right={2})".format(self.left, self.middle, self.right)

    @staticmethod
    def from_vector(vector):
        x = XPosition(0)
        x.left = vector[0]
        x.middle = vector[1]
        x.right = vector[2]

        return x

class Size:
    def __init__(self, fraction):
        self.large = trapezoid(0.65, 0.75, 2, 2, fraction)
        self.medium = trapezoid(0.25, 0.35, 0.65, 0.75, fraction)
        self.small = trapezoid(-1, -1, 0.25, 0.35, fraction)

    def __eq__(self, other):
        return (self.large == other.large and self.medium == other.medium and self.small == other.small)

    def __hash__(self):
        h1 = hash(self.large)
        h2 = h1 * 31 + hash(self.medium)
        h3 = h2 * 31 + hash(self.small)

        return h3

    def feature_vector(self):
        return [self.large, self.medium, self.small]

    def __str__(self):
        return "Size(larg={0}, medium={1}, small={2})".format(self.large, self.medium, self.small)

    @staticmethod
    def from_vector(vector):
        s = Size(0)

        s.large = vector[0]
        s.medium = vector[1]
        s.large = vector[2]

        return s

def trapezoid(left_min, left_max, right_max, right_min, value):
    if (value < left_min):
        return 0
    elif (value < left_max):
        return (value - left_min) / (left_max - left_min)
    elif (value < right_max):
        return 1
    elif (value < right_min):
        return 1 - (value - right_max) / (right_min - right_max)
    else:
        return 0

def follow_shape((x, y), image, with_mapping=False):
    start = (x, y)
    members = set([(x, y)])
    first_end = None
    previous_angle = None
    angle_difference = 0

    # We use a try-except block to break out of nested loops isn't python fun??
    try:
        # Walk the first limb of the shape.
        # We first walk the limb in a leftwards fashion.
        while (True):
            # Get the three by three region centered aroung the current pixel.
            region = image[x-1:x+2, y-1:y+2]

            # We are in the border region this actually shouldn't happen with
            # the way th MNIST dataset is build but we'll never know.
            # for now we just stop searching in that direction.
            if region.shape <> (3, 3):
                if first_end == None:
                    first_end = (x, y)
                    (x, y) = start
                    start = first_end
                    previous_angle = angle(first_end, (x, y))

                    if angle_difference <> 0:
                        angle_difference = -angle_difference

                    continue
                else:
                    break

            ps = [(p1 + x - 1, p2 + y - 1) for (p1, p2) in zip(*region.nonzero())]

            # If there are more than 3 non-zero points then we are at a junction.
            # Which means this is the end of the current limb of the shape.
            # If we have only walked one limb we revert back to the starting point
            # and walk in the other direction from there.
            # If we have already walked both limbs we end the search.
            #
            # This assumption about allowing a maximum of 3 points in the Shape
            # is unfortunately not true. However it mostly works so we leave it at that.
            # There are some situations wherein this assumption is not true while we
            # do want to follow the feature.
            #
            # Take the next situation:
            #   +----------+
            #   |          |
            #   |  111111  |
            #   |  1<--    |
            #   |   1      |
            #   |    1     |
            #   +----------+
            #
            # The one where the arrow points to has more than 2 live neighbours
            # however it is clearly not a junction.
            # To determine this we clearly need a larger area to consult.
            # To solve this we look at the 5x5 region around the pixel it is only
            # allowed to have a maximum of six live pixels. Two more than in the
            # 3x3 area. If there are more six live pixels in the 5x5 area the section
            # is most likely a branch.
            if (len(ps) > 3 and len(zip(*image[x-2:x+3, y-2:y+3].nonzero())) > 6):
                if first_end == None:
                    first_end = (x, y)
                    (x, y) = start
                    start = first_end
                    previous_angle = angle(first_end, (x, y))

                    if angle_difference <> 0:
                        angle_difference = -angle_difference

                    continue
                else:
                    break

            # Loop over all the members that are not currently part of the shape.
            # There should actually be only one.
            # However this is not true the first time this is done.
            # Therefore we follow the first one in this list and forget about the rest.
            # We will walk the other point second time we get around this shape.
            for point in [p for p in ps if p not in members]:
                # If the we have a previous angle then we check that the
                # difference between that angle and the current angle
                # has the same sign as the difference between the previous
                # angle and the angle before it.
                if previous_angle <> None:
                    # If we have previously calculated the angle sign of the
                    # difference in angles between points of the shape we use
                    # that to se if we are still following a single shape.
                    # We use this to only partially follow S-like shapes.
                    if angle_difference <> 0:
                        a = angle(start, point)
                        angle_diff = None

                        if np.sign(a) <> np.sign(previous_angle):
                            if np.abs(a - previous_angle) > np.pi:
                                a_adjusted = a - 2 * np.pi * np.sign(a)
                                angle_diff = np.sign(a_adjusted - previous_angle)
                        else:
                            angle_diff = np.sign(a - previous_angle)

                        # If the angle differences don't match up end this limb.
                        if angle_diff <> angle_difference:
                            if first_end == None:
                                first_end = (x, y)
                                (x, y) = start
                                start = first_end
                                previous_angle = angle(first_end, (x, y))

                                if angle_difference <> 0:
                                    angle_difference = -angle_difference

                                break
                            else:
                                raise GetOutOfLoop
                        
                        previous_angle = a

                    else:
                        a = angle(start, point)

                        # Now since the angle is given in the range of [-pi, pi]
                        # we might have a problem if the do not have the same sign.
                        # First of all we might go from just above zero to just below
                        # zero or the other way round.
                        # Secondly we could have wrapped around from +pi to -pi
                        # or the vice versa.
                        # Now both of these are fine and should not be a problem
                        # and in fact the first cases going from positive to over
                        # zero to negative or going the opposite way is not a problem.
                        # However wrapping around will probably give the wrong sign
                        # for the difference in angles.
                        # So what we do is adjust the new angle to be close to the
                        # previous angle either by adding or subtracting 2 * pi.
                        if np.sign(a) <> np.sign(previous_angle):
                            # So we only adjust it if the absolute difference 
                            # is more than pi.
                            if np.abs(a - previous_angle) > np.pi:
                                a_adjusted = a - 2 * np.pi * np.sign(a)
                                angle_difference = np.sign(a_adjusted - previous_angle)
                        else:
                            angle_difference = np.sign(a - previous_angle)

                        previous_angle = a
                            
                else:
                    previous_angle = angle(start, point)
                    
                members.add(point)
                (x, y) = point
                break
            else:
                # If there were no new points to add then we finished this limb.
                if first_end == None:
                    first_end = (x, y)
                    (x, y) = start
                    start = first_end
                    previous_angle = angle(first_end, (x, y))

                    if angle_difference <> 0:
                        angle_difference = -angle_difference

                    continue
                else:
                    break
    except GetOutOfLoop:
        pass

    # So now we have all the coordinates that belong to the feature.
    # Next we calculate all the data that describes the feature.

    l = len(members)
    d = np.hypot(first_end[0] - x, first_end[1] - y)
    ends = (first_end, (x, y))
    position = tuple(np.mean(np.array(zip(*members)), axis=1))

    if with_mapping == False:
        return (l, d, ends, position)
    else:
        mapping = np.zeros(image.shape, dtype=np.int8)
        mapping[zip(*members)] = 1
        return (l, d, ends, position, mapping)    
        

def angle((x1, y1), (x2, y2)):
    return np.arctan2(x1 - x2, y1 - y2)

class GetOutOfLoop(Exception):
    pass

if __name__ == "__main__":
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    features = Feature.recognize_features(image, with_mapping=True)

    print "Image"
    print image

    print "\n"
    for f in features:
        print f
        print f.mask

    print len(features)