import numpy as np
from numba import njit

"""
Pure Pursuit Helper Functions
"""

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast.
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]  # differences between consecutive waypoints
    l2s   = diffs[:,0]**2 + diffs[:,1]**2         # squared lengths of each segment

    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)

    # nearest point, dist to nearest point, param t, segment index
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Given a circle (point, radius) and a piecewise linear trajectory, find the
    first point on the trajectory that intersects with the circle.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None

    trajectory = np.ascontiguousarray(trajectory)

    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:] + 1e-6
        V = end - start

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            continue

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if 0.0 <= t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if 0.0 <= t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        else:
            if 0.0 <= t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif 0.0 <= t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    # Wrap around if no intersection found and wrap=True
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b*b - 4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if 0.0 <= t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif 0.0 <= t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Return desired speed and steering angle using pure pursuit geometry.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner that uses a simple pure pursuit strategy.
    """
    def __init__(self, conf, wheelbase):
        self.wheelbase = wheelbase
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.0  # maximum reacquire distance
        self.drawn_waypoints = []
        
        self._latest_speed = 0.0
        self._latest_steering_angle = 0.0
        self._current_segment_index = None
        self._current_deviation = 0.0

    def load_waypoints(self, conf):
        """
        Load waypoints from file.
        """
        self.waypoints = np.loadtxt(conf.wpt_path,
                                    delimiter=conf.wpt_delim,
                                    skiprows=conf.wpt_rowskip)

    def set_path(self, wpts_vector):
        """
        Accepts a (32,) numpy array of x,y coordinates,
        reshapes it to (16,2), and overrides self.waypoints 
        with a 3-column array [x, y, speed=1.0].
        
        Once this is called, the CSV data is effectively ignored.
        """
        
        # Reshape the flat array (32,) to (16,2)
        new_xy = wpts_vector.reshape((16, 2))

        '''
        If we want to use a third speed column
        new_wpts = np.zeros((16, 3))
        new_wpts[:, 0:2] = new_xy
        new_wpts[:, 2] = 1.0

        self.waypoints = new_wpts
        '''
        
        # Override the CSV-based waypoints with just x,y
        self.waypoints = new_xy
    
    def render_waypoints(self, e):
        """
        Update waypoints being drawn by EnvRenderer.
        """
        from pyglet.gl import GL_POINTS  # import here to avoid issues if not rendering
        points = np.vstack(
            (
                self.waypoints[:, self.conf.wpt_xind],
                self.waypoints[:, self.conf.wpt_yind]
            )
        ).T
        scaled_points = 50.0 * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ('c3B/stream', [183, 193, 222])
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0
                ]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        Gets the current waypoint to follow.
        """
        wpts = np.vstack((waypoints[:, self.conf.wpt_xind],
                          waypoints[:, self.conf.wpt_yind])).T

        nearest_point_, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

        self._current_segment_index = i
        self._current_deviation = nearest_dist

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]

            # speed if we use it from csv
            # current_waypoint[2] = waypoints[i, self.conf.wpt_vind]

            # speed without csv, constant
            current_waypoint[2] = 6.0
            
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            # If close, but not within lookahead distance, reacquire the nearest waypoint.
            return np.append(
                wpts[i, :],
                waypoints[i, self.conf.wpt_vind]
            )
        else:
            # Too far from track
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        Computes the speed and steering angle command using pure pursuit.
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        # If no valid waypoint found, just slow down and go straight
        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position,
            lookahead_distance, self.wheelbase
        )
        # Scale speed
        speed = vgain * speed

        self._latest_speed = speed
        self._latest_steering_angle = steering_angle

        return speed, steering_angle
    
    def get_controls(self):
        """
        Return the most recently computed steering angle and speed.
        This is useful if you need these values outside the main plan loop.
        """
        return self._latest_steering_angle, self._latest_speed
    
    def get_current_segment_index(self):
        """
        Returns the index of the waypoint at the start of 
        the line segment we are projecting on.
        """
        return self._current_segment_index
    
    def get_path_deviation(self):
        """
        Returns the distance (in meters) from the car to the path.
        """
        return self._current_deviation