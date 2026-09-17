import math

SOURCE_STRIDE_METERS = 1.4698379039764404
SOURCE_CYCLE_SECONDS = 29/30
HUMAN_START_DELAY = 2.8
HUMAN = {'start': [.4, 1.9], 'goal': [3.4, 1.9], 'yaw': 0.,
         'speed': SOURCE_STRIDE_METERS/SOURCE_CYCLE_SECONDS, 'radius': .23,
         'social_force_factor': 10.}
ROUTES = {
    'stop': [[1.8, .6], [1.8, .9]],
    'cross': [[1.8, .6], [1.8, 3.0]],
    'bypass': [[1.8, .6], [.8, 1.0], [.8, 2.8]],
}
ROBOT_SPEED = .25
ROBOT_RADIUS = .55


def prescribed_robot(case, t):
    route = ROUTES[case]
    travel = t*ROBOT_SPEED
    for start, goal in zip(route[:-1], route[1:]):
        dx, dy = goal[0]-start[0], goal[1]-start[1]
        length = math.hypot(dx, dy)
        direction = [dx/length, dy/length]
        yaw = math.atan2(dy, dx)
        if travel < length:
            return {'xy': [start[i]+travel*direction[i] for i in range(2)],
                    'velocity': [ROBOT_SPEED*x for x in direction], 'yaw': yaw, 'radius': ROBOT_RADIUS}
        travel -= length
    return {'xy': goal, 'velocity': [0., 0.], 'yaw': yaw, 'radius': ROBOT_RADIUS}
