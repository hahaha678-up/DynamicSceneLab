import math

from geometry_msgs.msg import Twist


class FixedSession:
    def __init__(self, node, route):
        self.node, self.route = node, route
        self.publisher = node.create_publisher(Twist, '/cmd_vel', 10)
        self.status = 'running'

    def tick(self):
        c = self.node.scenario_state
        start, goal = self.route
        length = math.dist(start, goal)
        fx, fy = (goal[0]-start[0])/length, (goal[1]-start[1])/length
        remaining = (goal[0]-c['x'])*fx+(goal[1]-c['y'])*fy
        lateral = fx*(c['y']-start[1])-fy*(c['x']-start[0])
        desired = math.atan2(fy, fx)-math.atan2(lateral, 1.)
        error = math.atan2(math.sin(desired-c['heading']), math.cos(desired-c['heading']))
        command = Twist()
        # The baseline only tracks its route and slows at its goal; it never reads the scan.
        command.linear.x = min(.8, max(0., .9*remaining))*max(0., math.cos(error)) if remaining>.07 else 0.
        command.angular.z = max(-.6, min(.6, 2.*error))
        if math.dist([c['x'], c['y']], goal)<.15 and c['speed']<.05:
            self.status = 'succeeded'
            command = Twist()
        self.publisher.publish(command)

    def summary(self):
        return {'status': self.status, 'detail': None, 'costmap_messages': 0, 'costmap_time': None}

    def close(self):
        pass
