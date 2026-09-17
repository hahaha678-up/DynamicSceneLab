import json
import math
import time
from pathlib import Path

from rclpy.action import ActionClient
from lifecycle_msgs.srv import GetState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath, OccupancyGrid
from nav2_msgs.action import FollowPath


class NavigationSession:
    def __init__(self, node, route, logfile):
        self.node, self.route = node, route
        self.log = Path(logfile).open('w')
        self.action = ActionClient(node, FollowPath, '/follow_path')
        self.state_client = node.create_client(GetState, '/controller_server/get_state')
        self.status, self.detail = 'waiting', None
        self.state_future, self.goal_future = None, None
        self.last_query = -100.
        self.costmaps = 0
        self.last_costmap_time = None
        node.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap, 10)
        node.create_subscription(RosPath, '/local_plan', self.local_plan, 10)

    def local_plan(self, msg):
        self.log.write(json.dumps({'kind': 'local_plan',
            'time': msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
            'frame': msg.header.frame_id,
            'points': [[p.pose.position.x, p.pose.position.y] for p in msg.poses]})+'\n')

    def costmap(self, msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        self.costmaps += 1
        self.last_costmap_time = stamp
        self.log.write(json.dumps({'kind': 'costmap', 'time': stamp, 'frame': msg.header.frame_id,
            'origin': [msg.info.origin.position.x, msg.info.origin.position.y],
            'resolution': msg.info.resolution, 'width': msg.info.width, 'height': msg.info.height,
            'data': list(msg.data)})+'\n')

    def tick(self):
        if self.status != 'waiting' or not self.costmaps:
            return
        if self.state_future and self.state_future.done():
            if self.state_future.result().current_state.id == 3 and self.action.server_is_ready():
                self.send_path()
                return
            self.state_future = None
        if self.state_future is None and time.monotonic()-self.last_query > .2:
            if self.state_client.service_is_ready():
                self.state_future = self.state_client.call_async(GetState.Request())
                self.last_query = time.monotonic()

    def send_path(self):
        goal = FollowPath.Goal()
        goal.controller_id, goal.goal_checker_id = 'FollowPath', 'goal_checker'
        path = RosPath()
        path.header.frame_id = 'odom'
        start, end = [list(p) for p in self.route]
        length = math.dist(start, end)
        yaw = math.atan2(end[1]-start[1], end[0]-start[0])
        for xy in [start, end]:
            xy[0] += .48*math.cos(yaw)
            xy[1] += .48*math.sin(yaw)
        count = max(2, math.ceil(length/.05)+1)
        for i in range(count):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = start[0]+i/(count-1)*(end[0]-start[0])
            p.pose.position.y = start[1]+i/(count-1)*(end[1]-start[1])
            p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            path.poses.append(p)
        goal.path = path
        self.status = 'pending'
        self.goal_future = self.action.send_goal_async(goal)
        self.goal_future.add_done_callback(self.accepted)

    def accepted(self, future):
        goal = future.result()
        if not goal.accepted:
            self.status, self.detail = 'failed', 'FollowPath goal rejected'
            return
        self.status = 'running'
        goal.get_result_async().add_done_callback(self.finished)

    def finished(self, future):
        result = future.result()
        self.status = 'succeeded' if result.status == 4 else 'failed'
        self.detail = {'action_status': result.status, 'result': str(result.result)}
        self.log.write(json.dumps({'kind': 'action_result', 'time': self.node.t,
                                  'status': self.status, 'detail': self.detail})+'\n')
        self.log.flush()

    def summary(self):
        return {'status': self.status, 'detail': self.detail, 'costmap_messages': self.costmaps,
                'costmap_time': self.last_costmap_time}

    def close(self):
        self.log.close()
