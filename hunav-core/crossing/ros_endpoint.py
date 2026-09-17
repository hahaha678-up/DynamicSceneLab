import argparse
import json
import math
import os
import select
import socket
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


class CarterEndpoint(Node):
    def __init__(self):
        super().__init__('carter_sim_endpoint')
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, '/cmd_vel', self.command, 10)
        self.t = 0.
        self.cmd = [0., 0.]
        self.cmd_t = -100.
        self.cmd_wall = -100.
        self.received = 0

    def command(self, msg):
        values = [msg.linear.x, msg.angular.z]
        if not all(math.isfinite(v) for v in values):
            self.cmd, self.cmd_t = [0., 0.], -100.
            return
        self.cmd = [max(-.8, min(.8, values[0])), max(-.6, min(.6, values[1]))]
        self.cmd_t, self.cmd_wall = self.t, time.monotonic()
        self.received += 1

    def publish_state(self, data):
        self.t = float(data['time'])
        ns = round(self.t*1e9)
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = divmod(ns, 1000000000)
        self.clock_pub.publish(clock)
        self.scenario_state = data['carter']
        c = dict(data['carter'])
        co, si = math.cos(c['heading']), math.sin(c['heading'])
        # DWB integrates a differential drive about its axle, while scenario metrics use the body center.
        c['x'] += .48*co
        c['y'] += .48*si
        c['vx'] -= .48*si*c['angular_velocity']
        c['vy'] += .48*co*c['angular_velocity']
        scan = LaserScan()
        scan.header.stamp = clock.clock
        scan.header.frame_id = 'laser'
        scan.angle_min = float(data['angle_min'])
        scan.angle_increment = float(data['angle_increment'])
        scan.angle_max = scan.angle_min+(len(data['ranges'])-1)*scan.angle_increment
        scan.range_min, scan.range_max = .1, 20.
        scan.scan_time, scan.time_increment = 1/30, 0.
        scan.ranges = [float(v) if v is not None else math.inf for v in data['ranges']]
        self.scan_pub.publish(scan)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = clock.clock, 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = c['x'], c['y']
        odom.pose.pose.position.z = c['z']
        odom.pose.pose.orientation.z = math.sin(c['heading']/2)
        odom.pose.pose.orientation.w = math.cos(c['heading']/2)
        co, si = math.cos(c['heading']), math.sin(c['heading'])
        odom.twist.twist.linear.x = co*c['vx']+si*c['vy']
        odom.twist.twist.linear.y = -si*c['vx']+co*c['vy']
        odom.twist.twist.angular.z = c['angular_velocity']
        self.odom_pub.publish(odom)
        body = TransformStamped()
        body.header, body.child_frame_id = odom.header, 'base_link'
        body.transform.translation.x = c['x']
        body.transform.translation.y = c['y']
        body.transform.translation.z = c['z']
        body.transform.rotation = odom.pose.pose.orientation
        laser = TransformStamped()
        laser.header.stamp, laser.header.frame_id = clock.clock, 'base_link'
        laser.child_frame_id = 'laser'
        laser.transform.translation.x = -.23
        laser.transform.translation.z = .6
        laser.transform.rotation.w = 1.
        self.tf.sendTransform([body, laser])

    def reply(self):
        stale = self.t-self.cmd_t > .35 or time.monotonic()-self.cmd_wall > .5
        return {'ok': True, 'cmd_vel': [0., 0.] if stale else self.cmd,
                'command_stale': stale, 'received_commands': self.received}


class InterfaceProbe(Node):
    def __init__(self):
        super().__init__('carter_interface_probe')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Clock, '/clock', self.clock, 10)
        self.create_subscription(LaserScan, '/scan', self.scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom, 10)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.counts = {'scan': 0, 'odom': 0, 'clock': 0}
        self.times = []
        self.front_ranges = []
        self.poses = []
        self.failures = []

    def clock(self, msg):
        t = msg.clock.sec+msg.clock.nanosec*1e-9
        self.counts['clock'] += 1
        # The last phase deliberately stops publishing to test the command watchdog.
        if t >= 9.:
            return
        command = Twist()
        if 2. <= t < 4.:
            command.linear.x = .4
        elif 5. <= t < 7.:
            command.angular.z = .3
        elif 8. <= t < 9.:
            command.linear.x = .3
        self.publisher.publish(command)

    def scan(self, msg):
        self.counts['scan'] += 1
        t = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.times.append(t)
        if msg.header.frame_id != 'laser' or len(msg.ranges) != 720:
            self.failures.append('Invalid scan frame or size')
        if any(math.isnan(v) for v in msg.ranges):
            self.failures.append('NaN scan range')
        if 1. < t < 2.:
            i = round(-msg.angle_min/msg.angle_increment)
            self.front_ranges.append(min(msg.ranges[i-2:i+3]))

    def odom(self, msg):
        self.counts['odom'] += 1
        self.poses.append([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            self.failures.append('Invalid odometry frames')

    def report(self):
        connected = self.buffer.can_transform('odom', 'laser', rclpy.time.Time())
        gaps = [b-a for a,b in zip(self.times, self.times[1:])]
        front = sum(self.front_ranges)/len(self.front_ranges) if self.front_ranges else None
        checks = {'topics_received': min(self.counts.values()) > 250,
                  'tf_odom_to_laser': connected,
                  'scan_30hz': bool(gaps) and max(abs(g-1/30) for g in gaps)<1e-6,
                  'front_person_range': front is not None and abs(front-2.018)<.04,
                  'message_contract': not self.failures}
        return {'success': all(checks.values()), 'checks': checks, 'counts': self.counts,
                'front_person_range_m': front, 'failures': self.failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--nav2', action='store_true')
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--route')
    parser.add_argument('--nav-log')
    args = parser.parse_args()
    endpoint = Path('/work/runtime/carter_ros.sock')
    if endpoint.exists():
        raise RuntimeError('Socket already exists; verify its owner before removing it')
    rclpy.init()
    node = CarterEndpoint()
    if sum([args.verify, args.nav2, args.fixed])>1:
        raise ValueError('The test publisher cannot run with Nav2')
    navigation = None
    if args.nav2:
        from nav2_session import NavigationSession
        navigation = NavigationSession(node, json.loads(args.route), args.nav_log)
    if args.fixed:
        from fixed_session import FixedSession
        navigation = FixedSession(node, json.loads(args.route))
    probe = InterfaceProbe() if args.verify else None
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    if probe:
        executor.add_node(probe)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)
    server.settimeout(90.)
    print('ROS_ENDPOINT_READY', flush=True)
    try:
        connection, _ = server.accept()
        with connection, connection.makefile('rwb') as stream:
            while True:
                line = stream.readline(100001)
                if not line:
                    break
                if len(line)>100000:
                    raise ValueError('Oversized sensor packet')
                data = json.loads(line)
                if data.get('op') == 'close':
                    break
                node.publish_state(data)
                if navigation:
                    navigation.tick()
                until = time.monotonic()+.008
                while time.monotonic()<until:
                    executor.spin_once(timeout_sec=.0005)
                reply = node.reply()
                if navigation:
                    reply['controller'] = navigation.summary()
                    reply['controller_mode'] = 'nav2' if args.nav2 else 'fixed'
                    if args.nav2:
                        reply['nav2'] = reply['controller']
                    if navigation.status != 'running':
                        reply['cmd_vel'] = [0., 0.]
                stream.write((json.dumps(reply, allow_nan=False)+'\n').encode())
                stream.flush()
        if probe:
            for _ in range(30):
                executor.spin_once(timeout_sec=.002)
            report = probe.report()
            Path(args.report).write_text(json.dumps(report, indent=2))
            print('ROS_INTERFACE_REPORT', json.dumps(report), flush=True)
    finally:
        if navigation:
            navigation.close()
        server.close()
        endpoint.unlink(missing_ok=True)
        executor.shutdown()
        node.destroy_node()
        if probe:
            probe.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
