import json
import socket
import numpy as np


class CarterRosLink:
    def __init__(self, sensor, lidar_path, endpoint='/repo/hunav-core/runtime/carter_ros.sock'):
        self.sensor, self.lidar_path = sensor, lidar_path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(2.)
        self.socket.connect(endpoint)
        self.stream = self.socket.makefile('rwb')

    def exchange(self, timestamp, carter):
        angles = np.asarray(self.sensor.get_azimuth_data(self.lidar_path))
        elevation = np.asarray(self.sensor.get_zenith_data(self.lidar_path))
        depth = np.asarray(self.sensor.get_linear_depth_data(self.lidar_path))
        if depth.shape != (len(angles), len(elevation)) or len(angles)<2:
            raise RuntimeError('LiDAR has no complete scan')
        central = int(np.argmin(np.abs(elevation)))
        if abs(elevation[central])>.001 or not np.allclose(np.diff(angles), np.diff(angles)[0], atol=1e-6):
            raise RuntimeError('LiDAR scan is not a uniformly sampled horizontal plane')
        ranges = depth[:, central]
        packet = {'time': float(timestamp), 'carter': carter,
                  'angle_min': float(angles[0]), 'angle_increment': float(angles[1]-angles[0]),
                  'ranges': [float(v) if np.isfinite(v) and .1<=v<19.999 else None for v in ranges]}
        self.last_packet = packet
        # Only sensor measurements and ego odometry cross into the navigation process.
        self.stream.write((json.dumps(packet, allow_nan=False)+'\n').encode())
        self.stream.flush()
        line = self.stream.readline(4097)
        if not line or len(line)>4096:
            raise ConnectionError('ROS control endpoint disconnected')
        reply = json.loads(line)
        if not reply.get('ok'):
            raise RuntimeError('ROS endpoint rejected sensor packet')
        command = np.asarray(reply['cmd_vel'], dtype=float)
        if command.shape!=(2,) or not np.isfinite(command).all():
            raise RuntimeError('Invalid velocity command')
        return command, reply

    def close(self):
        self.stream.close()
        self.socket.close()
