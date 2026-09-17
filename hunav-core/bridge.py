import json
import math
import signal
import socket
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, Pose
from hunav_msgs.msg import Agent, Agents
from hunav_msgs.srv import ComputeAgents

ROOT = Path('/work')
BIN = ROOT / 'ws/install/hunav_agent_manager/lib/hunav_agent_manager'


class Bridge:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('scene64_state_bridge')
        self.client = self.node.create_client(ComputeAgents, '/compute_agents')
        self.loader = subprocess.Popen([str(BIN / 'hunav_loader'), '--ros-args',
            '-p', 'yaml_base_name:=scene64', '-p', 'simulator:=Isaac Sim',
            '-p', 'publish_people:=false'], stdout=open(ROOT / 'runtime/loader.log', 'w'), stderr=subprocess.STDOUT)
        self.manager = None
        self.agent = None
        self.last_time = None
        self.arrived = False

    def stop_manager(self):
        if self.manager is not None:
            self.manager.terminate()
            try:
                self.manager.wait(5)
            except subprocess.TimeoutExpired:
                self.manager.kill()
                self.manager.wait()
            self.manager = None

    def reset(self, config):
        # Upstream reset leaves clock and behavior-tree state intact; restart for independent trials.
        self.stop_manager()
        self.manager = subprocess.Popen([str(BIN / 'hunav_agent_manager'), '--ros-args',
            '-p', 'behavior_tree_dir:=/work/behavior_trees', '-p', 'publish_tf:=false',
            '-p', 'publish_sfm_forces:=false'], stdout=open(ROOT / 'runtime/manager.log', 'w'), stderr=subprocess.STDOUT)
        time.sleep(.5)
        if not self.client.wait_for_service(timeout_sec=20):
            raise RuntimeError('HuNavSim /compute_agents unavailable; see manager.log')
        self.agent = Agent()
        a = self.agent
        a.id, a.type, a.name, a.group_id = 2, Agent.PERSON, 'LHM_walker', -1
        a.radius = float(config.get('radius', .23))
        a.desired_velocity = float(config.get('speed', .35))
        a.goal_radius = .12
        a.position.position.x, a.position.position.y = map(float, config['start'])
        a.yaw = float(config.get('yaw', 0.))
        a.position.orientation.z, a.position.orientation.w = math.sin(a.yaw/2), math.cos(a.yaw/2)
        g = Pose()
        g.position.x, g.position.y = map(float, config['goal'])
        g.orientation.w = 1.
        a.goals = [g]
        a.behavior.type = a.behavior.BEH_REGULAR
        a.behavior.configuration = a.behavior.BEH_CONF_CUSTOM
        a.behavior.goal_force_factor = 2.
        a.behavior.obstacle_force_factor = 10.
        a.behavior.social_force_factor = float(config.get('social_force_factor', 10.))
        a.behavior.other_force_factor = 1.
        self.last_time = None
        self.arrived = False
        self.config = config
        return {'ready': True, 'behavior': 'Regular', 'backend': 'official HuNavSim 2.0 + lightsfm'}

    def step(self, request):
        if self.agent is None:
            raise ValueError('reset is required before step')
        t = float(request['t'])
        if not math.isfinite(t) or t < 0 or (self.last_time is not None and not 0 < t-self.last_time <= .2):
            raise ValueError('time must increase by at most 0.2 simulation seconds')
        r = request['robot']
        values = list(r['xy']) + list(r['velocity']) + [r['yaw']]
        if len(r['xy']) != 2 or len(r['velocity']) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid robot state')
        if self.arrived:
            self.last_time = t
            return {**self.last_state, 't': t, 'service_ms': 0.}
        robot = Agent()
        robot.id, robot.type, robot.name, robot.group_id = 0, Agent.ROBOT, 'Carter', -1
        robot.radius = float(r.get('radius', .55))
        if not math.isfinite(robot.radius) or robot.radius <= 0.:
            raise ValueError('robot radius must be finite and positive')
        robot.position.position.x, robot.position.position.y = map(float, r['xy'])
        robot.yaw = float(r['yaw'])
        robot.position.orientation.z, robot.position.orientation.w = math.sin(robot.yaw/2), math.cos(robot.yaw/2)
        robot.velocity.linear.x, robot.velocity.linear.y = map(float, r['velocity'])
        robot.linear_vel = math.hypot(*r['velocity'])
        robot.velocity.angular.z = float(r.get('angular_velocity', 0.))
        self.agent.closest_obs = [Point(x=float(x), y=float(y), z=0.) for x, y in request.get('obstacles', [])]
        # Social force uses agent centers; obstacle force also needs the robot's physical boundary.
        dx = self.agent.position.position.x-robot.position.position.x
        dy = self.agent.position.position.y-robot.position.position.y
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            raise ValueError('human and robot centers overlap')
        self.agent.closest_obs.append(Point(
            x=robot.position.position.x+robot.radius*dx/distance,
            y=robot.position.position.y+robot.radius*dy/distance, z=0.))
        agents = Agents()
        agents.header.frame_id = 'map'
        nanoseconds = round(t*1e9)
        agents.header.stamp.sec, agents.header.stamp.nanosec = divmod(nanoseconds, 10**9)
        agents.agents = [self.agent]
        query = ComputeAgents.Request(current_agents=agents, robot=robot)
        start = time.perf_counter()
        future = self.client.call_async(query)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.)
        if not future.done() or future.exception() is not None:
            raise RuntimeError('HuNavSim response timeout or failure')
        updated = future.result().updated_agents.agents
        if len(updated) != 1 or updated[0].id != 2:
            raise RuntimeError('Unexpected agent response')
        a = self.agent = updated[0]
        self.last_time = t
        state = {'t': t, 'xy': [a.position.position.x, a.position.position.y],
                 'yaw': a.yaw, 'velocity': [a.velocity.linear.x, a.velocity.linear.y],
                 'speed': math.hypot(a.velocity.linear.x, a.velocity.linear.y),
                 'service_ms': (time.perf_counter()-start)*1000, 'behavior': int(a.behavior.type)}
        self.arrived = math.dist(state['xy'], self.config['goal']) <= .12
        state['arrived'] = self.arrived
        if self.arrived:
            # This single-destination scenario ends in standing; upstream removes the final SFM goal.
            state['velocity'], state['speed'] = [0., 0.], 0.
        if not all(math.isfinite(v) for v in state['xy'] + state['velocity'] + [state['yaw']]):
            raise RuntimeError('Non-finite HuNavSim state')
        self.last_state = state
        return state

    def close(self):
        self.stop_manager()
        self.loader.terminate()
        self.loader.wait(5)
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    endpoint = ROOT / 'runtime/hunav.sock'
    if endpoint.exists():
        raise RuntimeError(f'Existing endpoint: {endpoint}')
    bridge = Bridge()
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(endpoint))
            endpoint.chmod(0o666)
            server.listen(1)
            print('HUNAV_BRIDGE_READY', flush=True)
            while True:
                conn, _ = server.accept()
                with conn, conn.makefile('rwb') as stream:
                    while line := stream.readline(65537):
                        try:
                            if len(line) > 65536:
                                raise ValueError('request too large')
                            request = json.loads(line)
                            operation = request['op']
                            result = bridge.reset(request['human']) if operation == 'reset' else bridge.step(request) if operation == 'step' else None
                            if result is None:
                                raise ValueError('unknown operation')
                            response = {'ok': True, **result}
                        except Exception as exc:
                            response = {'ok': False, 'error': str(exc)}
                        stream.write((json.dumps(response, allow_nan=False)+'\n').encode())
                        stream.flush()
    finally:
        bridge.close()
        if endpoint.is_socket():
            endpoint.unlink()


if __name__ == '__main__':
    main()
