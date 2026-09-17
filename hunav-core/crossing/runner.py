import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from spec import load_spec, resolve
from metrics import pair_metrics, summarize
from occlusion import PATHS, OcclusionSpecResolver, summarize_visibility
from visibility import SENSOR, visibility, physx_static_raycast

parser = argparse.ArgumentParser()
parser.add_argument('scenarios', nargs='+')
parser.add_argument('--output', required=True)
parser.add_argument('--ros-control', action='store_true')
parser.add_argument('--fixed-control', action='store_true')
args = parser.parse_args()
if args.ros_control and len(args.scenarios) != 1:
    parser.error('ROS navigation currently accepts one episode per process')
root = Path('/work')
core = Path('/repo/hunav-core')
base = json.loads((root/'scene_config.json').read_text())
if base['scene'] != 'Habitat-GS scene55':
    raise ValueError('The Crossing scene binding requires scene55')
geometry = np.load(root/'output/geometry.npz')
clearance = np.load(root/'output/mesh_clearance.npz')['clearance']
origin, resolution = geometry['origin'], float(geometry['resolution'])
path = np.array(base['routes']['cross'])


def walkable(points, margin=0.):
    points = np.asarray(points)
    pixels = np.rint((points-origin)/resolution).astype(int)
    valid = ((pixels[:, 0] >= 0)&(pixels[:, 0] < clearance.shape[1])&
             (pixels[:, 1] >= 0)&(pixels[:, 1] < clearance.shape[0]))
    result = np.zeros(len(points), bool)
    result[valid] = ((geometry['walkable'][pixels[valid, 1], pixels[valid, 0]] > 0)&
                     (clearance[pixels[valid, 1], pixels[valid, 0]] >= margin))
    return result


specs = []
for filename in args.scenarios:
    spec = load_spec(filename)
    if spec['event']['type'] == 'occlusion':
        resolved = {'path': PATHS[spec['carter']['path']]}
        if not walkable(np.linspace(*resolved['path'], 201), .60).all():
            raise ValueError('Occlusion Carter path lacks body clearance')
        specs.append((Path(filename), spec, resolved))
        continue
    resolved = resolve(spec, path)
    human = resolved['person']
    if not walkable(np.linspace(path[0], path[-1], 201), .60).all():
        raise ValueError('Carter path lacks scene support or body clearance')
    if not walkable(np.linspace(human['start'], human['goal'], 101), .28).all():
        raise ValueError(f'{filename}: derived Person route intersects static geometry')
    specs.append((Path(filename), spec, resolved))

batch = Path(args.output).resolve()
if batch.parent != (root/'output').resolve():
    raise ValueError('Episode output must be inside /work/output on the data disk')
batch.mkdir(parents=True, exist_ok=False)
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
base_hash = hashlib.sha256((root/'scene_config.json').read_bytes()).hexdigest()

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,
                     'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']})
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
from omni.isaac.core import World
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from scipy.spatial.transform import Rotation
import omni.physx

sys.path.insert(0, str(core))
from hunav_client import HuNavClient

world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/120 if args.ros_control else 1/30)
stage = world.stage
mesh = UsdGeom.Mesh.Define(stage, '/World/Environment/CollisionMesh')
mesh.CreatePointsAttr(geometry['points'].astype(np.float32))
mesh.CreateFaceVertexCountsAttr(np.full(len(geometry['faces']), 3, np.int32))
mesh.CreateFaceVertexIndicesAttr(geometry['faces'].ravel())
mesh.CreateSubdivisionSchemeAttr('none')
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr(.005)
mesh.MakeInvisible()
forward = (path[-1]-path[0])/np.linalg.norm(path[-1]-path[0])
heading = math.atan2(forward[1], forward[0])
rotation = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
offset = np.array(base['navigation_center_offset'])
spawn = np.r_[path[0]-rotation@offset, .12]
quaternion = np.array([math.cos(heading/2), 0., 0., math.sin(heading/2)])
robot = world.scene.add(WheeledRobot(prim_path='/World/Carter', name='carter',
    wheel_dof_names=['joint_wheel_left', 'joint_wheel_right'], create_robot=True,
    usd_path=str(root/'assets/isaac-4.0/Isaac/Robots/Carter/nova_carter.usd'),
    position=spawn, orientation=quaternion))
controller = DifferentialController(name='crossing_drive', wheel_radius=.14, wheel_base=.3452000021934509 if args.ros_control else .834)
proxy = UsdGeom.Capsule.Define(stage, '/World/Person')
proxy.CreateAxisAttr('Z'); proxy.CreateRadiusAttr(.23); proxy.CreateHeightAttr(1.25)
translation = UsdGeom.Xformable(proxy).AddTranslateOp()
translation.Set(Gf.Vec3d(0., 0., -10.))
collision_api = UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
collision_enabled = collision_api.CreateCollisionEnabledAttr(False)
UsdPhysics.RigidBodyAPI.Apply(proxy.GetPrim()).CreateKinematicEnabledAttr(True)
proxy.MakeInvisible()
ros_link = None
if args.ros_control:
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension('omni.isaac.range_sensor')
    from omni.isaac.range_sensor import _range_sensor
    from carter_ros import CarterRosLink
    import omni.kit.commands
    ok, lidar = omni.kit.commands.execute('RangeSensorCreateLidar',
        path='/NavigationLidar', parent='/World/Carter/chassis_link',
        min_range=.1, max_range=20., draw_points=False, draw_lines=False,
        horizontal_fov=360., vertical_fov=4., horizontal_resolution=.5,
        vertical_resolution=2., rotation_rate=0., high_lod=True,
        yaw_offset=0., enable_semantics=False)
    if not ok:
        raise RuntimeError('Cannot create navigation LiDAR')
    lidar.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(-.23, 0., .6))
    ros_link = CarterRosLink(_range_sensor.acquire_lidar_sensor_interface(), str(lidar.GetPath()))
query = omni.physx.get_physx_scene_query_interface()
zero = controller.forward(np.zeros(2))
client = HuNavClient(core/'runtime/hunav.sock')
results = []


def carter_state():
    position, q = robot.get_world_pose()
    yaw = float(Rotation.from_quat(np.r_[q[1:], q[0]]).as_euler('xyz')[2])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    center = position[:2]+rot@offset
    velocity = robot.get_linear_velocity()[:2]
    angular = float(robot.get_angular_velocity()[2])
    center_velocity = velocity+angular*np.array([-(rot@offset)[1], (rot@offset)[0]])
    return {'x': float(center[0]), 'y': float(center[1]), 'z': float(position[2]),
            'heading': yaw, 'speed': float(np.linalg.norm(center_velocity)),
            'vx': float(center_velocity[0]), 'vy': float(center_velocity[1]), 'angular_velocity': angular}


def nominal(t, resolved, spec):
    def state(start, goal, speed, elapsed):
        delta = np.array(goal)-start
        length = np.linalg.norm(delta)
        d = min(max(elapsed, 0)*speed, length)
        xy = np.array(start)+delta*d/length
        velocity = delta/length*speed if 0 <= elapsed < length/speed else np.zeros(2)
        return {'x': float(xy[0]), 'y': float(xy[1]), 'vx': float(velocity[0]), 'vy': float(velocity[1]),
                'heading': math.atan2(delta[1], delta[0]), 'speed': float(np.linalg.norm(velocity)), 'present': elapsed >= 0}
    person = resolved['person']
    path = np.asarray(resolved['path'])
    return (state(path[0], path[-1], spec['carter']['speed'], t),
            state(person['start'], person['goal'], person['speed'], t-person['spawn_time']))


def run(filename, spec, resolved, index):
    episode = batch/f'{index:02d}_{filename.stem}'
    episode.mkdir()
    (episode/'scenario.yaml').write_text(yaml.safe_dump(spec,sort_keys=False))
    rows, nominal_rows = [], []
    error, reason = None, 'timeout'
    path = np.asarray(resolved['path'])
    forward = (path[-1]-path[0])/np.linalg.norm(path[-1]-path[0])
    heading = math.atan2(forward[1],forward[0])
    rotation = np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])
    spawn = np.r_[path[0]-rotation@offset,.12]
    quaternion = np.array([math.cos(heading/2),0.,0.,math.sin(heading/2)])
    collision_enabled.Set(False)
    translation.Set(Gf.Vec3d(0., 0., -10.))
    world.reset()
    robot.set_world_pose(spawn, quaternion)
    robot.set_linear_velocity(np.zeros(3)); robot.set_angular_velocity(np.zeros(3))
    controller.reset()
    robot.set_joint_velocities(np.zeros(robot.num_dof))
    for _ in range(360):
        robot.apply_wheel_actions(zero)
        world.step(render=False)
    initial_z = carter_state()['z']
    raycast = physx_static_raycast(query)
    if spec['event']['type'] == 'occlusion':
        resolved = OcclusionSpecResolver(raycast, walkable).resolve(spec, path)
    human = resolved['person']
    (episode/'scenario.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    if ros_link:
        nav_parameters = (core/'crossing/nav2_params.yaml').read_bytes()
        (episode/'nav2_params.yaml').write_bytes(nav_parameters)
    (episode/'resolved.json').write_text(json.dumps({'spec': spec, 'resolved': resolved, 'base_scene': base,
        'base_config_sha256': base_hash, 'source_sha256': source_hashes,
        'backend': 'Isaac Sim 4.0 PhysX + official HuNavSim Regular/SFM',
        'physics_dt': 1/120, 'state_dt': 1/30, 'rendering': False,
        'robot_control': ('Matched Fixed via ROS' if args.fixed_control else 'Nav2 DWB from PhysX LiDAR') if args.ros_control else 'fixed route baseline',
        'nav2_params_sha256': hashlib.sha256(nav_parameters).hexdigest() if ros_link else None}, indent=2))
    client.request(op='reset', human=human)
    nav_reply, nav_command = None, np.zeros(2)
    if ros_link:
        warmup_deadline = time.monotonic()+60.
        while True:
            for i in range(4):
                robot.apply_wheel_actions(zero)
                world.step(render=i==3)
            nav_command, nav_reply = ros_link.exchange(world.current_time, carter_state())
            if nav_reply['controller']['status'] == 'failed':
                raise RuntimeError(nav_reply['controller'])
            if nav_reply['controller']['status']=='running' and np.linalg.norm(nav_command)>.01:
                break
            if time.monotonic()>warmup_deadline:
                raise TimeoutError('Nav2 was not ready to drive within 60 seconds')
            time.sleep(1/30)
    start_time = world.current_time
    person = {'present': False, 'x': None, 'y': None, 'heading': None, 'speed': None,
              'vx': None, 'vy': None, 'arrived': False}
    done_since = None
    started = time.monotonic()
    sensor_output = (episode/'observations.jsonl').open('w') if ros_link else None
    try:
        with (episode/'trajectory.jsonl').open('w') as output:
            for step in range(int(spec['max_seconds']*120)+1):
                t = float(world.current_time-start_time)
                if abs(t-step/120) > 1e-5:
                    raise RuntimeError('Simulation clock drift')
                car = carter_state()
                if not all(math.isfinite(v) for v in car.values()) or abs(car['z']-initial_z) > .25:
                    raise RuntimeError('Carter pose invalid or lost floor contact')
                if step % 4 == 0:
                    if ros_link:
                        nav_command, nav_reply = ros_link.exchange(world.current_time, car)
                        if nav_reply['controller']['status'] == 'failed':
                            raise RuntimeError(nav_reply['controller'])
                        sensor_output.write(json.dumps({'timestamp': t, 'observation': ros_link.last_packet,
                                                        'control': nav_reply}, allow_nan=False)+'\n')
                    if t+1e-8 >= human['spawn_time']:
                        xy = np.array([person['x'], person['y']]) if person['present'] else np.array(human['start'])
                        obstacles = []
                        for angle in np.linspace(0, 2*math.pi, 32, endpoint=False):
                            direction = np.array([math.cos(angle), math.sin(angle), 0.])
                            ray_origin = np.r_[xy, .8]+.25*direction
                            hit = query.raycast_closest(tuple(ray_origin), tuple(direction), 3.)
                            if hit['hit'] and 'Environment/CollisionMesh' in hit.get('rigidBody', ''):
                                obstacles.append(list(hit['position'][:2]))
                        state = client.request(op='step', t=t,
                            robot={'xy': [car['x'], car['y']], 'velocity': [car['vx'], car['vy']],
                                   'yaw': car['heading'], 'angular_velocity': car['angular_velocity'], 'radius': .55},
                            obstacles=obstacles)
                        person = {'present': True, 'x': state['xy'][0], 'y': state['xy'][1],
                                  'heading': state['yaw'], 'speed': state['speed'], 'vx': state['velocity'][0],
                                  'vy': state['velocity'][1], 'arrived': state['arrived'],
                                  'service_ms': state['service_ms'], 'obstacle_samples': len(obstacles)}
                        translation.Set(Gf.Vec3d(person['x'], person['y'], .86))
                        collision_enabled.Set(True)
                    nc, nperson = nominal(t, resolved, spec)
                    row = {'timestamp': t, 'carter': car, 'person': dict(person),
                           'pair': pair_metrics(car, person), 'nominal_carter': nc, 'nominal_person': nperson}
                    if spec['event']['type'] == 'occlusion':
                        row['visibility'] = visibility(car, person, raycast)
                        row['nominal_visibility'] = visibility(nc, nperson, raycast)
                    if ros_link:
                        row['navigation'] = nav_reply
                    rows.append(row)
                    nominal_rows.append({'timestamp': t, 'carter': nc, 'person': nperson, 'pair': pair_metrics(nc, nperson)})
                    output.write(json.dumps(row, allow_nan=False)+'\n')
                    if step % 120 == 0:
                        output.flush()
                    goal_error = math.dist([car['x'], car['y']], path[-1])
                    complete = goal_error < .15 and car['speed'] < .05 and person['arrived']
                    done_since = (t if done_since is None else done_since) if complete else None
                    if done_since is not None and t-done_since >= 1.:
                        reason = 'both_goals_reached'
                        break
                if ros_link:
                    robot.apply_wheel_actions(controller.forward(nav_command))
                    world.step(render=step%4==3)
                    if step%4==3:
                        time.sleep(max(0., started+(step+1)/120-time.monotonic()))
                else:
                    # Keep Carter's controller independent of Person; HuNav receives its actual motion.
                    remaining = float((path[-1]-[car['x'], car['y']])@forward)
                    lateral = float(np.cross(forward, np.array([car['x'], car['y']])-path[0]))
                    desired = heading-math.atan2(lateral, 1.)
                    yaw_error = math.atan2(math.sin(desired-car['heading']), math.cos(desired-car['heading']))
                    linear = min(spec['carter']['speed'], max(0., .9*remaining))*max(0., math.cos(yaw_error)) if remaining > .07 else 0.
                    angular = float(np.clip(2.*yaw_error, -.6, .6))
                    robot.apply_wheel_actions(controller.forward(np.array([linear, angular])))
                    world.step(render=False)
    except Exception as exc:
        error, reason = repr(exc), 'runtime_error'
    finally:
        robot.apply_wheel_actions(zero)
        collision_enabled.Set(False)
        if sensor_output:
            sensor_output.close()
    metrics = summarize(rows, {'person': human}, path)
    nominal_metrics = summarize(nominal_rows, {'person': human}, path)
    active = [r for r in rows if r['person']['present']]
    hp = np.array([[r['person']['x'], r['person']['y']] for r in active])
    cp = np.array([[r['carter']['x'], r['carter']['y']] for r in rows])
    normal = forward
    if spec['event']['type'] == 'occlusion':
        direction = np.asarray(human['goal'])-human['start']
        normal = np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
    lateral = np.abs((hp-np.array(human['start']))@normal) if len(hp) else np.array([])
    moving = [r['person']['speed'] for r in active if r['timestamp'] > human['spawn_time']+1. and
              math.dist([r['person']['x'], r['person']['y']], human['goal']) > .5 and not r['person']['arrived']]
    cruise = [r['carter']['speed'] for r in rows if r['timestamp'] > 1. and math.dist([r['carter']['x'],r['carter']['y']],path[-1]) > 2.]
    metrics.update(episode=str(episode), scenario=filename.name, status='completed' if reason=='both_goals_reached' else 'failed',
        termination=reason, error=error, simulated_seconds=rows[-1]['timestamp'] if rows else 0.,
        wall_seconds=time.monotonic()-started, arrival_gap=spec['event'].get('arrival_gap'), person_speed=human['speed'],
        nominal={k:nominal_metrics[k] for k in ['minimum_distance_m','minimum_clearance_m','minimum_ttc_s','collision']},
        carter_mean_cruise_speed_mps=float(np.mean(cruise)) if cruise else None,
        person_max_lateral_deviation_m=float(lateral.max()) if len(lateral) else None,
        person_min_active_speed_mps=min(moving) if moving else None,
        person_on_walkable=bool(walkable(hp).all()) if len(hp) else False,
        carter_on_walkable=bool(walkable(cp).all()) if len(cp) else False,
        carter_goal_error_m=math.dist(cp[-1],path[-1]) if len(cp) else None,
        person_goal_reached=bool(active and active[-1]['person']['arrived']), frames=len(rows))
    if ros_link:
        metrics['navigation'] = nav_reply
    metrics['event_valid'] = metrics['actual_path_crossing'] and metrics['person_on_walkable'] and metrics['carter_on_walkable']
    if spec['event']['type'] == 'occlusion':
        metrics.update(summarize_visibility(rows, resolved))
        nominal_vis_rows = [dict(r, visibility=r['nominal_visibility'], person=r['nominal_person'],
                                carter=r['nominal_carter'],pair=pair_metrics(r['nominal_carter'],r['nominal_person'])) for r in rows]
        metrics['nominal_visibility'] = summarize_visibility(nominal_vis_rows, resolved)
        metrics['reaction_window_requested'] = spec['event']['reaction_window']
        metrics['reaction_window_resolved'] = resolved['reaction_window_resolved']
        metrics['event_valid'] = bool(metrics['event_valid'] and metrics['initially_occluded'] and
            metrics['first_visible_time'] is not None and metrics['occlusion_duration'] >= .1 and
            metrics['pre_visibility_out_of_fov_frames'] == 0)
        metrics['reaction_window_error'] = (metrics['reaction_window_actual']-spec['event']['reaction_window']
            if metrics['reaction_window_actual'] is not None else None)
    (episode/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print('EPISODE', json.dumps({k:v for k,v in metrics.items() if k not in ['crossing','band_definition','collision_definition','difficulty_note']}), flush=True)
    return metrics


try:
    for index, (filename, spec, resolved) in enumerate(specs):
        try:
            result = run(filename, spec, resolved, index)
        except Exception as exc:
            episode = batch/f'{index:02d}_{filename.stem}'
            episode.mkdir(exist_ok=True)
            result = {'scenario': filename.name, 'episode': str(episode), 'status': 'failed',
                      'termination': 'setup_or_save_error', 'error': repr(exc),
                      'actual_path_crossing': None, 'minimum_clearance_m': None,
                      'minimum_ttc_s': None, 'collision': None}
            if hasattr(exc, 'report'):
                result.update(termination='unresolvable_scenario', resolution_report=exc.report, event_valid=False)
            (episode/'metrics.json').write_text(json.dumps(result, indent=2))
            print('EPISODE_ERROR', repr(exc), flush=True)
        results.append(result)
        (batch/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
finally:
    if ros_link:
        ros_link.close()
    client.close()
    app.close()
