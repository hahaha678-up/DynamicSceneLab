import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from demo_tracks import Track
from metrics import pair_metrics, summarize
from visibility import SENSOR, visibility, physx_static_raycast

parser = argparse.ArgumentParser()
parser.add_argument('scenario')
parser.add_argument('--output', required=True)
args = parser.parse_args()
root, core = Path('/work'), Path('/repo/hunav-core')
spec = json.loads(Path(args.scenario).read_text())
tracks = [Track(t) for t in spec['tracks']]
path = np.asarray(spec['carter']['route'], dtype=float)
if path.shape != (2, 2) or np.linalg.norm(path[1]-path[0]) < 1:
    raise ValueError('Carter route needs two distinct positions')
episode = Path(args.output)
episode.mkdir(exist_ok=False)
base = json.loads((root/'scene_config.json').read_text())
geometry = np.load(root/'output/geometry.npz')
clearance = np.load(root/'output/mesh_clearance.npz')['clearance']


def walkable(points, margin=0.):
    p = np.rint((np.asarray(points)-geometry['origin'])/float(geometry['resolution'])).astype(int)
    good = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < clearance.shape[1]) & (p[:, 1] < clearance.shape[0])
    result = np.zeros(len(p), bool)
    result[good] = (geometry['walkable'][p[good, 1], p[good, 0]] > 0) & (clearance[p[good, 1], p[good, 0]] >= margin)
    return result


if not walkable(np.linspace(*path, 401), .85).all():
    raise ValueError('Carter route lacks configured navigation footprint clearance')
for track in tracks:
    if not walkable(track.sampled_path(), .23).all():
        raise ValueError('Person track leaves supported walking area')
resolved = {'path': path.tolist(), 'person': tracks[0].resolved(),
            'people': [t.resolved() for t in tracks], 'sensor': SENSOR}
nav_bytes = (core/'crossing/nav2_params.yaml').read_bytes()
(episode/'nav2_params.yaml').write_bytes(nav_bytes)
spec.update(motion_mode='trajectory_replay', controller_mode='nav2',
            proxy={'carter_radius_m': .55, 'person_radius_m': .23})
(episode/'scenario.json').write_text(json.dumps(spec, indent=2))
manifest = {'spec': dict(spec, event={'type': spec['event_type']}), 'resolved': resolved,
            'base_scene': base, 'backend': 'Isaac Sim 4.0 / PhysX / trajectory replay',
            'motion_mode': 'trajectory_replay', 'proxy': spec['proxy'],
            'physics_dt': 1/120, 'state_dt': 1/30,
            'robot_control': 'Nav2 DWB from PhysX LiDAR',
            'source_sha256': {n: hashlib.sha256((core/'crossing'/n).read_bytes()).hexdigest()
                              for n in ['demo_run.py', 'demo_tracks.py', 'carter_ros.py']},
            'nav2_params_sha256': hashlib.sha256(nav_bytes).hexdigest()}
(episode/'resolved.json').write_text(json.dumps(manifest, indent=2))

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,
                     'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']})
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
from omni.isaac.core import World
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.utils.extensions import enable_extension
from scipy.spatial.transform import Rotation
import omni.physx
import omni.kit.commands
enable_extension('omni.isaac.range_sensor')
from omni.isaac.range_sensor import _range_sensor
from carter_ros import CarterRosLink

world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/120)
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
offset = np.asarray(base['navigation_center_offset'])
forward = (path[1]-path[0])/np.linalg.norm(path[1]-path[0])
heading = math.atan2(forward[1], forward[0])
rot = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
robot = world.scene.add(WheeledRobot(prim_path='/World/Carter', name='carter',
    wheel_dof_names=['joint_wheel_left', 'joint_wheel_right'], create_robot=True,
    usd_path=str(root/'assets/isaac-4.0/Isaac/Robots/Carter/nova_carter.usd'),
    position=np.r_[path[0]-rot@offset, .12], orientation=np.array([math.cos(heading/2), 0., 0., math.sin(heading/2)])))
controller = DifferentialController(name='demo_drive', wheel_radius=.14, wheel_base=.3452000021934509)
actors = []
for i in range(len(tracks)):
    proxy = UsdGeom.Capsule.Define(stage, f'/World/Person_{i}')
    proxy.CreateAxisAttr('Z'); proxy.CreateRadiusAttr(.23); proxy.CreateHeightAttr(1.25)
    translation = UsdGeom.Xformable(proxy).AddTranslateOp()
    translation.Set(Gf.Vec3d(0., 0., -10.))
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(proxy.GetPrim()).CreateKinematicEnabledAttr(True)
    proxy.MakeInvisible()
    actors.append((proxy, translation))
ok, lidar = omni.kit.commands.execute('RangeSensorCreateLidar',
    path='/NavigationLidar', parent='/World/Carter/chassis_link', min_range=.1, max_range=20.,
    draw_points=False, draw_lines=False, horizontal_fov=360., vertical_fov=4.,
    horizontal_resolution=.5, vertical_resolution=2., rotation_rate=0., high_lod=True,
    yaw_offset=0., enable_semantics=False)
if not ok:
    raise RuntimeError('Cannot create LiDAR')
lidar.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(-.23, 0., .6))
query = omni.physx.get_physx_scene_query_interface()
world.reset()
zero = controller.forward(np.zeros(2))
for _ in range(360):
    robot.apply_wheel_actions(zero); world.step(render=False)


def carter_state():
    position, q = robot.get_world_pose()
    yaw = float(Rotation.from_quat(np.r_[q[1:], q[0]]).as_euler('xyz')[2])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    off = rotation@offset
    xy = position[:2]+off
    angular = float(robot.get_angular_velocity()[2])
    v = robot.get_linear_velocity()[:2]+angular*np.array([-off[1], off[0]])
    return dict(x=float(xy[0]), y=float(xy[1]), z=float(position[2]), heading=yaw,
                speed=float(np.linalg.norm(v)), vx=float(v[0]), vy=float(v[1]), angular_velocity=angular)


def move_people(t):
    states = []
    for track, (proxy, translation) in zip(tracks, actors):
        state = track.state(t)
        if state['present']:
            floors = []
            def hit(h):
                if 'Environment/CollisionMesh' in str(h.rigid_body) and h.normal[2] > .4:
                    floors.append(h.position[2])
                return True
            query.raycast_all((state['x'], state['y'], .4), (0., 0., -1.), 1.5, hit)
            if not floors:
                raise RuntimeError('Person has no floor support')
            state['floor_z'] = float(max(floors))
            translation.Set(Gf.Vec3d(state['x'], state['y'], state['floor_z']+.865))
        else:
            translation.Set(Gf.Vec3d(0., 0., -10.))
        states.append(state)
    return states


ros = CarterRosLink(_range_sensor.acquire_lidar_sensor_interface(), str(lidar.GetPath()))
rows, error, reason = [], None, 'timeout'
initial_z = carter_state()['z']
raycast = physx_static_raycast(query)
started = time.monotonic()
try:
    deadline = started+60.
    while True:
        for i in range(4):
            robot.apply_wheel_actions(zero); world.step(render=i==3)
        command, reply = ros.exchange(world.current_time, carter_state())
        if reply['controller']['status'] == 'failed' or time.monotonic() > deadline:
            raise RuntimeError('Nav2 startup failed')
        if reply['controller']['status'] == 'running' and np.linalg.norm(command) > .01:
            break
        time.sleep(1/30)
    start_time, pacing = world.current_time, time.monotonic()
    states = move_people(0.)
    world.render()
    with (episode/'trajectory.jsonl').open('w') as out, (episode/'observations.jsonl').open('w') as obs:
        for step in range(round(spec.get('max_seconds', 45)*120)+1):
            t = float(world.current_time-start_time)
            if abs(t-step/120) > 1e-5:
                raise RuntimeError('Simulation clock drift')
            if step % 4 == 0:
                car = carter_state()
                if abs(car['z']-initial_z) > .25 or not all(math.isfinite(v) for v in car.values()):
                    raise RuntimeError('Invalid Carter pose')
                command, reply = ros.exchange(world.current_time, car)
                row = {'timestamp': t, 'carter': car, 'person': states[0], 'people': states,
                       'pair': pair_metrics(car, states[0]), 'navigation': reply,
                       'visibility': visibility(car, states[0], raycast)}
                rows.append(row)
                out.write(json.dumps(row, allow_nan=False)+'\n')
                obs.write(json.dumps({'timestamp': t, 'observation': ros.last_packet, 'control': reply}, allow_nan=False)+'\n')
                if step % 120 == 0:
                    out.flush(); obs.flush()
                if reply['controller']['status'] == 'failed':
                    reason = 'nav_abort'; break
                if reply['controller']['status'] == 'succeeded' and t >= spec.get('minimum_seconds', 0):
                    reason = 'goal_reached'; break
            states = move_people((step+1)/120)
            robot.apply_wheel_actions(controller.forward(command))
            world.step(render=step%4==3)
            if step%4==3:
                time.sleep(max(0., pacing+(step+1)/120-time.monotonic()))
except Exception as exc:
    error, reason = repr(exc), 'runtime_error'
finally:
    robot.apply_wheel_actions(zero)
    ros.close()
    metrics = summarize(rows, resolved, path)
    metrics.update(status='completed' if reason=='goal_reached' else 'failed', termination=reason,
                   error=error, frames=len(rows), simulated_seconds=rows[-1]['timestamp'] if rows else 0.,
                   wall_seconds=time.monotonic()-started,
                   carter_on_walkable=bool(rows and walkable([[r['carter']['x'], r['carter']['y']] for r in rows]).all()),
                   person_on_walkable=True, navigation=rows[-1]['navigation'] if rows else None)
    (episode/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('DEMO_RESULT', json.dumps(metrics), flush=True)
    app.close()
if error:
    raise SystemExit(error)
