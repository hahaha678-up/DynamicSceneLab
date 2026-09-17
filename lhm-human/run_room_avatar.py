import argparse
import json
import math
import sys
import time
import types
import socket
import struct
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=35.)
parser.add_argument('--name', default='carter_scene64_lhm')
parser.add_argument('--record', action='store_true')
parser.add_argument('--live-avatar', action='store_true')
parser.add_argument('--walking', action='store_true')
parser.add_argument('--habitat', action='store_true')
parser.add_argument('--motion-preview', action='store_true')
args = parser.parse_args()
horizontal_fov = 100 if (args.walking or args.habitat) else 85
ROOT = Path('/work')
OUT = ROOT / 'output'
config = json.loads((ROOT / 'scene_config.json').read_text())
geometry = np.load(OUT / 'geometry.npz')

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 960, 'height': 600,
                     'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,
                     'renderer': 'RayTracedLighting',
                     'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']})

import cv2
import imageio.v2 as imageio
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdLux, PhysxSchema
from scipy.spatial.transform import Rotation
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera as IsaacCamera
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/30)
stage = world.stage
mesh = UsdGeom.Mesh.Define(stage, '/World/Environment/CollisionMesh')
mesh.CreatePointsAttr(geometry['points'].astype(np.float32))
mesh.CreateFaceVertexCountsAttr(np.full(len(geometry['faces']), 3, np.int32))
mesh.CreateFaceVertexIndicesAttr(geometry['faces'].ravel())
mesh.CreateSubdivisionSchemeAttr('none')
mesh.CreateDoubleSidedAttr(True)
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
collision.CreateContactOffsetAttr(.005)
collision.CreateRestOffsetAttr(0.)
# Geometry remains active in PhysX; its appearance is supplied by the GS pass.
mesh.MakeInvisible()
gs_prim = stage.DefinePrim('/World/Environment/GaussianBackground', 'Xform')
gs_prim.CreateAttribute('gs:asset', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(ROOT / 'assets/scene64/scene64.gs.ply')))
gs_prim.CreateAttribute('gs:transform', Sdf.ValueTypeNames.Matrix4d).Set(Gf.Matrix4d(np.array(config['gs_to_world']).T.tolist()))

route = np.array(config['route'])
heading = math.atan2(*(route[1]-route[0])[::-1])
center_offset = np.array(config['navigation_center_offset'])
rotation2 = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
spawn_xy = route[0] - rotation2 @ center_offset
axle_route = route - rotation2 @ center_offset
robot = world.scene.add(WheeledRobot(
    prim_path='/World/Carter', name='carter',
    wheel_dof_names=['joint_wheel_left', 'joint_wheel_right'], create_robot=True,
    usd_path=str(ROOT / 'assets/isaac-4.0/Isaac/Robots/Carter/nova_carter.usd'),
    position=np.r_[spawn_xy, .12], orientation=np.array([math.cos(heading/2), 0., 0., math.sin(heading/2)])))
controller = DifferentialController(name='carter_differential', wheel_radius=.14, wheel_base=.834)

human_position = np.array([1.0, 2.35, 0.0])
if args.walking or args.habitat:
    human_position = np.array([.25,1.85,0.])
human_proxy = UsdGeom.Capsule.Define(stage, '/World/LHMHuman/CollisionProxy')
human_proxy.CreateAxisAttr('Z')
human_proxy.CreateRadiusAttr(.23)
human_proxy.CreateHeightAttr(1.25)
human_translate = UsdGeom.Xformable(human_proxy).AddTranslateOp()
human_translate.Set(Gf.Vec3d(*(human_position + [0,0,.86])))
UsdPhysics.CollisionAPI.Apply(human_proxy.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(human_proxy.GetPrim()).CreateKinematicEnabledAttr(True)
human_proxy.MakeInvisible()

ambient = UsdLux.DomeLight.Define(stage, '/World/Lighting/Ambient')
ambient.CreateIntensityAttr(650.)
ambient.CreateColorAttr(Gf.Vec3f(1., .98, .95))
ambient.GetPrim().CreateAttribute('visibleInPrimaryRay', Sdf.ValueTypeNames.Bool).Set(False)
key = UsdLux.RectLight.Define(stage, '/World/Lighting/Ceiling')
key.CreateIntensityAttr(1200.); key.CreateWidthAttr(4.); key.CreateHeightAttr(4.)
UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(2., 2., 3.5))

width, height = 960, 600
camera = world.scene.add(IsaacCamera('/World/Cameras/Overview', name='overview', resolution=(width, height), frequency=30))
eye = np.array(config['overview_eye'])
target = np.array(config['overview_target'])
forward = target-eye; forward /= np.linalg.norm(forward)
right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
down = np.cross(forward, right)
c2w = np.eye(4); c2w[:3, :3] = np.column_stack([right, down, forward]); c2w[:3, 3] = eye
q = Rotation.from_matrix(c2w[:3, :3] @ np.diag([1., -1., -1.])).as_quat()
camera.set_world_pose(eye, np.r_[q[3], q[:3]], camera_axes='usd')
world.reset()
camera.initialize()
camera.set_focal_length(24.)
camera.set_horizontal_aperture(48*math.tan(math.radians(horizontal_fov)/2))
camera.set_vertical_aperture(camera.get_horizontal_aperture()*height/width)
camera.set_clipping_range(.02, 100.)
camera.add_distance_to_image_plane_to_frame()
print('ROBOT_DOFS', robot.dof_names, flush=True)
zero = controller.forward(np.array([0., 0.]))
for step in range(360):
    robot.apply_wheel_actions(zero)
    world.step(render=step % 4 == 0)
initial_position, initial_quat = robot.get_world_pose()
print('SETTLED', initial_position.tolist(), initial_quat.tolist(), flush=True)
stage.GetRootLayer().Export(str(OUT / (args.name + '.usda')))

sys.path.insert(0, '/repo/re3sim/gaussian_splatting')
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera as GSCamera
model = GaussianModel(0)
model.load_ply(str(ROOT / 'assets/scene64/scene64.gs.ply'))
pipe = types.SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
world_to_gs = np.linalg.inv(np.array(config['gs_to_world']))
w2c = np.linalg.inv(world_to_gs @ c2w)
gs_camera = GSCamera(0, w2c[:3, :3].T, w2c[:3, 3], math.radians(horizontal_fov),
                     2*math.atan(height/width*math.tan(math.radians(horizontal_fov)/2)),
                     torch.zeros(3, height, width, device='cuda'), None, 'overview', 0)
with torch.no_grad():
    background = render(gs_camera, model, pipe, torch.zeros(3, device='cuda'))['render']
    alpha = render(gs_camera, model, pipe, torch.zeros(3, device='cuda'), override_color=torch.ones_like(model.get_xyz))['render'][0]
    z = (model.get_xyz @ torch.tensor(w2c[:3, :3].T, device='cuda', dtype=torch.float32) + torch.tensor(w2c[:3, 3], device='cuda', dtype=torch.float32))[:, 2].clamp_min(0)
    depth = render(gs_camera, model, pipe, torch.zeros(3, device='cuda'), override_color=z[:, None].repeat(1, 3))['render'][0]/alpha.clamp_min(1e-6)
background = (background.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
alpha, depth = alpha.cpu().numpy(), depth.cpu().numpy()

human_frames = np.load('/repo/lhm-human/output/' + ('habitat_motion.npy' if args.habitat else ('walking_motion.npy' if args.walking else 'human_motion.npy')), mmap_mode='r')
assert human_frames.ndim == 3 and human_frames.shape[-1] == 14
avatar_socket = None
if args.live_avatar:
    avatar_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    avatar_socket.settimeout(30)
    avatar_socket.connect('/repo/lhm-human/output/avatar.sock')
static_count = len(model.get_xyz)
fields = ('_xyz', '_rotation', '_scaling', '_features_dc', '_features_rest', '_opacity')
static_fields = {field: getattr(model, field).detach().clone() for field in fields}
first = np.array(human_frames[0])
solid = first[:,13] > .5
assert solid.sum() > 100
human_origin = np.array([np.median(first[solid,0]), np.quantile(first[solid,1], .998), np.median(first[solid,2])])
# LHM motion is in a CV camera frame (Y down); Isaac world uses Z up.
human_to_world = np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
if args.walking or args.habitat:
    human_to_world = Rotation.from_euler('z', 90, degrees=True).as_matrix() @ human_to_world
human_to_gs = world_to_gs[:3,:3] @ human_to_world
transform_tensor = torch.tensor(human_to_gs.T, dtype=torch.float32, device='cuda')
translation_tensor = torch.tensor(world_to_gs[:3,:3] @ human_position + world_to_gs[:3,3], dtype=torch.float32, device='cuda')
origin_tensor = torch.tensor(human_origin, dtype=torch.float32, device='cuda')
rotation_xyzw = Rotation.from_matrix(human_to_gs).as_quat()
rotation_wxyz = torch.tensor(np.r_[rotation_xyzw[3],rotation_xyzw[:3]], dtype=torch.float32, device='cuda')
view_r = torch.tensor(w2c[:3,:3].T, dtype=torch.float32, device='cuda')
view_t = torch.tensor(w2c[:3,3], dtype=torch.float32, device='cuda')


def receive_exact(count):
    data = bytearray()
    while len(data) < count:
        chunk = avatar_socket.recv(count-len(data))
        if not chunk:
            raise ConnectionError('LHM driver disconnected during frame transfer')
        data.extend(chunk)
    return data


@torch.no_grad()
def render_human_frame(sim_seconds):
    # Ping-pong keeps this short, supplied motion continuous at replay boundaries.
    phase = int(sim_seconds * 30) % (2 * (len(human_frames)-1))
    frame_index = min(phase, 2*(len(human_frames)-1)-phase)
    if args.walking or args.habitat:
        frame_index = min(int(sim_seconds*30),len(human_frames)-1)
    if avatar_socket is not None:
        avatar_socket.sendall(struct.pack('<I', frame_index))
        size = struct.unpack('<I', receive_exact(4))[0]
        assert size == human_frames.shape[1]*14*4, 'Unexpected Gaussian payload size'
        frame_array = np.frombuffer(receive_exact(size), dtype=np.float32).reshape(-1,14).copy()
    else:
        frame_array = np.array(human_frames[frame_index])
    assert np.isfinite(frame_array).all(), 'Non-finite human Gaussian parameters'
    values = torch.from_numpy(frame_array).cuda()
    # Ground the retargeted clip; its camera-space translation is not a room floor constraint.
    if not args.habitat:
        origin_tensor[1] = float(np.quantile(frame_array[frame_array[:,13]>.5,1], .998))
    q = values[:,3:7]
    a, b, c, d = rotation_wxyz
    w, x, y, z = q.unbind(1)
    rotated_q = torch.stack((a*w-b*x-c*y-d*z, a*x+b*w+c*z-d*y,
                             a*y-b*z+c*w+d*x, a*z+b*y-c*x+d*w), dim=1)
    human_fields = {
        '_xyz': (values[:,:3]-origin_tensor) @ transform_tensor + translation_tensor,
        '_rotation': rotated_q,
        '_scaling': values[:,7:10].clamp_min(1e-8).log(),
        '_features_dc': ((values[:,10:13]-.5)/.28209479177387814)[:,None,:],
        '_features_rest': torch.zeros((len(values),0,3),device='cuda'),
        '_opacity': torch.logit(values[:,13:14].clamp(1e-6,1-1e-6)),
    }
    for field in fields:
        setattr(model, field, torch.cat((static_fields[field], human_fields[field]), dim=0))
    center_cv = np.median(frame_array[frame_array[:,13]>.5,:3], axis=0)
    center_world = human_to_world @ (center_cv - origin_tensor.cpu().numpy()) + human_position
    human_translate.Set(Gf.Vec3d(float(center_world[0]),float(center_world[1]),.86))
    rgb = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'))['render']
    opacity = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'), override_color=torch.ones_like(model.get_xyz))['render'][0]
    z = (model.get_xyz @ view_r + view_t)[:,2].clamp_min(0)
    distance = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'), override_color=z[:,None].repeat(1,3))['render'][0]/opacity.clamp_min(1e-6)
    return (rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8), opacity.cpu().numpy(), distance.cpu().numpy()

writer = imageio.get_writer(str(OUT / f'{args.name}.mp4'), fps=15, codec='libx264', quality=8,
                           macro_block_size=1, ffmpeg_log_level='error') if args.record else None
trajectory, waypoint, stopped_steps = [], 1, 0
aligning = False
wall_start = time.monotonic()
failure = None


def capture(index, center, speed, reached):
    background, alpha, depth = render_human_frame(index/120)
    rgb = np.asarray(camera.get_rgba())[:, :, :3].copy()
    mesh_depth = np.asarray(camera.get_current_frame()['distance_to_image_plane']).squeeze()
    valid = np.isfinite(mesh_depth) & (mesh_depth > .02) & (mesh_depth < 100.)
    visible = valid & ((alpha < .5) | (mesh_depth <= depth + .025))
    composite = np.where(visible[:, :, None], rgb, background)
    goal_view = c2w[:3, :3].T @ (np.r_[route[-1], -.03]-eye)
    focal = width/(2*math.tan(math.radians(horizontal_fov)/2))
    goal_pixel = tuple(np.rint([width/2+focal*goal_view[0]/goal_view[2],
                               height/2+focal*goal_view[1]/goal_view[2]]).astype(int))
    cv2.circle(composite, goal_pixel, 10, (60, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(composite, 'Goal', (goal_pixel[0]+14, goal_pixel[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (60, 220, 100), 1, cv2.LINE_AA)
    cv2.rectangle(composite, (12, 542), (750, 590), (28, 28, 28), -1)
    text = f'LHM human + Carter | goal {np.linalg.norm(route[-1]-center):.2f} m | {speed:.2f} m/s'
    cv2.putText(composite, text, (24, 573), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
    if writer is not None:
        writer.append_data(composite)
    if index == 0 or reached or index == 600:
        suffix = '_goal.png' if reached else ('_middle.png' if index else '_start.png')
        cv2.imwrite(str(OUT / (args.name + suffix)), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


try:
    for step in range(int(args.seconds*120)):
        position, quaternion = robot.get_world_pose()
        yaw = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_euler('xyz')[2]
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        center = position[:2] + rot @ center_offset
        if not np.isfinite(position).all() or abs(position[2]-initial_position[2]) > .3:
            raise RuntimeError('Robot pose invalid or lost floor contact')
        # Steering a point behind the axle can create positive feedback in turns.
        # Control the drive axle, while retaining the body center for clearance.
        delta = axle_route[waypoint]-position[:2]
        distance = np.linalg.norm(delta)
        if distance < .16 and waypoint < len(route)-1:
            waypoint += 1
            delta = axle_route[waypoint]-position[:2]
            distance = np.linalg.norm(delta)
        if waypoint == len(route)-1 and distance < .07:
            aligning = True
        desired = math.atan2(delta[1], delta[0])
        if aligning:
            desired = heading
        error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
        reached = aligning and np.linalg.norm(route[-1]-center) < .12 and abs(error) < .06
        linear = 0. if reached or aligning else min(.25, .8*distance)*max(0., math.cos(error))
        angular = 0. if reached else float(np.clip(1.5*error, -.45, .45))
        robot.apply_wheel_actions(controller.forward(np.array([linear, angular])))
        world.step(render=step % 8 == 0)
        speed = float(np.linalg.norm(robot.get_linear_velocity()[:2]))
        if step % 8 == 0:
            capture(step, center, speed, reached and stopped_steps == 0)
            trajectory.append({'t': step/120, 'position': position.tolist(), 'center': center.tolist(),
                               'yaw': float(yaw), 'speed': speed, 'waypoint': waypoint,
                               'goal_distance': float(np.linalg.norm(route[-1]-center)),
                               'wheel_velocity': robot.get_joint_velocities().tolist()})
        if reached:
            stopped_steps += 1
            if stopped_steps >= 240:
                break
        if step % 600 == 0:
            print('PROGRESS', step/120, center.tolist(), 'goal_distance', float(np.linalg.norm(route[-1]-center)), 'speed', speed, flush=True)
    if not stopped_steps and not args.motion_preview:
        failure = 'Goal not reached within time limit'
except Exception as exc:
    failure = repr(exc)
finally:
    if avatar_socket is not None:
        avatar_socket.close()
    robot.apply_wheel_actions(zero)
    if writer is not None:
        writer.close()
    goal_reached = bool(trajectory and stopped_steps >= 240 and trajectory[-1]['goal_distance'] < .12)
    preview_complete = args.motion_preview and len(trajectory) == len(range(0, int(args.seconds*120), 8))
    result = {'success': failure is None and (preview_complete or goal_reached),
              'mode': 'motion_preview' if args.motion_preview else 'navigation',
              'robot_goal_reached': goal_reached,
              'failure': failure, 'steps': len(trajectory), 'sim_seconds': step/120,
              'wall_seconds': time.monotonic()-wall_start,
              'initial_position': initial_position.tolist(),
              'final_goal_distance': trajectory[-1]['goal_distance'] if trajectory else None,
              'final_speed': trajectory[-1]['speed'] if trajectory else None,
              'trajectory': trajectory}
    (OUT / f'{args.name}_result.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({k: v for k, v in result.items() if k != 'trajectory'}), flush=True)
    app.close()
if not result['success']:
    raise SystemExit(2)
