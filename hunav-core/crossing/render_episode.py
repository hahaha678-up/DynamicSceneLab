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
parser.add_argument('--episode', required=True)
parser.add_argument('--check-only', action='store_true')
parser.add_argument('--two-views', action='store_true')
parser.add_argument('--name', default='crossing_gap0_person1_episode')
args = parser.parse_args()
args.live_avatar = True
args.walking = True
args.habitat = True
args.case = 'cross'
horizontal_fov = 75
ROOT = Path('/work')
OUT = ROOT / 'output'
episode_path = Path(args.episode)
manifest = json.loads((episode_path / 'resolved.json').read_text())
rows = [json.loads(line) for line in (episode_path / 'trajectory.jsonl').read_text().splitlines()]
assert rows and all(b['timestamp'] > a['timestamp'] for a,b in zip(rows, rows[1:]))
config = manifest['base_scene']
config['route'] = manifest['resolved']['path']
HUMAN = manifest['resolved']['person']
HUMAN_START_DELAY = HUMAN['spawn_time']
event = manifest['spec']['event']
is_occlusion = event['type'] == 'occlusion'
episode_metrics = json.loads((episode_path/'metrics.json').read_text())
if args.two_views:
    config['overview_eye'] = [.3,6.,2.3]
    config['overview_target'] = [2.75,12.85,1.]
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
gs_prim.CreateAttribute('gs:asset', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(ROOT / config.get('gs_asset', 'assets/scene64/scene64.gs.ply'))))
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
human_position = np.r_[HUMAN['start'], 0.]
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


sys.path.insert(0, '/repo/re3sim/gaussian_splatting')
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera as GSCamera
model = GaussianModel(0)
model.load_ply(str(ROOT / config.get('gs_asset', 'assets/scene64/scene64.gs.ply')))
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

human_frames = np.load('/repo/lhm-human/output/social_reference.npy', mmap_mode='r')
walk = json.loads(Path('/repo/lhm-human/output/social_walk.json').read_text())
human_state = {'xy': HUMAN['start'], 'yaw': HUMAN['yaw'], 'speed': 0., 'velocity': [0.,0.],
               't': 0., 'service_ms': 0., 'behavior': 1, 'arrived': False}
walk_distance = 0.
gait_cycles = 0.
human_yaw = HUMAN['yaw']
import omni.physx
scene_query = omni.physx.get_physx_scene_query_interface()
assert human_frames.ndim == 3 and human_frames.shape[-1] == 14
avatar_socket = None
if args.live_avatar:
    avatar_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    avatar_socket.settimeout(30)
    avatar_socket.connect('/repo/lhm-human/output/social_avatar.sock')
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


cached_frame_index = None
cached_frame_array = None


def receive_exact(count):
    data = bytearray()
    while len(data) < count:
        chunk = avatar_socket.recv(count-len(data))
        if not chunk:
            raise ConnectionError('LHM driver disconnected during frame transfer')
        data.extend(chunk)
    return data


@torch.no_grad()
def render_scene():
    rgb = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'))['render']
    opacity = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'), override_color=torch.ones_like(model.get_xyz))['render'][0]
    z = (model.get_xyz @ view_r + view_t)[:,2].clamp_min(0)
    distance = render(gs_camera, model, pipe, torch.zeros(3,device='cuda'), override_color=z[:,None].repeat(1,3))['render'][0]/opacity.clamp_min(1e-6)
    return (rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8), opacity.cpu().numpy(), distance.cpu().numpy()


@torch.no_grad()
def render_human_frame(sim_seconds):
    global human_to_world, cached_frame_index, cached_frame_array
    if not human_present:
        for field in fields:
            setattr(model,field,static_fields[field])
        return render_scene()
    frame_index = int(gait_cycles*walk['cycle_frames']) % walk['cycle_frames']
    frame_index += walk['cycle_frames']*int(round(np.clip(human_state['speed']/.08,0.,1.)*100))
    if cached_frame_index == frame_index:
        frame_array = cached_frame_array
    elif avatar_socket is not None:
        avatar_socket.sendall(struct.pack('<I', frame_index))
        size = struct.unpack('<I', receive_exact(4))[0]
        assert size == human_frames.shape[1]*14*4, 'Unexpected Gaussian payload size'
        frame_array = np.frombuffer(receive_exact(size), dtype=np.float32).reshape(-1,14).copy()
    else:
        frame_array = np.array(human_frames[frame_index])
    cached_frame_index, cached_frame_array = frame_index, frame_array
    assert np.isfinite(frame_array).all(), 'Non-finite human Gaussian parameters'
    values = torch.from_numpy(frame_array).cuda()
    # Ground the retargeted clip; its camera-space translation is not a room floor constraint.
    origin_tensor[1] = float(np.quantile(frame_array[frame_array[:,13]>.5,1], .998))
    human_position[:2] = human_state['xy']
    human_to_world = Rotation.from_euler('z', 90+math.degrees(human_yaw), degrees=True).as_matrix() @ np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
    human_to_gs = world_to_gs[:3,:3] @ human_to_world
    transform_tensor = torch.tensor(human_to_gs.T, dtype=torch.float32, device='cuda')
    translation_tensor = torch.tensor(world_to_gs[:3,:3] @ human_position + world_to_gs[:3,3], dtype=torch.float32, device='cuda')
    rotation_xyzw = Rotation.from_matrix(human_to_gs).as_quat()
    rotation_wxyz = torch.tensor(np.r_[rotation_xyzw[3],rotation_xyzw[:3]], dtype=torch.float32, device='cuda')
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
    if np.linalg.norm(center_world[:2]-human_position[:2]) > .18:
        raise RuntimeError('Gaussian body center drifted from the collision proxy')
    return render_scene()


def set_view(view_name, carter):
    global gs_camera, view_r, view_t
    if view_name == 'Carter camera':
        sensor=manifest['resolved']['sensor']
        yaw=carter['heading']
        fwd=np.array([math.cos(yaw),math.sin(yaw),0.])
        view_eye=np.array([carter['x'],carter['y'],sensor['height_m']])+fwd*sensor['forward_offset_m']
        view_target=view_eye+fwd
        fov_x=math.radians(sensor['horizontal_fov_deg'])
        fov_y=math.radians(sensor['vertical_fov_deg'])
    else:
        view_eye=np.array(config['overview_eye'])
        view_target=np.array(config['overview_target'])
        fov_x=math.radians(horizontal_fov)
        fov_y=2*math.atan(height/width*math.tan(fov_x/2))
    fwd=view_target-view_eye;fwd/=np.linalg.norm(fwd)
    right=np.cross(fwd,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(fwd,right)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([right,down,fwd]);pose[:3,3]=view_eye
    quat=Rotation.from_matrix(pose[:3,:3]@np.diag([1.,-1.,-1.])).as_quat()
    view=np.linalg.inv(world_to_gs@pose)
    gs_camera=GSCamera(0,view[:3,:3].T,view[:3,3],fov_x,fov_y,
        torch.zeros(3,height,width,device='cuda'),None,view_name,0)
    view_r=torch.tensor(view[:3,:3].T,device='cuda',dtype=torch.float32)
    view_t=torch.tensor(view[:3,3],device='cuda',dtype=torch.float32)


writer = imageio.get_writer(str(OUT / f'{args.name}.mp4'), fps=15, codec='libx264', quality=8,
                           macro_block_size=1, ffmpeg_log_level='error')
wall_start = time.monotonic()
failure = None
frame_count = 0
max_pose_error = 0.
previous_person = None
previous_carter = None
wheel_angle = 0.
human_present = False
try:
    for index, row in enumerate(rows):
        person = row['person']
        human_present = person['present']
        if human_present:
            xy = np.array([person['x'], person['y']])
            if previous_person is not None:
                walk_distance += float(np.linalg.norm(xy-previous_person))
            previous_person = xy
            gait_cycles = walk_distance/walk['stride_m']
            human_state = {'xy': xy.tolist(), 'yaw': person['heading'], 'speed': person['speed']}
            human_yaw = person['heading']
            human_translate.Set(Gf.Vec3d(float(xy[0]),float(xy[1]),.86))
        carter = row['carter']
        center = np.array([carter['x'], carter['y']])
        if previous_carter is not None:
            wheel_angle += float(np.linalg.norm(center-previous_carter))/.14
        previous_carter = center
        if args.check_only and index != 330:
            continue
        if index % 2:
            continue
        yaw = carter['heading']
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        position = np.r_[center-rot@center_offset, carter['z']]
        quaternion = np.array([math.cos(yaw/2),0.,0.,math.sin(yaw/2)])
        robot.set_world_pose(position, quaternion)
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        joint_positions = robot.get_joint_positions()
        for dof in ['joint_wheel_left','joint_wheel_right']:
            joint_positions[robot.get_dof_index(dof)] = math.remainder(wheel_angle, 2*math.pi)
        robot.set_joint_positions(joint_positions)
        robot.set_joint_velocities(np.zeros_like(joint_positions))
        # Render without stepping physics so the saved interaction remains unchanged.
        omni.physx.get_physx_interface().update_transformations(False, True, True, False)
        world.render()
        world.render()
        observed, _ = robot.get_world_pose()
        max_pose_error = max(max_pose_error, float(np.linalg.norm(observed-position)))
        if max_pose_error > .001:
            raise RuntimeError('Replay robot pose differs from saved state')
        images=[]
        view_names=['Overview','Carter camera'] if args.two_views else ['Overview']
        for view_name in view_names:
            if args.two_views:
                set_view(view_name,carter)
            gs_rgb, gs_alpha, gs_depth = render_human_frame(row['timestamp'])
            rgb=np.asarray(camera.get_rgba())[:,:,:3].copy()
            mesh_depth=np.asarray(camera._custom_annotators['distance_to_image_plane'].get_data()).squeeze()
            assert rgb.shape==(height,width,3)
            valid=np.isfinite(mesh_depth)&(mesh_depth>.02)&(mesh_depth<100.)
            visible=valid&((gs_alpha<.5)|(mesh_depth<=gs_depth+.025))
            # The forward view excludes Carter's own body; the USD camera stays fixed for the overview.
            composite=gs_rgb.copy() if view_name=='Carter camera' else np.where(visible[:,:,None],rgb,gs_rgb)
            cv2.rectangle(composite,(10,10),(950,76),(28,28,28),-1)
            title=f"{event['type'].title()} | {view_name} | t = {row['timestamp']:.2f} s"
            cv2.putText(composite,title,(24,36),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
            if is_occlusion:
                status=('NOT SPAWNED' if not human_present else 'VISIBLE' if row['visibility']['visible'] else
                        'OUT OF VIEW' if row['visibility']['in_fov_samples']==0 else 'OCCLUDED')
                first=episode_metrics['first_visible_time']
                detail=f"Carter ray visibility: {status} | first visible {first:.2f} s"
                color=(100,230,140) if status=='VISIBLE' else (255,210,90)
            else:
                detail=f"Nominal arrival gap {event['arrival_gap']:g} s"
                color=(240,240,240)
            cv2.putText(composite,detail,(24,62),cv2.FONT_HERSHEY_SIMPLEX,.54,color,1,cv2.LINE_AA)
            cv2.rectangle(composite,(10,542),(950,590),(28,28,28),-1)
            if human_present:
                pair=row['pair'];ttc='none' if pair['ttc'] is None else f"{pair['ttc']:.2f}s"
                label=f"Carter {carter['speed']:.2f} m/s | Person {person['speed']:.2f} m/s | Gap {pair['clearance']:.2f} m | TTC {ttc}"
            else:
                label=f"Carter {carter['speed']:.2f} m/s | Person spawns hidden at {HUMAN_START_DELAY:.2f} s"
            cv2.putText(composite,label,(24,573),cv2.FONT_HERSHEY_SIMPLEX,.56,(240,240,240),1,cv2.LINE_AA)
            images.append(composite)
        composite=np.concatenate(images,axis=1)
        writer.append_data(composite)
        frame_count += 1
        if frame_count % 45 == 0:
            print('REPLAY',frame_count,row['timestamp'],flush=True)
except Exception as exc:
    failure=repr(exc)
finally:
    if avatar_socket is not None:
        avatar_socket.close()
    writer.close()
    import hashlib
    result={'success':failure is None and frame_count == (1 if args.check_only else len(rows[::2])),
            'mode':'saved_episode_replay','source_episode':str(episode_path),
            'trajectory_sha256':hashlib.sha256((episode_path/'trajectory.jsonl').read_bytes()).hexdigest(),
            'views':2 if args.two_views else 1,'frames':frame_count,'fps':15,'duration_s':frame_count/15,
            'max_robot_pose_error_m':max_pose_error,'failure':failure,
            'wall_seconds':time.monotonic()-wall_start}
    (OUT/f'{args.name}_result.json').write_text(json.dumps(result,indent=2))
    print('REPLAY_RESULT',json.dumps(result),flush=True)
    app.close()
