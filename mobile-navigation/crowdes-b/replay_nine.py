import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
B = Path('/work/crowdes-b')
dataset = B/'datasets/clean40_v1'
manifest = json.loads((dataset/'manifest.json').read_text())
assert manifest['accepted'] == 9, 'Replay is bound to the nine existing accepted trajectories'
inputs = []
for item in manifest['samples']:
    source = dataset/item['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    frame = pd.read_csv(source)
    assert np.allclose(frame.time_s, frame.frame/25., atol=1e-9)
    assert (np.diff(frame.time_s)>0).all()
    inputs.append((item, frame))
out = B/'runs'/args.run
out.mkdir(exist_ok=False)
(out/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
geometry = np.load('/work/output/geometry.npz')

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
                     'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,
                     'renderer': 'RayTracedLighting',
                     'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']})
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, UsdLux
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation
import omni.physx
import cv2
import imageio.v2 as imageio

world = World(stage_units_in_meters=1., physics_dt=.01, rendering_dt=.01)
stage = world.stage
mesh = UsdGeom.Mesh.Define(stage, '/World/Environment/CollisionMesh')
mesh.CreatePointsAttr(geometry['points'].astype(np.float32))
mesh.CreateFaceVertexCountsAttr(np.full(len(geometry['faces']), 3, np.int32))
mesh.CreateFaceVertexIndicesAttr(geometry['faces'].ravel())
mesh.CreateSubdivisionSchemeAttr('none')
mesh.CreateDoubleSidedAttr(True)
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr(.005)
mesh.MakeInvisible()

# The visual cutaway removes the ceiling only; PhysX retains the complete collision mesh.
visible = UsdGeom.Mesh.Define(stage, '/World/RoomCutaway')
faces = geometry['faces']
faces = faces[geometry['points'][faces, 2].mean(axis=1)<1.5]
visible.CreatePointsAttr(geometry['points'].astype(np.float32))
visible.CreateFaceVertexCountsAttr(np.full(len(faces), 3, np.int32))
visible.CreateFaceVertexIndicesAttr(faces.ravel())
visible.CreateSubdivisionSchemeAttr('none')
visible.CreateDoubleSidedAttr(True)
visible.CreateDisplayColorAttr([(.48, .52, .57)])
light = UsdLux.DomeLight.Define(stage, '/World/Light')
light.CreateIntensityAttr(1000.)

capsule = UsdGeom.Capsule.Define(stage, '/World/Person')
capsule.CreateAxisAttr('Z')
capsule.CreateRadiusAttr(.23)
capsule.CreateHeightAttr(1.25)
capsule.CreateDisplayColorAttr([(.1, .75, .3)])
enabled = UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim()).CreateKinematicEnabledAttr(True)
person = world.scene.add(RigidPrim('/World/Person', name='replayed_person', position=np.array([0., 0., -10.])))
camera = world.scene.add(Camera('/World/Camera', name='replay_camera', resolution=(640,480), frequency=100))
world.reset()
camera.initialize()
camera.set_focal_length(18.)
camera.set_horizontal_aperture(24.)
camera.set_vertical_aperture(18.)
camera.set_clipping_range(.05, 100.)
query = omni.physx.get_physx_scene_query_interface()
encoded = PhysicsSchemaTools.encodeSdfPath(capsule.GetPath())


def position_person(xyz):
    person.set_world_pose(xyz)
    # Path-based scene queries read USD, while the live rigid-body view updates PhysX.
    capsule.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float, xyz)))


def ground_at(xy):
    hits = []
    def report(hit):
        if '/World/Environment/CollisionMesh' in str(hit.rigid_body) and hit.normal[2]>.4:
            hits.append((float(hit.distance), float(hit.position[2])))
        return True
    query.raycast_all((float(xy[0]), float(xy[1]), .4), (0.,0.,-1.), 1.5, report)
    return min(hits)[1] if hits else None


def overlaps():
    hits = []
    def report(hit):
        if '/World/Environment/CollisionMesh' in str(hit.rigid_body):
            hits.append(str(hit.collision))
        return True
    query.overlap_shape(int(encoded[0]), int(encoded[1]), report, False)
    return sorted(set(hits))


def move_camera(xy):
    eye = np.r_[xy+np.array([3.5,-5.]), 6.]
    target = np.r_[xy, .6]
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.,0.,1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.column_stack([right, down, forward])@np.diag([1.,-1.,-1.])
    quat = Rotation.from_matrix(matrix).as_quat()
    camera.set_world_pose(eye, np.r_[quat[3],quat[:3]], camera_axes='usd')


def save_progress(results, complete=False):
    payload = {'complete': complete, 'expected': 1 if args.smoke else 9, 'finished':len(results),
               'backend': 'Isaac Sim 4.0 / PhysX', 'control': 'original trajectory kinematic replay',
               'robot_navigation': False, 'physics_dt_s': .01, 'video_fps': 10,
               'results': results}
    temp = out/'status.tmp'
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(out/'status.json')


results = []
save_progress(results)
try:
    # Verify that the PhysX overlap callback can detect a known penetration into the floor.
    xy = inputs[0][1][['x','y']].to_numpy()[0]
    move_camera(xy)
    for _ in range(4): world.step(render=True)
    floor = ground_at(xy)
    assert floor is not None, 'No floor at first trajectory start'
    position_person(np.r_[xy, floor+.3])
    enabled.Set(True)
    world.step(render=True)
    assert overlaps(), 'PhysX overlap positive control failed'
    position_person(np.array([0.,0.,-10.]))
    for item, frame in (inputs[:1] if args.smoke else inputs):
        started = time.monotonic()
        label = Path(item['file']).stem
        times = frame.time_s.to_numpy()
        xy = frame[['x','y']].to_numpy()
        duration = min(float(times[-1]), 1.) if args.smoke else float(times[-1])
        maximum_error = 0.
        overlap_steps, missing_floor, active_steps = 0, 0, 0
        maximum_clock_error = 0.
        frame_count = 0
        sampled_original = set()
        failure = None
        writer = imageio.get_writer(str(out/f'{label}.mp4'), fps=10, codec='libx264',
                                    quality=7, macro_block_size=1, ffmpeg_log_level='error')
        capsule.MakeInvisible()
        position_person(np.array([0.,0.,-10.]))
        world.step(render=False)
        initial_clock = world.current_time
        try:
            with (out/f'{label}_executed.jsonl').open('w') as log:
                for step in range(round(duration/.01)+1):
                    t = step*.01
                    present = times[0]-1e-8<=t<=times[-1]+1e-8
                    target = np.array([np.interp(t,times,xy[:,axis]) for axis in [0,1]])
                    if present:
                        floor = ground_at(target)
                        if floor is None:
                            missing_floor += 1
                            raise RuntimeError(f'No supporting floor at time {t}')
                        enabled.Set(True)
                        capsule.MakeVisible()
                        # CSV has no Z; ground height comes from the actual collision mesh.
                        position_person(np.r_[target,floor+.865])
                    if step%10==0:
                        move_camera(target)
                    world.step(render=step%10==0)
                    sim_t = world.current_time-initial_clock-.01
                    maximum_clock_error = max(maximum_clock_error, abs(sim_t-t))
                    if present:
                        actual, _ = person.get_world_pose()
                        error = float(np.linalg.norm(actual[:2]-target))
                        maximum_error = max(maximum_error,error)
                        collision = overlaps()
                        overlap_steps += bool(collision)
                        active_steps += 1
                        if step%4==0:
                            sampled_original.add(round(t*25))
                        log.write(json.dumps({'time_s':t,'source_time_s':t,'present':True,
                                  'target_xy':target.tolist(),'actual_xyz':actual.tolist(),
                                  'floor_z':floor,'position_error_m':error,'overlaps':collision})+'\n')
                    else:
                        log.write(json.dumps({'time_s':t,'present':False})+'\n')
                    if step%10==0:
                        rgba = camera.get_rgba()
                        if rgba is None or rgba.size==0: raise RuntimeError('Camera produced no frame')
                        rgb = np.ascontiguousarray(rgba[:,:,:3])
                        cv2.putText(rgb,f'{label} | Isaac PhysX replay | t={t:.1f}s',
                                    (12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                        cv2.putText(rgb,'Green: kinematic person proxy; room: visual cutaway',
                                    (12,465),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
                        writer.append_data(rgb)
                        frame_count += 1
        except Exception as error:
            failure = repr(error)
        finally:
            writer.close()
            capsule.MakeInvisible()
            position_person(np.array([0.,0.,-10.]))
        expected_frames = set(frame.frame.astype(int))
        complete_samples = expected_frames <= sampled_original if not args.smoke else None
        result = {'file':item['file'],'seed':item['seed'],'agent_id':item['agent_id'],
                  'source_sha256':item['sha256'],'video':f'{label}.mp4',
                  'maximum_xy_error_m':maximum_error,'maximum_clock_error_s':maximum_clock_error,
                  'static_overlap_steps':overlap_steps,'missing_floor_steps':missing_floor,
                  'active_physics_steps':active_steps,'source_frames_replayed':len(sampled_original),
                  'expected_source_frames':len(frame),'all_source_frames_replayed':complete_samples,
                  'video_frames':frame_count,'wall_seconds':time.monotonic()-started,'error':failure}
        result['passed'] = (failure is None and overlap_steps==0 and missing_floor==0
                            and maximum_error<1e-4 and maximum_clock_error<1e-5
                            and (complete_samples or args.smoke))
        results.append(result)
        save_progress(results)
        print('REPLAY_RESULT',json.dumps(result),flush=True)
        if failure: raise RuntimeError(failure)
    save_progress(results,complete=True)
except BaseException:
    import traceback
    traceback.print_exc()
    (out/'failure.txt').write_text(traceback.format_exc())
finally:
    app.close()
