import json
import math
import time
from pathlib import Path
import numpy as np
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,
    'extra_args':['--/rtx/verifyDriverVersion/enabled=false']})
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('omni.isaac.range_sensor')
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core import World
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
from scipy.spatial.transform import Rotation
import omni.kit.commands
import omni.physx

root=Path('/work');out=root/'output'/('person_lidar_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()))
out.mkdir(exist_ok=False)
g=dict(np.load(root/'output/geometry.npz'))
world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30)
stage=world.stage
mesh=UsdGeom.Mesh.Define(stage,'/World/Environment/CollisionMesh')
mesh.CreatePointsAttr(g['points'].astype(np.float32));mesh.CreateFaceVertexCountsAttr(np.full(len(g['faces']),3,np.int32));mesh.CreateFaceVertexIndicesAttr(g['faces'].ravel());mesh.CreateSubdivisionSchemeAttr('none')
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr(.005)
mesh.MakeInvisible()
center_offset=np.array([-.48,0.])
yaw=math.pi/2
robot=world.scene.add(WheeledRobot(prim_path='/World/Carter',name='carter',create_robot=True,
 wheel_dof_names=['joint_wheel_left','joint_wheel_right'],
 usd_path=str(root/'assets/isaac-4.0/Isaac/Robots/Carter/nova_carter.usd'),
 position=np.array([1.3,9.48,.12]),orientation=np.array([math.cos(yaw/2),0.,0.,math.sin(yaw/2)])))
controller=DifferentialController(name='scan_check_drive',wheel_radius=.14,wheel_base=.834)
proxy=UsdGeom.Capsule.Define(stage,'/World/Person')
proxy.CreateAxisAttr('Z');proxy.CreateRadiusAttr(.23);proxy.CreateHeightAttr(1.25)
translate=UsdGeom.Xformable(proxy).AddTranslateOp();translate.Set(Gf.Vec3d(1.3,11.5,.86))
collision=UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr(False)
UsdPhysics.RigidBodyAPI.Apply(proxy.GetPrim()).CreateKinematicEnabledAttr(True)
proxy.MakeInvisible()
ok,lidar=omni.kit.commands.execute('RangeSensorCreateLidar',path='/CheckLidar',parent='/World/Carter/chassis_link',
 min_range=.1,max_range=20.,draw_points=False,draw_lines=False,horizontal_fov=360.,vertical_fov=4.,
 horizontal_resolution=.5,vertical_resolution=2.,rotation_rate=0.,high_lod=True,yaw_offset=0.,enable_semantics=True)
assert ok
lidar_path=str(lidar.GetPath())
lidar.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(-.23,0.,.6))
sensor=_range_sensor.acquire_lidar_sensor_interface()
world.reset()
zero=controller.forward(np.zeros(2))
for i in range(240):
 robot.apply_wheel_actions(zero);world.step(render=i%4==0)
settled_z=float(robot.get_world_pose()[0][2])
results=[];arrays={};failure=None


def capture(name,car_xy,person_xy,enabled=True,car_heading=math.pi/2):
    rot=Rotation.from_euler('z',car_heading).as_matrix()[:2,:2]
    robot.set_world_pose(np.r_[np.asarray(car_xy)-rot@center_offset,settled_z],
                         np.array([math.cos(car_heading/2),0.,0.,math.sin(car_heading/2)]))
    robot.set_linear_velocity(np.zeros(3));robot.set_angular_velocity(np.zeros(3))
    robot.set_joint_velocities(np.zeros(robot.num_dof))
    translate.Set(Gf.Vec3d(float(person_xy[0]),float(person_xy[1]),.86));collision.Set(enabled)
    proxy.MakeInvisible()
    for step in range(36):
        robot.apply_wheel_actions(zero);world.step(render=step%4==0)
    depth=np.asarray(sensor.get_linear_depth_data(lidar_path)).copy()
    prims=np.asarray(sensor.get_prim_data(lidar_path)).astype(str).reshape(depth.shape)
    azimuth=np.asarray(sensor.get_azimuth_data(lidar_path)).copy()
    zenith=np.asarray(sensor.get_zenith_data(lidar_path)).copy()
    print('SCAN_SHAPES',name,depth.shape,azimuth.shape,zenith.shape,flush=True)
    mask=np.char.find(prims,'/World/Person')>=0
    cache=UsdGeom.XformCache()
    eye=np.array(cache.GetLocalToWorldTransform(lidar.GetPrim()).ExtractTranslation())
    proxy_center=np.array(cache.GetLocalToWorldTransform(proxy.GetPrim()).ExtractTranslation())
    assert np.linalg.norm(proxy_center[:2]-person_xy)<1e-5
    surface_expected=float(np.linalg.norm(np.asarray(person_xy)-eye[:2])-.23)
    hit_indices=np.where(mask)[0]
    result={'case':name,'collision_enabled':enabled,'proxy_visibility':str(proxy.ComputeVisibility()),
        'person_xy':list(person_xy),'sensor_world_xyz':eye.tolist(),'hit_count':int(mask.sum()),
        'person_min_range_m':float(depth[mask].min()) if mask.any() else None,
        'expected_nearest_surface_m':surface_expected,
        'person_mean_azimuth_deg':float(np.degrees(azimuth[hit_indices]).mean()) if len(hit_indices) else None,
        'unique_hit_prims':np.unique(prims).tolist()}
    arrays[name+'_range_m']=depth;arrays[name+'_hit_person']=mask;arrays[name+'_azimuth']=azimuth;arrays[name+'_zenith']=zenith
    results.append(result);print('SCAN_RESULT',json.dumps(result),flush=True)
    return result


try:
    absent=capture('absent',[1.3,9.],[1.3,11.5],False)
    front=capture('front',[1.3,9.],[1.3,11.5])
    side=capture('shifted_right',[1.3,9.],[2.,11.5])
    near=capture('nearer',[1.3,9.],[1.3,10.5])
    removed=capture('removed',[1.3,9.],[1.3,10.5],False)
    source=Path('/work/output/crossing_20260915T140713_3603181/04_window2_person1.0/trajectory.jsonl')
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    hidden=next(r for r in rows if r['timestamp']>=3.2 and r['person']['present'] and not r['visibility']['visible'])
    emerged=next(r for r in rows if r['visibility']['clear_samples']==9)
    def replay_case(name,row):
        c,p=row['carter'],row['person']
        return capture(name,[c['x'],c['y']],[p['x'],p['y']],True,c['heading'])
    blocked=replay_case('real_pillar_hidden',hidden)
    visible=replay_case('real_pillar_emerged',emerged)
    assert absent['hit_count']==0 and removed['hit_count']==0
    for r in [front,side,near,visible]:
        assert r['hit_count']>0, r['case']+' was not detected'
        assert abs(r['person_min_range_m']-r['expected_nearest_surface_m'])<.04, r
        assert r['proxy_visibility']=='invisible'
    assert blocked['hit_count']==0, 'LiDAR detected person through the real pillar'
    assert abs(front['person_mean_azimuth_deg']-side['person_mean_azimuth_deg'])>10.
    assert front['person_min_range_m']-near['person_min_range_m']>.9
except Exception as exc:
    failure=repr(exc)
finally:
    np.savez_compressed(out/'scans.npz',**arrays)
    report={'success':failure is None,'failure':failure,'sensor':'Isaac Sim 4.0 PhysX RangeSensorCreateLidar',
        'lidar_path':lidar_path,'scene':'Habitat-GS scene55 collision mesh','person_representation':'hidden kinematic capsule, radius 0.23 m',
        'measurement_source':'get_linear_depth_data / get_prim_data from the actual LiDAR interface',
        'scope':'sensor detection verification only; not Nav2, ROS scan publication, RTX LiDAR or raw depth camera verification',
        'cases':results}
    (out/'report.json').write_text(json.dumps(report,indent=2));print('REPORT',str(out),json.dumps({'success':failure is None,'failure':failure}),flush=True)
    app.close()
