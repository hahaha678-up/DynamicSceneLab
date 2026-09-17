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

root=Path('/work');out=root/'output'/('ros_interfaces_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()))
out.mkdir(exist_ok=False)
g=dict(np.load(root/'output/geometry.npz'))
world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/120)
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
joints=[p for p in stage.Traverse() if p.GetName() in ['joint_wheel_left','joint_wheel_right']]
anchors=[np.array(UsdPhysics.Joint(p).GetLocalPos0Attr().Get()) for p in joints]
for p,a in zip(joints,anchors):
 print('WHEEL_ANCHOR',str(p.GetPath()),a.tolist(),[str(x) for x in UsdPhysics.Joint(p).GetBody0Rel().GetTargets()],flush=True)
wheel_base=float(np.linalg.norm(anchors[0]-anchors[1]))
assert .3<wheel_base<.6, wheel_base
controller=DifferentialController(name='scan_check_drive',wheel_radius=.14,wheel_base=wheel_base)
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

from carter_ros import CarterRosLink
link=CarterRosLink(sensor,lidar_path)
collision.Set(True)
records=[];failure=None
start_time=world.current_time

def carter_state():
    position,q=robot.get_world_pose()
    heading=float(Rotation.from_quat(np.r_[q[1:],q[0]]).as_euler('xyz')[2])
    rot=Rotation.from_euler('z',heading).as_matrix()[:2,:2]
    center=position[:2]+rot@center_offset
    omega=float(robot.get_angular_velocity()[2])
    velocity=robot.get_linear_velocity()[:2]+omega*np.array([-(rot@center_offset)[1],(rot@center_offset)[0]])
    return dict(x=float(center[0]),y=float(center[1]),z=float(position[2]),heading=heading,
                vx=float(velocity[0]),vy=float(velocity[1]),speed=float(np.linalg.norm(velocity)),angular_velocity=omega)

command=np.zeros(2)
try:
    for step in range(11*120):
        if step%4==0:
            t=float(world.current_time-start_time)
            assert abs(t-step/120)<1e-5, (t,step/120)
            state=carter_state()
            command,reply=link.exchange(t,state)
            records.append(dict(timestamp=t,carter=state,command=command.tolist(),**reply))
        robot.apply_wheel_actions(controller.forward(command))
        world.step(render=step%4==3)
    def near(t):
        return min(records,key=lambda r:abs(r['timestamp']-t))
    a,b=near(2.),near(4.)
    forward_distance=b['carter']['y']-a['carter']['y']
    yaw_change=near(7.)['carter']['heading']-near(5.)['carter']['heading']
    late=[r for r in records if r['timestamp']>10.]
    checks={
        'forward_motion': .55<forward_distance<1.0,
        'positive_turn': .4<yaw_change<.8,
        'explicit_stop': near(4.9)['carter']['speed']<.03,
        'command_timeout_zero': all(r['command_stale'] and r['command']==[0.,0.] for r in late),
        'timeout_physical_stop': max(r['carter']['speed'] for r in late)<.03,
        'received_ros_commands': records[-1]['received_commands']>200,
        'floor_contact': max(abs(r['carter']['z']-settled_z) for r in records)<.05}
    if not all(checks.values()):
        raise AssertionError(checks)
except Exception as exc:
    failure=repr(exc)
finally:
    robot.apply_wheel_actions(zero)
    link.close()
    report=dict(success=failure is None,failure=failure,checks=locals().get('checks',{}),
                forward_displacement_m=locals().get('forward_distance'),
                yaw_change_rad=locals().get('yaw_change'),
                source='real ROS Twist -> socket -> differential wheel command -> PhysX state',
                odometry='simulation ground-truth ego pose and velocity; no wheel-noise model',wheel_base_m=wheel_base)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('ISAAC_INTERFACE_REPORT',str(out),json.dumps(report),flush=True)
    app.close()
