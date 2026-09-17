import json, math, sys
from pathlib import Path
import numpy as np
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,
 'extra_args':['--/rtx/verifyDriverVersion/enabled=false']})
from omni.isaac.core import World
from pxr import UsdGeom,UsdPhysics
import omni.physx
from visibility import visibility,physx_static_raycast
world=World(stage_units_in_meters=1.)
g=dict(np.load('/work/output/geometry.npz'))
c=np.load('/work/output/mesh_clearance.npz')['clearance']
m=UsdGeom.Mesh.Define(world.stage,'/World/Environment/CollisionMesh')
m.CreatePointsAttr(g['points']);m.CreateFaceVertexCountsAttr(np.full(len(g['faces']),3,np.int32));m.CreateFaceVertexIndicesAttr(g['faces'].ravel())
UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
world.reset()
for _ in range(10):world.step(render=False)
ray=physx_static_raycast(omni.physx.get_physx_scene_query_interface())
def good(points,margin):
 pix=np.rint((np.asarray(points)-g['origin'])/g['resolution']).astype(int)
 return bool(((g['walkable'][pix[:,1],pix[:,0]]>0)&(c[pix[:,1],pix[:,0]]>=margin)).all())
print('RAW_QUERY',omni.physx.get_physx_scene_query_interface().raycast_closest((2.5,12.1,.6),(1.,0,0),3.),flush=True)
print('RAY_TEST',ray(np.array([2.5,12.1,.6]),np.array([1.,0,0]),3.),flush=True)
print('KNOWN_VIS',visibility({'x':2.6,'y':11,'heading':math.pi/2},{'present':True,'x':4.4,'y':13},ray),flush=True)
results=[]
counts={'valid_route':0,'hidden':0,'outside':0,'clear':0}
for x in [2.7,2.75]:
 if not good(np.array([[x,y] for y in np.linspace(8.5,15.5,150)]),.60):continue
 for y in np.arange(11.3,14.31,.15):
  point=np.array([x,y]);tc=(y-8.5)/.8
  for angle in np.arange(-70,86,5):
   approach=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
   for pre in [1.,1.5,2.,2.5,3.,3.5]:
    start=point+pre*approach;goal=point-1.5*approach
    if not good(np.linspace(start,goal,60),.28):continue
    spawn=tc-pre
    if spawn<0:continue
    counts['valid_route']+=1
    hidden=None;first=None;hit0=None
    for t in np.arange(spawn,tc+.001,.05):
     xy=start-approach*(t-spawn)
     v=visibility({'x':x,'y':8.5+.8*t,'heading':math.pi/2}, {'present':True,'x':xy[0],'y':xy[1]},ray)
     if hidden is None:
      hidden=v['in_fov_samples']>0 and not v['visible'] and v['blocked_samples']>0
      hit0=v['blocker_points']
      if not hidden:break
      counts['hidden']+=1
     if v['visible']:first=t;break
    if hidden and first is not None:
     results.append({'x':x,'y':round(y,3),'angle':int(angle),'pre':pre,'rw':round(tc-first,3),'start':start.tolist(),'goal':goal.tolist(),'hit':hit0[0]})
print('COUNTS',counts,flush=True)
print('CANDIDATES',json.dumps(results),flush=True)
Path('/work/output/occlusion_probe.json').write_text(json.dumps(results))
app.close()
