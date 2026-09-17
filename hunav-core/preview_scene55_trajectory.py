import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/repo/hunav-core')
from hunav_client import HuNavClient

root = Path('/work')
config = json.loads((root/'scene_config.json').read_text())
geometry = np.load(root/'output/geometry.npz')
mesh = np.load(root/'output/mesh_clearance.npz')
origin, resolution = geometry['origin'], float(geometry['resolution'])
obstacles = mesh['obstacles'].astype(bool)
route = np.array(config['routes']['cross'], dtype=float)
assert route.shape == (2, 2), 'This preview requires a straight crossing route'
human_config = config['human']
robot_speed = 1.0
robot_radius = .55
length = np.linalg.norm(route[1]-route[0])
direction = (route[1]-route[0])/length
angles = np.arange(32)*2*np.pi/32
rays = np.column_stack([np.cos(angles), np.sin(angles)])
ranges = np.arange(.25, 3.26, .05)

def obstacle_points(xy):
    points = np.asarray(xy)[None,None,:] + rays[:,None,:]*ranges[None,:,None]
    pixels = np.rint((points-origin)/resolution).astype(int)
    valid = ((pixels[:,:,0]>=0)&(pixels[:,:,0]<obstacles.shape[1])&
             (pixels[:,:,1]>=0)&(pixels[:,:,1]<obstacles.shape[0]))
    hits = np.zeros(valid.shape, bool)
    hits[valid] = obstacles[pixels[:,:,1][valid],pixels[:,:,0][valid]]
    return [points[i,np.flatnonzero(row)[0]].tolist() for i,row in enumerate(hits) if row.any()]

runs = []
client = HuNavClient('/repo/hunav-core/runtime/hunav.sock')
try:
    for delay in [float(config['human_start_delay']), 2.5]:
        client.request(op='reset', human=human_config)
        human = {'xy': human_config['start'], 'speed': 0., 'arrived': False}
        rows = []
        for frame in range(541):
            t = frame/30
            distance = min(robot_speed*t, length)
            xy = route[0]+direction*distance
            velocity = direction*robot_speed if distance < length else np.zeros(2)
            robot = {'xy':xy.tolist(), 'velocity':velocity.tolist(),
                     'yaw':math.atan2(direction[1],direction[0]), 'radius':robot_radius}
            if t >= delay:
                human = client.request(op='step',t=t,robot=robot,obstacles=obstacle_points(human['xy']))
            rows.append({'t':t,'robot':robot,'human':dict(human),
                         'gap':float(np.linalg.norm(xy-human['xy'])-robot_radius-human_config['radius'])})
        hp = np.array([r['human']['xy'] for r in rows])
        pixels = np.rint((hp-origin)/resolution).astype(int)
        assert np.isfinite(hp).all()
        assert ((pixels[:,0]>=0)&(pixels[:,0]<obstacles.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<obstacles.shape[0])).all()
        on_nav = geometry['walkable'][pixels[:,1],pixels[:,0]].astype(bool)
        lateral = abs(hp[:,1]-human_config['start'][1]).max()
        closest = min(rows,key=lambda r:r['gap'])
        summary = {'delay_s':delay,'robot_speed_mps':robot_speed,'robot_crossing_s':float((10-route[0,1])/robot_speed),
                   'minimum_disc_gap_m':closest['gap'],'closest_time_s':closest['t'],
                   'max_human_lateral_offset_m':float(lateral),'human_arrived':rows[-1]['human']['arrived'],
                   'human_on_walkable_fraction':float(on_nav.mean())}
        runs.append({'summary':summary,'rows':rows})
        print(json.dumps(summary),flush=True)
finally:
    client.close()

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(12,9),layout='constrained')
nav = geometry['walkable'].astype(bool)
canvas = np.ones((*nav.shape,3))*.96
canvas[nav] = [.88,.90,.92]
canvas[obstacles] = [.28,.31,.35]
extent = [origin[0]-.5*resolution,origin[0]+(nav.shape[1]-.5)*resolution,
          origin[1]-.5*resolution,origin[1]+(nav.shape[0]-.5)*resolution]
for ax,run in zip(axes,runs):
    rows,summary = run['rows'],run['summary']
    rp = np.array([r['robot']['xy'] for r in rows]);hp=np.array([r['human']['xy'] for r in rows])
    ax.imshow(canvas,origin='lower',extent=extent)
    ax.plot(rp[:,0],rp[:,1],color='#1976d2',lw=2.8,label='Carter: 1.00 m/s')
    ax.plot(hp[:,0],hp[:,1],color='#e87519',lw=2.8,label='Person: HuNav response')
    ax.scatter(*rp[0],marker='s',s=65,color='#1976d2');ax.scatter(*hp[0],marker='s',s=65,color='#e87519')
    ax.scatter(*rp[-1],marker='*',s=150,color='#1976d2');ax.scatter(*human_config['goal'],marker='*',s=150,color='#e87519')
    for t in [3,5,7]:
        row=rows[t*30]
        for key,color,radius,dx in [('robot','#1976d2',robot_radius,-.65),('human','#e87519',human_config['radius'],.3)]:
            xy=row[key]['xy'];ax.scatter(*xy,s=35,color=color,zorder=5)
            if not (key=='human' and t<summary['delay_s'] and t!=5):
                ax.annotate(f'{t}s',xy,xytext=(xy[0]+dx,xy[1]+.12+(t-5)*.16),color=color,fontsize=10)
            if t==5:ax.add_patch(plt.Circle(xy,radius,fill=False,ec=color,lw=1.5,ls='--'))
    close=min(rows,key=lambda r:r['gap'])
    ax.plot([close['robot']['xy'][0],close['human']['xy'][0]],
            [close['robot']['xy'][1],close['human']['xy'][1]],'--',color='#8d51aa',lw=1.3)
    ax.set(xlim=(-2,7),ylim=(3,16.5),aspect='equal',xlabel='X (m)',ylabel='Y (m)')
    ax.set_title(f"Person starts at {summary['delay_s']:.1f} s\nMin clearance: {summary['minimum_disc_gap_m']:.2f} m; deviation: {summary['max_human_lateral_offset_m']:.2f} m",fontsize=12,pad=12)
    ax.legend(loc='lower right',fontsize=9,framealpha=.95)
    ax.grid(alpha=.15)
fig.suptitle('Scene55 | Crossing trajectory preview | No video rendering',fontsize=17)
fig.savefig(root/'output/scene55_trajectory_1mps.png',dpi=150)
(root/'output/scene55_trajectory_1mps.json').write_text(json.dumps({'mode':'kinematic Carter + official HuNav, mesh-projected obstacle rays','scene':config['scene'],'runs':runs}))