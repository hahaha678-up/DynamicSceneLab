import bisect
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

ROOT = Path('/work')
OUT = Path(sys.argv[1])
retake = OUT/'retake_status.json'
if retake.exists() and json.loads(retake.read_text()).get('state')=='queued':
    print('COMPOSITION_DEFERRED_UNTIL_MULTI_RETAKE',flush=True)
    raise SystemExit(0)
FPS = 15
INK = (232,237,243)
MUTED = (142,155,170)
CYAN = (74,214,222)
ORANGE = (255,169,76)
BG = (14,21,31)


def text(image, label, xy, scale=.7, color=INK, width=1):
    cv2.putText(image,str(label),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,width,cv2.LINE_AA)


def read_rows(path, name='trajectory.jsonl'):
    return [json.loads(line) for line in (path/name).read_text().splitlines()]


def number(v, digits=2):
    return '--' if v is None or not math.isfinite(v) else f'{v:.{digits}f}'


def panel(title, width=960, height=540):
    image = np.full((height,width,3),BG,np.uint8)
    text(image,title,(28,40),.8)
    return image


def label_image(image, label):
    image=image.copy()
    cv2.rectangle(image,(0,0),(image.shape[1],48),BG,-1)
    text(image,label,(24,33),.75)
    return image


def speed_plot(image, rows, i, box, total=None):
    x,y,w,h=box
    total=total or max(1.,rows[-1]['timestamp'])
    for v in [0.,.4,.8]:
        py=round(y+h-v/.9*h)
        cv2.line(image,(x,py),(x+w,py),(44,57,72),1)
        text(image,f'{v:.1f}',(x-36,py+5),.45,MUTED)
    points=np.asarray([[x+min(r['timestamp']/total,1)*w,
                        y+h-min(r['carter']['speed']/.9,1.)*h] for r in rows[:i+1]],np.int32)
    if len(points)>1:cv2.polylines(image,[points],False,CYAN,3,cv2.LINE_AA)
    text(image,'Carter speed (m/s)',(x,y-12),.6,MUTED)
    text(image,'0 s',(x,y+h+23),.45,MUTED)
    text(image,f'{total:.0f} s',(x+w-38,y+h+23),.45,MUTED)


def metrics_panel(rows,i):
    row=rows[i];pair=row['pair']
    image=panel('CLOSED-LOOP RESPONSE')
    text(image,number(row['carter']['speed']),(35,128),1.9,CYAN,3)
    text(image,'m/s  Carter speed',(220,122),.65,MUTED)
    text(image,'Clearance',(500,88),.65,MUTED)
    text(image,number(pair['clearance'])+' m',(500,134),1.1,ORANGE,2)
    text(image,'TTC',(750,88),.65,MUTED)
    text(image,number(pair['ttc'])+' s',(750,134),1.1,INK,2)
    speed_plot(image,rows,i,(80,219,810,180))
    state=row['navigation']['controller']['status'].upper()
    text(image,'Nav2  '+state,(40,476),.72)
    text(image,f"Simulation time  {row['timestamp']:.2f} s",(525,476),.65,MUTED)
    text(image,'Clearance / TTC: Carter + person disc proxies',(40,517),.48,MUTED)
    return image


def scan_points(obs):
    d=obs['observation'];c=d['carter'];yaw=c['heading']
    ranges=np.array([np.nan if r is None else r for r in d['ranges']])
    angles=d['angle_min']+np.arange(len(ranges))*d['angle_increment']+yaw
    origin=np.array([c['x']+.25*math.cos(yaw),c['y']+.25*math.sin(yaw)])
    return origin+ranges[:,None]*np.column_stack([np.cos(angles),np.sin(angles)])


class SensorPanel:
    def __init__(self,episode,tag):
        self.obs=read_rows(episode,'observations.jsonl')
        path=Path('/repo/hunav-core/runtime')/tag/'costmaps.jsonl'
        journal=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        self.maps=sorted([r for r in journal if r['kind']=='costmap'],key=lambda r:r['time'])
        self.map_times=[r['time'] for r in self.maps]
        self.plans=sorted([r for r in journal if r['kind']=='local_plan'],key=lambda r:r['time'])
        self.plan_times=[r['time'] for r in self.plans]

    def draw(self,row,index):
        image=panel('LiDAR / LOCAL COSTMAP')
        c=row['carter'];center=np.array([c['x'],c['y']]);scale=54.
        def pixel(xy):
            p=(np.asarray(xy)-center)*np.array([scale,-scale])+[480,296]
            return np.rint(p).astype(int)
        observation=self.obs[min(index,len(self.obs)-1)]
        stamp=observation['observation']['time']
        j=bisect.bisect_right(self.map_times,stamp)-1
        if j>=0:
            m=self.maps[j]
            array=np.array(m['data']).reshape(m['height'],m['width'])
            values=np.clip(array,0,100)/100
            colors=np.stack([35+values*160,45+values*67,56+values*37],axis=-1).astype(np.uint8)
            lower=pixel(m['origin'])
            size=np.rint([m['width']*m['resolution']*scale,m['height']*m['resolution']*scale]).astype(int)
            raster=cv2.resize(np.flipud(colors),tuple(size),interpolation=cv2.INTER_NEAREST)
            x,y=int(lower[0]),int(lower[1]-size[1])
            x0,y0=max(0,x),max(55,y);x1,y1=min(960,x+size[0]),min(535,y+size[1])
            if x1>x0 and y1>y0:image[y0:y1,x0:x1]=raster[y0-y:y1-y,x0-x:x1-x]
        points=scan_points(observation)
        for p in points[np.isfinite(points).all(axis=1)]:
            px,py=pixel(p)
            if 12<px<948 and 58<py<528:cv2.circle(image,(px,py),1,CYAN,-1,cv2.LINE_AA)
        j=bisect.bisect_right(self.plan_times,stamp)-1
        if j>=0 and self.plans[j]['frame']=='odom' and len(self.plans[j]['points'])>1:
            points2=pixel(self.plans[j]['points'])
            if len(points2)>1:cv2.polylines(image,[points2],False,(126,244,135),2,cv2.LINE_AA)
        xy=pixel(center)
        cv2.circle(image,tuple(xy),round(.55*scale),INK,2,cv2.LINE_AA)
        tip=pixel(center+.7*np.array([math.cos(c['heading']),math.sin(c['heading'])]))
        cv2.arrowedLine(image,tuple(xy),tuple(tip),INK,2,tipLength=.3)
        text(image,'LiDAR returns',(20,91),.5,CYAN)
        text(image,'Obstacle cost',(20,119),.5,ORANGE)
        if self.plans:text(image,'DWB local plan',(20,147),.5,(126,244,135))
        return image


def writer(name):
    return imageio.get_writer(OUT/name,fps=FPS,codec='libx264',quality=8,
                             macro_block_size=1,ffmpeg_log_level='error')


segments=[]
with writer('01_dynamic_room.mp4') as output:
    reader=imageio.get_reader(OUT/'multi_views.mp4')
    for frame in reader:output.append_data(cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_CUBIC))
    reader.close()
segments.append('01_dynamic_room.mp4')
print('COMPOSED dynamic room',flush=True)

with writer('02_closed_loop.mp4') as output:
    for name in ['crossing','occlusion']:
        episode=OUT/name
        tag='ab_'+OUT.name+'_'+name
        rows=read_rows(episode);sensor=SensorPanel(episode,tag)
        reader=imageio.get_reader(OUT/(name+'_views.mp4'))
        for k,frame in enumerate(reader):
            i=min(k*2,len(rows)-1)
            canvas=np.empty((1080,1920,3),np.uint8)
            canvas[:540,:960]=label_image(frame[:,:960],name.upper()+' / OVERVIEW')
            canvas[:540,960:]=label_image(frame[:,960:], 'CARTER RGB VIEW')
            canvas[540:,:960]=sensor.draw(rows[i],i)
            canvas[540:,960:]=metrics_panel(rows,i)
            cv2.line(canvas,(960,0),(960,1080),BG,4)
            cv2.line(canvas,(0,540),(1920,540),BG,4)
            output.append_data(canvas)
        reader.close()
segments.append('02_closed_loop.mp4')
print('COMPOSED closed loop',flush=True)

geometry=np.load(ROOT/'output/geometry.npz')
mask=geometry['walkable'];origin=geometry['origin'];resolution=float(geometry['resolution'])
accepted=[]
colors=[(75,214,226),(250,181,92),(160,132,251),(241,110,138),(129,220,165),(145,191,252),(217,202,133)]
for n in [1,2,4,5,7,8,9]:
    rows=list(csv.DictReader((Path('/repo/examples/trajectories')/f'person_{n:03d}.csv').open()))
    accepted.append(np.array([[float(r['x']),float(r['y'])] for r in rows]))


def topdown(width,height,highlight=False,person=None,carter=None):
    bounds=np.array([[-2.,-7.5],[7.,17.]])
    scale=min((width-80)/(bounds[1,0]-bounds[0,0]),(height-80)/(bounds[1,1]-bounds[0,1]))
    center=bounds.mean(axis=0)
    def p(xy):return np.rint((np.asarray(xy)-center)*[scale,-scale]+[width/2,height/2]).astype(int)
    yy,xx=np.mgrid[:height,:width]
    wx=(xx-width/2)/scale+center[0];wy=-(yy-height/2)/scale+center[1]
    ix=np.rint((wx-origin[0])/resolution).astype(int);iy=np.rint((wy-origin[1])/resolution).astype(int)
    good=(ix>=0)&(iy>=0)&(ix<mask.shape[1])&(iy<mask.shape[0])
    image=np.full((height,width,3),BG,np.uint8)
    floor=np.zeros(good.shape,bool);floor[good]=mask[iy[good],ix[good]]>0
    image[floor]=(49,61,73)
    for i,track in enumerate(accepted):
        color=colors[i] if not highlight or i==2 else (72,84,97)
        cv2.polylines(image,[p(track)],False,color,3 if i==2 and highlight else 2,cv2.LINE_AA)
        cv2.circle(image,tuple(p(track[0])),5,color,-1,cv2.LINE_AA)
    if person is not None and person['present']:cv2.circle(image,tuple(p([person['x'],person['y']])),7,ORANGE,-1,cv2.LINE_AA)
    if carter is not None:cv2.circle(image,tuple(p([carter['x'],carter['y']])),7,INK,2,cv2.LINE_AA)
    return image


with writer('03_crowdes.mp4') as output:
    for k in range(6*FPS):
        canvas=np.full((1080,1920,3),BG,np.uint8)
        canvas[:,80:980]=topdown(900,1080,k>=3*FPS)
        text(canvas,'CrowdES',(1080,220),1.8,INK,3)
        text(canvas,'Generated person trajectories',(1080,290),.85,MUTED)
        text(canvas,'Same room / different paths',(1080,362),.85)
        text(canvas,'Existing pretrained samples',(1080,407),.7,MUTED)
        text(canvas,'Selected: person_004',(1080,590),1.0,ORANGE,2)
        text(canvas,'Original spatial path and timestamps',(1080,646),.7,MUTED)
        text(canvas,'Trajectory  >  Isaac  >  Nav2',(1080,805),.85,CYAN)
        output.append_data(canvas)
    rows=read_rows(OUT/'crowdes')
    reader=imageio.get_reader(OUT/'crowdes_views.mp4')
    for k,frame in enumerate(reader):
        row=rows[min(k*2,len(rows)-1)]
        if not 10. <= row['timestamp'] <= 25.:continue
        canvas=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(canvas,(0,0),(1920,66),BG,-1)
        text(canvas,'CrowdES / SAME TRAJECTORY IN ISAAC',(32,44),.85)
        text(canvas,f"t = {row['timestamp']:.2f} s",(1650,44),.7,MUTED)
        inset=topdown(300,600,True,row['person'],row['carter'])
        canvas[450:1050,1590:1890]=inset
        cv2.rectangle(canvas,(1590,450),(1890,1050),MUTED,2)
        text(canvas,'Nav2 speed  '+number(row['carter']['speed'])+' m/s',(35,1035),.8,INK,2)
        output.append_data(canvas)
    reader.close()
segments.append('03_crowdes.mp4')
print('COMPOSED CrowdES',flush=True)

names=['variant_a','variant_b','variant_c']
names.sort(key=lambda name:json.loads((OUT/name/'result.json').read_text())['metrics']['min_clearance_m'],reverse=True)
datasets=[read_rows(OUT/name) for name in names]
configs=[json.loads((OUT/name/'scenario.json').read_text()) for name in names]
results=[json.loads((OUT/name/'result.json').read_text()) for name in names]
readers=[imageio.get_reader(OUT/(name+'_views.mp4')) for name in names]
last=[None]*3
duration=min(20.,min(rows[-1]['timestamp'] for rows in datasets))
with writer('04_parent_variants.mp4') as output:
    for k in range(math.ceil(duration*FPS)):
        canvas=np.full((1080,1920,3),BG,np.uint8)
        text(canvas,'ONE HIGH-RISK PARENT / THREE PARAMETER VARIANTS',(34,48),1.0,INK,2)
        text(canvas,'Same spatial trajectory | manually selected timing and speed parameters',(36,88),.65,MUTED)
        for j,(reader,rows,spec,result) in enumerate(zip(readers,datasets,configs,results)):
            try:last[j]=reader.get_next_data()
            except (IndexError,StopIteration):pass
            if last[j] is None:raise RuntimeError('Missing variant frame')
            x=j*640;i=min(k*2,len(rows)-1);row=rows[i]
            image=cv2.resize(last[j][:,192:768],(640,600),interpolation=cv2.INTER_CUBIC)
            canvas[155:755,x:x+640]=image
            text(canvas,'VARIANT '+chr(65+j),(x+25,136),.8)
            params=spec['variant_parameters']
            text(canvas,f"Shift {params['time_shift_s']:+.2f} s   Speed x{params['speed_scale']:.2f}",(x+25,792),.65,ORANGE)
            text(canvas,'Carter  '+number(row['carter']['speed'])+' m/s',(x+25,838),.78,CYAN,2)
            text(canvas,'Clearance '+number(row['pair']['clearance'])+' m',(x+25,878),.65)
            text(canvas,'TTC '+number(row['pair']['ttc'])+' s',(x+375,878),.65)
            speed_plot(canvas,rows,i,(x+62,935,520,73),duration)
            text(canvas,'Min. clearance '+number(result['metrics']['min_clearance_m'])+' m',(x+25,1060),.56,MUTED)
            if j:cv2.line(canvas,(x,110),(x,1080),(65,76,91),2)
        output.append_data(canvas)
for reader in readers:reader.close()
segments.append('04_parent_variants.mp4')
print('COMPOSED variants',flush=True)

concat=OUT/'concat.txt'
concat.write_text(''.join("file '"+name+"'\n" for name in segments))
final=OUT/'real2sim_demo.mp4'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0',
                '-i',str(concat),'-c','copy','-movflags','+faststart',str(final)],check=True)
cap=cv2.VideoCapture(str(final));count=0;brightness=[]
width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ok,frame=cap.read()
    if not ok:break
    count+=1
    if count%FPS==0:brightness.append(float(frame.mean()))
cap.release()
if width!=1920 or height!=1080 or count<60*FPS or not brightness or min(brightness)<3:
    raise RuntimeError('Final video failed basic frame verification')
(OUT/'video_result.json').write_text(json.dumps({'file':str(final),'width':width,'height':height,
    'fps':FPS,'decoded_frames':count,'duration_s':count/FPS,'segments':segments,
    'variant_parameter_selection':'manual demo parameters; no risk-feedback search claimed',
    'crowdes_source':'existing pretrained person_004.csv, original timing and positions'},indent=2))
print('FINAL_VIDEO',str(final),count/FPS,flush=True)
