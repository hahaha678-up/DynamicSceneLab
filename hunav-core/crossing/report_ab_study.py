import argparse
import base64
import io
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('study')
args = parser.parse_args()
study = Path(args.study).resolve()
assert study.parent == Path('/work/output')
manifest = json.loads((study/'manifest.json').read_text())
palette = {'fixed':'#2474b5', 'nav2':'#e27626'}


def load(run):
    episode = Path('/work/output')/run['tag']/('00_'+Path(run['scenario']).stem)
    path = episode/'trajectory.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    resolved = json.loads((episode/'resolved.json').read_text()) if (episode/'resolved.json').exists() else {}
    m = run['metrics']
    result = {'event':run['event'], 'scenario':Path(run['scenario']).stem, 'mode':run['mode'],
              'status':m['status'], 'collision':m.get('collision'), 'clearance':m.get('minimum_clearance_m'),
              'ttc':m.get('minimum_ttc_s'), 'goal_error':m.get('carter_goal_error_m'),
              'person_goal':m.get('person_goal_reached'), 'duration':m.get('simulated_seconds'),
              'first_visible':m.get('first_visible_time'), 'reaction_window':m.get('reaction_window_actual'),
              'event_valid':m.get('event_valid'), 'termination':m.get('termination'),
              'person_lateral':m.get('person_max_lateral_deviation_m'),
              'person_min_speed':m.get('person_min_active_speed_mps'), 'first_lidar':None}
    if rows:
        goal = resolved['resolved']['path'][-1]
        active = [r for r in rows if r['person']['present'] and
                  math.dist([r['carter']['x'],r['carter']['y']],goal)>1.]
        result['carter_min_speed'] = min((r['carter']['speed'] for r in active),default=None)
        result['braking_duration'] = sum(r['carter']['speed']<.64 for r in active)/30
        human = resolved['resolved']['person']
        result['person_speed_reduction'] = human['speed']-result['person_min_speed'] if result['person_min_speed'] is not None else None
        headings = [abs(math.atan2(math.sin(r['person']['heading']-human['yaw']),
                                  math.cos(r['person']['heading']-human['yaw']))) for r in rows if r['person']['present'] and r['person']['speed']>.05]
        result['person_heading_change_deg'] = math.degrees(max(headings)) if headings else None
        observation_file = episode/'observations.jsonl'
        if observation_file.exists():
            with observation_file.open() as f:
                for row, line in zip(rows, f):
                    if not row['person']['present']:
                        continue
                    c,p = row['carter'], row['person']
                    scan = json.loads(line)['observation']
                    angles = scan['angle_min']+np.arange(len(scan['ranges']))*scan['angle_increment']+c['heading']
                    ranges = np.array([v if v is not None else np.nan for v in scan['ranges']])
                    eye = np.array([c['x']+.25*np.cos(c['heading']),c['y']+.25*np.sin(c['heading'])])
                    hits = eye+np.c_[np.cos(angles),np.sin(angles)]*ranges[:,None]
                    if np.any(np.linalg.norm(hits-[p['x'],p['y']],axis=1)<.30):
                        result['first_lidar'] = row['timestamp']
                        break
        result['braking_before_lidar'] = sum(r['carter']['speed']<.64 and
            (result['first_lidar'] is None or r['timestamp']<result['first_lidar']) for r in active)/30
    return {'run':run,'episode':episode,'rows':rows,'resolved':resolved,'stats':result}


items = [load(r) for r in manifest['runs'] if r['state']=='finished']
pairs = {}
for item in items:
    pairs.setdefault((item['run']['event'],item['stats']['scenario']),{})[item['run']['mode']] = item
comparisons = []
for key,pair in pairs.items():
    if set(pair)!={'fixed','nav2'}:
        continue
    a,b = pair['fixed'],pair['nav2']
    matched = bool(a['resolved'] and b['resolved'] and a['resolved']['spec']==b['resolved']['spec'] and
        a['resolved']['resolved']==b['resolved']['resolved'] and a['resolved']['base_config_sha256']==b['resolved']['base_config_sha256'])
    position_error = None
    if a['rows'] and b['rows']:
        ac,bc = a['rows'][0]['carter'],b['rows'][0]['carter']
        position_error = math.dist([ac['x'],ac['y']],[bc['x'],bc['y']])
        matched = matched and position_error<.02 and abs(ac['heading']-bc['heading'])<.01 and abs(ac['speed']-bc['speed'])<.02
    else:
        matched = False
    complete = a['stats']['status']=='completed' and b['stats']['status']=='completed'
    row = {'event':key[0],'scenario':key[1],'matched':matched,'initial_position_difference_m':position_error,
           'both_completed':complete,'fixed':a['stats'],'nav2':b['stats']}
    if matched and complete:
        row['delta_person_lateral'] = b['stats']['person_lateral']-a['stats']['person_lateral']
        row['delta_clearance'] = b['stats']['clearance']-a['stats']['clearance']
        if a['stats']['person_min_speed'] is not None and b['stats']['person_min_speed'] is not None:
            row['delta_person_min_speed'] = b['stats']['person_min_speed']-a['stats']['person_min_speed']
        active_a = [r for r in a['rows'] if r['person']['present']]
        active_b = [r for r in b['rows'] if r['person']['present']]
        if active_a and active_b:
            ta = np.array([r['timestamp'] for r in active_a]); tb = np.array([r['timestamp'] for r in active_b])
            mask = (ta>=tb[0])&(ta<=tb[-1]); xy = np.array([[r['person']['x'],r['person']['y']] for r in active_a])[mask]
            other = np.c_[np.interp(ta[mask],tb,[r['person']['x'] for r in active_b]),np.interp(ta[mask],tb,[r['person']['y'] for r in active_b])]
            row['person_response_rms_difference_m'] = float(np.sqrt(np.mean(np.sum((xy-other)**2,axis=1))))
    comparisons.append(row)


def limits(pair):
    arrays=[]
    for item in pair.values():
        if not item['resolved']:
            continue
        h=item['resolved']['resolved']['person']; arrays.extend([h['start'],h['goal'],*item['resolved']['resolved']['path']])
    xy=np.array(arrays)
    return xy.min(axis=0)-1.2,xy.max(axis=0)+1.2


def backdrop(ax,event,low,high):
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x (m)',ylabel='y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.15)
    if event=='occlusion':
        ax.add_patch(Rectangle((3.2,11.7),1.05,1.,color='#777777',alpha=.75,label='Existing pillar region'))


def png(fig):
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=135,bbox_inches='tight');plt.close(fig)
    return '<img style="width:100%" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'


figures=[]
for event,scenario in [('crossing','gap0_person1.0'),('occlusion','window2_person1.0')]:
    pair=pairs.get((event,scenario),{})
    if set(pair)!={'fixed','nav2'} or not all(i['rows'] for i in pair.values()):
        continue
    low,high=limits(pair)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    backdrop(axes[0,0],event,low,high)
    h=pair['fixed']['resolved']['resolved']['person']; line=np.array([h['start'],h['goal']]); direction=line[1]-line[0];direction/=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    axes[0,0].plot(line[:,0],line[:,1],'k--',alpha=.6,label='Person nominal route')
    for mode,item in pair.items():
        rows=item['rows'];active=[r for r in rows if r['person']['present']]
        c=np.array([[r['carter']['x'],r['carter']['y']] for r in rows]); p=np.array([[r['person']['x'],r['person']['y']] for r in active])
        axes[0,0].plot(c[:,0],c[:,1],color=palette[mode],alpha=.5,label=mode+' Carter')
        axes[0,0].plot(p[:,0],p[:,1],color=palette[mode],linewidth=2.3,label=mode+' Person')
        axes[0,1].plot([r['timestamp'] for r in rows],[r['carter']['speed'] for r in rows],color=palette[mode],label=mode)
        axes[1,0].plot([r['timestamp'] for r in active],[r['person']['speed'] for r in active],color=palette[mode],label=mode)
        axes[1,1].plot([r['timestamp'] for r in active],np.abs((p-line[0])@normal),color=palette[mode],label=mode)
    axes[0,0].legend(fontsize=8)
    for ax,title,ylabel in [(axes[0,1],'Carter speed','m/s'),(axes[1,0],'Person speed','m/s'),(axes[1,1],'Person lateral deviation','m')]:
        ax.set(title=title,xlabel='Episode time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend();ax.axvline(h['spawn_time'],color='gray',linestyle=':',label='spawn')
    fig.suptitle(event+' | Fixed vs Nav2 | '+scenario,fontsize=16);fig.tight_layout()
    figures.append((event,png(fig)))
    video=study/(event+'_fixed_vs_nav2.mp4')
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),dpi=100)
    end=max(i['rows'][-1]['timestamp'] for i in pair.values())
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,720))
    if not writer.isOpened():raise RuntimeError('Cannot create comparison video')
    for t in np.arange(0,end+1/30,1/15):
        for ax,(mode,item) in zip(axes,pair.items()):
            ax.clear();backdrop(ax,event,low,high);rows=item['rows']
            index=min(round(t*30),len(rows)-1);r=rows[index];c=r['carter'];p=r['person']
            ax.plot(line[:,0],line[:,1],'k--',alpha=.4)
            history=rows[:index+1]
            ax.plot([x['carter']['x'] for x in history],[x['carter']['y'] for x in history],color='#2474b5',alpha=.55)
            active=[x for x in history if x['person']['present']]
            if active:ax.plot([x['person']['x'] for x in active],[x['person']['y'] for x in active],color='#e27626',alpha=.6)
            ax.add_patch(Circle((c['x'],c['y']),.55,color='#2474b5',alpha=.7))
            ax.arrow(c['x'],c['y'],.65*math.cos(c['heading']),.65*math.sin(c['heading']),width=.03,color='#173d64')
            if p['present']:ax.add_patch(Circle((p['x'],p['y']),.23,color='#e27626'))
            pv=f"{p['speed']:.2f}" if p['present'] else '--'
            ax.set_title(f"{mode.upper()} | final: {item['stats']['status']} | t={t:.2f}s\nCarter {c['speed']:.2f} m/s | Person {pv} m/s",fontsize=12)
        fig.suptitle(event+' | matched scenario, different Carter controller',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.95));fig.canvas.draw()
        frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
        writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    writer.release();plt.close(fig)
    cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release()
    if count<15:raise RuntimeError('Comparison video verification failed')

fig,axes=plt.subplots(2,4,figsize=(14,7))
for row,event in enumerate(['crossing','occlusion']):
    values=[2.,1.,0.] if event=='crossing' else [3.,2.5,2.]
    for col,(metric,title) in enumerate([('delta_person_lateral','Person lateral B-A (m)'),('delta_person_min_speed','Person min speed B-A (m/s)'),('delta_clearance','Clearance B-A (m)'),('success','Nav2 goal success')]):
        grid=np.full((3,3),np.nan)
        for comparison in comparisons:
            if comparison['event']!=event:continue
            item=pairs[(event,comparison['scenario'])]['nav2']
            if not item['resolved']:continue
            spec=item['resolved']['spec'];v=spec['event']['arrival_gap' if event=='crossing' else 'reaction_window'];speed=spec['person']['speed']
            grid[values.index(v),[.8,1.,1.2].index(speed)]=(float(comparison['nav2']['status']=='completed' and not comparison['nav2']['collision']) if metric=='success' else comparison.get(metric,np.nan))
        ax=axes[row,col];extent=max(.01,float(np.nanmax(np.abs(grid)))) if np.isfinite(grid).any() else 1.
        im=ax.imshow(grid,cmap='RdYlGn' if metric=='success' else 'coolwarm',vmin=0 if metric=='success' else -extent,vmax=1 if metric=='success' else extent)
        ax.set(xticks=range(3),xticklabels=['0.8','1.0','1.2'],yticks=range(3),yticklabels=values,title=title,xlabel='Person speed (m/s)',ylabel=event+' '+('gap' if event=='crossing' else 'window')+' (s)')
        for i in range(3):
            for j in range(3):ax.text(j,i,'--' if not np.isfinite(grid[i,j]) else f'{grid[i,j]:.2f}',ha='center',va='center',fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.75)
fig.tight_layout();heatmap=png(fig)
summary={'status':manifest['status'],'runs_finished':len(items),'runs_planned':len(manifest['runs']),
         'comparisons':comparisons,'metrics_definitions':{
             'carter_min_speed':'After Person spawn and more than 1 m from goal; avoids startup and terminal stopping.',
             'braking_duration':'Seconds below 0.64 m/s in that interval, including static-obstacle response.',
             'person_min_speed':'Excludes first second after spawn and final 0.5 m near Person goal.',
             'lateral':'Maximum perpendicular distance from the nominal Person line.',
             'delta':'Nav2 minus Fixed. Only complete, condition-matched pairs receive difference metrics.',
             'first_lidar':'Offline geometric association of actual returns within 0.30 m of Person center.',
             'first_visible':'Existing geometric camera visibility; distinct from LiDAR detection.'}}
(study/'comparison.json').write_text(json.dumps(summary,indent=2))
def fmt(v):return '--' if v is None else f'{v:.3f}' if isinstance(v,(float,int)) and not isinstance(v,bool) else str(v)
columns=[('event','事件'),('scenario','配置'),('mode','控制'),('status','完成'),('event_valid','事件有效'),('carter_min_speed','车最低速度'),('braking_duration','减速时长'),('person_min_speed','人最低速度'),('person_lateral','人最大侧移'),('clearance','最小间隙'),('ttc','最小TTC'),('first_lidar','首次雷达检测'),('first_visible','首次相机可见'),('collision','碰撞')]
table='<table><tr>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>'
for item in items:table+='<tr>'+''.join('<td>'+fmt(item['stats'].get(k))+'</td>' for k,_ in columns)+'</tr>'
table+='</table>'
html='''<!doctype html><meta charset="utf-8"><title>Fixed vs Nav2 — Crossing / Occlusion</title>
<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1400px;margin:32px auto;color:#203046;padding:0 20px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;border:1px solid #d7dee6}th{background:#edf3f9}h1,h2{color:#173d64}.note{background:#fff4d8;padding:18px;border-radius:8px}video{width:100%}</style>
<h1>Fixed Carter vs Nav2 Carter</h1><p>Crossing 与 Occlusion 配对对照；双方状态均来自实际仿真。</p>'''
html+=f'<p>已完成 {len(items)} / {len(manifest["runs"])} 条运行。批次状态：{manifest["status"]}。</p>'
html+='''<div class="note">只改变 Carter 控制器，人物初始条件、HuNav 参数、雷达和物理参数保持一致。Fixed 以 0.8 m/s 跟踪直线，并在终点减速；Nav2 通过实际雷达闭环控制。每个条件仅运行一对，结果为描述性对照，不代表统计显著性。Occlusion 中车也会响应静态柱子；人物仍由现有 HuNav 全状态交互驱动，不能把所有运动差异归因于首次看见人物。未完成或条件不匹配的配对不计算全程差值。</div>'''
for event,figure in figures:
    comparison=next(c for c in comparisons if c['event']==event and c['scenario']==('gap0_person1.0' if event=='crossing' else 'window2_person1.0'))
    html+=f'<h2>{event} 代表场景</h2><p>条件匹配：{comparison["matched"]}；双方运行均完成：{comparison["both_completed"]}；初始车位置差：{fmt(comparison["initial_position_difference_m"])} m。</p>'+figure
    html+=f'<video controls preload="metadata" src="{event}_fixed_vs_nav2.mp4"></video>'
html+='<h2>配置热力图</h2><p>B-A：正值表示 Nav2 更大。侧移越小不一定越好，需结合碰撞、间隙、成功情况解释。-- 为未完成或不可比较。</p>'+heatmap
html+='<h2>完整结果表</h2><p>速度 m/s，侧移与间隙 m，时长和 TTC s。车最低速度与减速时长排除了起步和距目标 1 m 内的收车。</p>'+table
html+='<p>首次相机可见为几何可见性指标；首次雷达检测来自实际扫描回波的离线关联。轨迹动画中的柱子为既有遮挡区域示意边界。</p>'
(study/'report.html').write_text(html,encoding='utf-8')
print('REPORT',str(study/'report.html'),json.dumps({'runs':len(items),'pairs':len(comparisons),'matched':sum(c['matched'] for c in comparisons)}),flush=True)
