import argparse
import fcntl
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'hunav-core'
OUT = ROOT/'mobile-navigation/output'
NAME = os.environ.get('DEMO_NAME', 'demo_public_v1')
if not NAME or Path(NAME).name != NAME or NAME in ('.', '..'):
    raise ValueError('DEMO_NAME must be a directory name')
STUDY = OUT/NAME
RUNTIME = CORE/'runtime'/NAME
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['simulate', 'render', 'compose', 'all'], default='all')
parser.add_argument('--only')
parser.add_argument('--snapshot', type=float)
args = parser.parse_args()
STUDY.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)
lock = (CORE/'runtime/demo.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status_path = STUDY/'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {'runs':{}, 'renders':{}, 'phase':'preparing'}
status.pop('error',None)


def save():
    status['updated'] = time.time()
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(status_path)


def circle(cx, cy, rx, ry, period=16., phase=0.):
    return [[i/10, cx+rx*math.cos(2*math.pi*(i/10)/period+phase),
             cy+ry*math.sin(2*math.pi*(i/10)/period+phase)] for i in range(251)]


def scenario(name, event, route, tracks, **extra):
    result = dict(scenario_id=name, source='parameterized', event_type=event, seed=0,
                  max_seconds=45, carter={'route':route, 'speed':.8}, tracks=tracks)
    result.update(extra)
    return result


configs = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT/'examples/scenarios').glob('*.json'))}
if not configs:
    raise RuntimeError('No example scenarios found')
if args.only is not None and args.only not in configs:
    raise ValueError('Unknown scenario: '+args.only)


def container_path(host_path):
    return '/work/output/'+str(host_path.relative_to(OUT))


def simulate(name, spec):
    tag = 'ab_'+NAME+'_'+name
    logs = CORE/'runtime'/tag
    logs.mkdir(exist_ok=True)
    scenario_file = STUDY/(name+'.json')
    if scenario_file.exists() and json.loads(scenario_file.read_text()) != spec:
        raise RuntimeError('Preserve existing run; use a new suite version for changed scenarios')
    scenario_file.write_text(json.dumps(spec, indent=2))
    episode = STUDY/name
    if (episode/'result.json').exists():
        previous=json.loads((episode/'result.json').read_text())
        if not previous['quality']['valid'] or previous['outcome']['program_error']:
            raise RuntimeError('Previous episode needs a new recording: '+str(episode))
        status['runs'][name] = {'state':'completed', 'episode':str(episode)}
        save(); return
    if episode.exists():
        raise RuntimeError('Incomplete episode exists: '+str(episode))
    status['phase'], status['active'] = 'simulation', name
    status['runs'][name] = {'state':'running', 'episode':str(episode), 'log':str(logs/'isaac.log')}
    save()
    stack_log = (logs/'endpoint.log').open('w')
    stack = subprocess.Popen(['docker','exec','re3sim-nav2','bash','/work/crossing/nav2_stack.sh',
                              tag,'nav2',json.dumps(spec['carter']['route'])], stdout=stack_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic()+30
    while not (CORE/'runtime/carter_ros.sock').exists():
        if stack.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('Navigation endpoint did not start')
        time.sleep(.1)
    with (logs/'isaac.log').open('w') as log:
        sim = subprocess.run(['docker','exec','re3sim-mobile-scene64','bash','/work/python_env.sh',
            '/repo/hunav-core/crossing/demo_run.py',container_path(scenario_file),
            '--output',container_path(episode)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
    stack_code = stack.wait(timeout=100)
    stack_log.close()
    if sim.returncode or stack_code:
        raise RuntimeError('Simulation infrastructure failed: '+str(logs))
    import sys
    sys.path.insert(0,str(CORE/'crossing'))
    from episode_evaluator import evaluate_episode
    config = json.loads((CORE/'crossing/evaluator_config_v1.json').read_text())
    result = evaluate_episode(episode,config)
    (episode/'result.json').write_text(json.dumps(result,indent=2))
    if not result['quality']['valid'] or result['outcome']['program_error']:
        raise RuntimeError('Episode recording invalid: '+str(episode))
    status['runs'][name].update(state='completed',quality=result['quality']['valid'],
        outcome=result['outcome'],metrics=result['metrics'])
    save()
    print('SIMULATED',name,result['outcome'],flush=True)


def render_episode(name, episode, views=False):
    output_name = NAME+'/'+name+('_preview' if args.snapshot is not None else '_views')
    result_file = OUT/(output_name+'_result.json')
    if args.snapshot is None and result_file.exists() and json.loads(result_file.read_text()).get('success'):
        status['renders'][name] = {'state':'completed','video':str(OUT/(output_name+'.mp4'))}
        save(); return
    status['phase'], status['active'] = 'rendering',name
    save()
    socket_path = ROOT/'lhm-human/output/social_avatar.sock'
    if socket_path.exists():
        raise RuntimeError('Avatar endpoint is already in use')
    avatar_log = (RUNTIME/(name+'_avatar.log')).open('w')
    avatar = subprocess.Popen(['docker','exec','re3sim-lhm-human','/work/venv/bin/python',
        '/work/serve_social_avatar.py'],stdout=avatar_log,stderr=subprocess.STDOUT)
    deadline = time.monotonic()+90.
    while not socket_path.exists():
        if avatar.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('Avatar renderer startup failed')
        time.sleep(.2)
    eye, target = ([.3,0.,2.3],[2.,10.,.8]) if name=='multi' else ([.3,0.,2.3],[1.8,10.,.8])
    if name=='occlusion':
        eye,target=[-.3,4.,2.3],[2.6,12.5,1.]
    if name=='crowdes':
        eye,target=[.0,1.,3.],[1.5,9.5,.8]
    cmd = ['docker','exec','re3sim-mobile-scene64','bash','/work/python_env.sh',
        '/repo/hunav-core/crossing/demo_render.py','--episode',episode,'--name',output_name,
        '--overview-eye',*map(str,eye),'--overview-target',*map(str,target)]
    if views: cmd.append('--two-views')
    if args.snapshot is not None: cmd += ['--snapshot',str(args.snapshot)]
    with (RUNTIME/(name+'_render.log')).open('w') as log:
        result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    if result.returncode and avatar.poll() is None and socket_path.exists():
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
    avatar.wait(timeout=90)
    avatar_log.close()
    if result.returncode or not result_file.exists() or not json.loads(result_file.read_text()).get('success'):
        raise RuntimeError('Rendering failed: '+str(RUNTIME/(name+'_render.log')))
    status['renders'][name] = {'state':'preview' if args.snapshot is not None else 'completed',
                               'video':str(OUT/(output_name+'.mp4'))}
    save()
    print('RENDERED',name,flush=True)


try:
    if args.phase in ['simulate','all']:
        for name,spec in configs.items():
            if args.only is None or args.only==name: simulate(name,spec)
    if args.phase in ['render','all']:
        episodes = {name:container_path(STUDY/name) for name in configs}
        for name,episode in episodes.items():
            if args.only is None or args.only==name:
                render_episode(name,episode,name in ['crossing','occlusion'])
    if args.phase in ['compose','all']:
        status['phase']='composing';save()
        with (RUNTIME/'compose.log').open('w') as log:
            subprocess.run(['docker','exec','re3sim-mobile-scene64','bash','/work/python_env.sh',
                '/repo/hunav-core/crossing/demo_compose.py',container_path(STUDY)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        status['phase']='completed';save()
except Exception as exc:
    status['phase'],status['error']='needs_attention',repr(exc);save()
    raise
