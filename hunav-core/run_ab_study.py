import argparse
import fcntl
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

CORE = Path(__file__).resolve().parent
OUTPUT = CORE.parent/'mobile-navigation/output'
parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--limit', type=int)
args = parser.parse_args()
lock = (CORE/'runtime/demo.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
critical = ['runner.py', 'spec.py', 'occlusion.py', 'metrics.py', 'visibility.py',
            'carter_ros.py', 'fixed_session.py', 'nav2_session.py', 'ros_endpoint.py', 'nav2_params.yaml']
hashes = {n: hashlib.sha256((CORE/'crossing'/n).read_bytes()).hexdigest() for n in critical}
if args.resume:
    study = Path(args.resume).resolve()
    assert study.parent == OUTPUT.resolve()
    manifest = json.loads((study/'manifest.json').read_text())
    if manifest['source_sha256'] != hashes:
        raise RuntimeError('Study source changed; do not mix controller revisions')
else:
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    study = OUTPUT/('ab_study_'+stamp)
    study.mkdir()
    ordered = [('crossing', 'scenarios/gap0_person1.0.yaml'),
               ('occlusion', 'occlusion_scenarios/window2_person1.0.yaml')]
    for event, folder in [('crossing', 'scenarios'), ('occlusion', 'occlusion_scenarios')]:
        for p in sorted((CORE/'crossing'/folder).glob('*.yaml')):
            item = (event, folder+'/'+p.name)
            if item not in ordered:
                ordered.append(item)
    manifest = {'study': str(study), 'stamp': stamp, 'source_sha256': hashes,
                'status': 'running', 'runs': []}
    for event, scenario in ordered:
        for mode in ['fixed', 'nav2']:
            manifest['runs'].append({'event': event, 'scenario': scenario, 'mode': mode, 'state': 'pending'})


def save():
    temp = study/'manifest.tmp'
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(study/'manifest.json')


for name in ['re3sim-hunav-core', 're3sim-mobile-scene64', 're3sim-nav2']:
    if subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',name],text=True).strip()!='true':
        subprocess.run(['docker','start',name],check=True)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
    probe.settimeout(2.)
    probe.connect(str(CORE/'runtime/hunav.sock'))
manifest['status'] = 'running'
save()
print('STUDY', str(study), flush=True)
finished = 0
try:
    for index, run in enumerate(manifest['runs']):
        if run['state'] == 'finished':
            continue
        if args.limit is not None and finished>=args.limit:
            break
        tag = 'ab_'+manifest['stamp']+'_'+str(index).zfill(2)+'_'+run['event']+'_'+run['mode']
        logs = CORE/'runtime'/tag
        logs.mkdir(exist_ok=False)
        route = [[1.3,5.],[1.3,15.]] if run['event']=='crossing' else [[2.75,8.5],[2.75,15.5]]
        run.update(state='running', tag=tag, logs=str(logs))
        save()
        with (logs/'endpoint.log').open('w') as log:
            stack = subprocess.Popen(['docker','exec','re3sim-nav2','bash','/work/crossing/nav2_stack.sh',
                tag,run['mode'],json.dumps(route)],stdout=log,stderr=subprocess.STDOUT)
        deadline = time.monotonic()+15.
        while not (CORE/'runtime/carter_ros.sock').exists():
            if stack.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('ROS stack startup failed: '+str(logs))
            time.sleep(.1)
        cmd = ['docker','exec','re3sim-mobile-scene64','bash','/work/python_env.sh',
               '/repo/hunav-core/crossing/runner.py','--ros-control','--output','/work/output/'+tag,
               '/repo/hunav-core/crossing/'+run['scenario']]
        if run['mode']=='fixed':
            cmd.append('--fixed-control')
        with (logs/'isaac.log').open('w') as log:
            sim = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        stack_code = stack.wait(timeout=100)
        result = OUTPUT/tag/'summary.json'
        if sim.returncode or stack_code or not result.exists():
            raise RuntimeError('Infrastructure failure: '+str(logs))
        m = json.loads(result.read_text())[0]
        run.update(state='finished', episode=str(OUTPUT/tag/('00_'+Path(run['scenario']).stem)),
                   metrics=m)
        finished += 1
        save()
        print('RUN', index+1, '/',len(manifest['runs']),run['event'],run['mode'],
              Path(run['scenario']).stem,m['status'],m.get('minimum_clearance_m'),flush=True)
    manifest['status'] = 'completed' if all(r['state']=='finished' for r in manifest['runs']) else 'paused_after_representatives'
except Exception as exc:
    manifest['status'], manifest['error'] = 'infrastructure_error', repr(exc)
    raise
finally:
    save()
print('BATCH_STATUS',manifest['status'],flush=True)
