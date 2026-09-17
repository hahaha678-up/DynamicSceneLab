import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path

core = Path(__file__).resolve().parent
lock = (core/'runtime/demo.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
tag = 'nav2_crossing_'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
logs = core/'runtime'/tag
logs.mkdir()
for container in ['re3sim-nav2', 're3sim-hunav-core', 're3sim-mobile-scene64']:
    running = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', container], text=True).strip()
    if running != 'true':
        subprocess.run(['docker', 'start', container], check=True)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
    probe.settimeout(2.)
    probe.connect(str(core/'runtime/hunav.sock'))
stack_log = (logs/'endpoint.log').open('w')
stack = subprocess.Popen(['docker', 'exec', 're3sim-nav2', 'bash', '/work/crossing/nav2_stack.sh', tag],
                         stdout=stack_log, stderr=subprocess.STDOUT)
print('NAV2_RUN', tag, flush=True)
for _ in range(100):
    if (core/'runtime/carter_ros.sock').exists():
        break
    if stack.poll() is not None:
        raise RuntimeError('Nav2 startup failed: '+str(logs))
    time.sleep(.1)
sim_log = (logs/'isaac.log').open('w')
sim = subprocess.run(['docker', 'exec', 're3sim-mobile-scene64', 'bash', '/work/python_env.sh',
    '/repo/hunav-core/crossing/runner.py', '--ros-control', '--output', '/work/output/'+tag,
    '/repo/hunav-core/crossing/scenarios/gap0_person1.0.yaml'],
    stdout=sim_log, stderr=subprocess.STDOUT, timeout=240)
print('ISAAC_EXIT', sim.returncode, flush=True)
print('NAV2_STACK_EXIT', stack.wait(timeout=100), flush=True)
result = core.parent/'mobile-navigation/output'/tag/'summary.json'
data = json.loads(result.read_text())
print('RESULT', str(result), json.dumps(data), flush=True)
if len(data)!=1 or data[0]['status']!='completed' or data[0].get('collision', True):
    raise SystemExit('Navigation episode failed; inspect saved evidence')

subprocess.run(['docker', 'exec', 're3sim-mobile-scene64', 'bash', '/work/python_env.sh',
    '/repo/hunav-core/crossing/validate_nav2_episode.py', tag], check=True)
