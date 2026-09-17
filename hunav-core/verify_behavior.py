import json
import math
from pathlib import Path

from hunav_client import HuNavClient
from scenarios import HUMAN, HUMAN_START_DELAY, prescribed_robot

root = Path('/work')
client = HuNavClient(root / 'runtime/hunav.sock')
runs = {}
try:
    for case in ('stop', 'cross', 'bypass', 'cross_repeat', 'no_robot'):
        client.request(op='reset', human=HUMAN)
        rows = []
        for step in range(451):
            t = step/30
            robot = (prescribed_robot(case.replace('_repeat', ''), t) if case != 'no_robot' else
                     {'xy': [100.,100.], 'velocity': [0.,0.], 'yaw': 0., 'radius': .55})
            if t < HUMAN_START_DELAY:
                human = {'t': t, 'xy': HUMAN['start'], 'yaw': HUMAN['yaw'], 'velocity': [0.,0.],
                         'speed': 0., 'service_ms': 0., 'behavior': 1, 'arrived': False}
            else:
                human = client.request(op='step', t=t, robot=robot)
            rows.append({'t': t, 'robot': robot, 'human': human})
        runs[case] = rows
        minimum = min(math.dist(row['human']['xy'], row['robot']['xy'])-.78 for row in rows)
        error = math.dist(rows[-1]['human']['xy'], HUMAN['goal'])
        print(case, 'minimum_disc_gap', round(minimum, 3), 'human_goal_error', round(error, 3), flush=True)
    repeated = max(math.dist(a['human']['xy'], b['human']['xy']) for a, b in zip(runs['cross'], runs['cross_repeat']))
    differences = {case: max(math.dist(a['human']['xy'], b['human']['xy']) for a, b in zip(runs['cross'], runs[case])) for case in ('stop', 'bypass')}
    report = {'backend': 'official HuNavSim 2.0 Regular/SFM', 'human_config': HUMAN,
              'repeat_max_error_m': repeated, 'max_trajectory_difference_m': differences, 'runs': runs}
    (root / 'output/behavior_comparison.json').write_text(json.dumps(report))
    assert repeated < 1e-5, repeated
    assert min(differences.values()) > .2, differences
    response_difference = max(math.dist(a['human']['xy'], b['human']['xy']) for a,b in zip(runs['cross'],runs['no_robot']))
    response_window = [row for row in runs['cross'] if row['t'] > HUMAN_START_DELAY+.8 and not row['human']['arrived']]
    assert response_difference > .3
    assert min(row['human']['speed'] for row in response_window) < HUMAN['speed']*.6
    for case, rows in runs.items():
        assert all(row['human']['behavior'] == 1 for row in rows), case
        assert max(row['human']['speed'] for row in rows) <= HUMAN['speed']+.001, case
        assert min(math.dist(row['human']['xy'], row['robot']['xy'])-.78 for row in rows) > 0., case
        assert rows[-1]['human']['arrived'] and rows[-1]['human']['speed'] == 0., case
    client.request(op='reset', human=HUMAN)
    robot = prescribed_robot('cross', 0.)
    initial = client.request(op='step', t=0., robot=robot)
    assert math.dist(initial['xy'], HUMAN['start']) < 1e-8
    for invalid_time in (0., -.1, .3):
        try:
            client.request(op='step', t=invalid_time, robot=robot)
        except RuntimeError as exc:
            assert 'time must increase' in str(exc)
        else:
            raise AssertionError('invalid simulation time was accepted')
    client.request(op='step', t=1/30, robot=robot)
    print('BEHAVIOR_VERIFIED', repeated, differences, flush=True)
finally:
    client.close()
