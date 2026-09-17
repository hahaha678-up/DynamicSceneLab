import datetime
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


B = Path(__file__).resolve().parent
STATE = B/'runs/clean40_job.json'
STARTED = time.time()
DEADLINE = STARTED + 7*3600
MAX_SEEDS = 500
TARGET = 40


def state(phase, **details):
    value = {'phase': phase, 'target': TARGET, 'pid': os.getpid(),
             'started_at': datetime.datetime.fromtimestamp(STARTED).astimezone().isoformat(),
             'updated_at': datetime.datetime.now().astimezone().isoformat(),
             'deadline': datetime.datetime.fromtimestamp(DEADLINE).astimezone().isoformat(),
             'max_seeds': MAX_SEEDS, **details}
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATE)
    print(json.dumps(value), flush=True)


def collect():
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--name', 're3sim-crowdes-clean40-collect', '-v', f'{B}:/work/crowdes-b:rw',
               '--entrypoint', '/bin/bash', 're3sim:1.0.0-cuda118',
               '/work/crowdes-b/python_env.sh', '-B', '/work/crowdes-b/collect_clean.py']
    with (B/'runs/clean40_collection.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((B/'runs/clean40_progress.json').read_text())


def main():
    lock = (B/'runs/clean40.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior_run = 'scene55_exact_batch_20260916_seed03_19'
    exit_file = B/'runs'/f'{prior_run}.exitcode'
    state('waiting_for_existing_batch', run=prior_run)
    while not exit_file.exists():
        if time.time() > DEADLINE:
            raise RuntimeError('Existing batch did not finish within the overnight window')
        time.sleep(30)
    if exit_file.read_text().strip() != '0':
        raise RuntimeError('Existing batch failed; inspect its log before continuing')
    while True:
        state('validating')
        progress = collect()
        accepted = progress['accepted']
        if accepted == TARGET:
            state('ready_for_review', accepted=accepted, candidate_count=progress['candidate_count'],
                  dataset=str(B/'datasets/clean40_v1'))
            return
        next_seed = max(progress['completed_seeds'], default=-1)+1
        if next_seed >= MAX_SEEDS or time.time() >= DEADLINE:
            state('needs_attention', accepted=accepted, reason='overnight_budget_exhausted',
                  candidate_count=progress['candidate_count'])
            return
        if shutil.disk_usage(B).free < 2*1024**3:
            state('needs_attention', accepted=accepted, reason='less_than_2_GiB_free')
            return
        count = min(10, MAX_SEEDS-next_seed)
        name = f'scene55_exact_delivery_seed{next_seed:03d}_{next_seed+count-1:03d}'
        state('generating', accepted=accepted, run=name, first_seed=next_seed, seeds=count,
              candidate_count=progress['candidate_count'])
        with (B/'runs/clean40_batches.log').open('a') as log:
            subprocess.run(['bash', str(B/'run_batch.sh'), name, 'scene55_exact', str(count), str(next_seed)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=repr(error))
        raise
