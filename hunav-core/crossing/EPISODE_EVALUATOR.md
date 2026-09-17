# Crossing offline EpisodeEvaluator

`episode_evaluator.py` reads existing episode files and writes `result.json` in
each episode plus one batch `summary.csv`. It uses Python's standard library;
no simulator, ROS node, dependency installation, or rollout is started.

## Run

```bash
python3 -B /path/to/DynamicSceneLab/hunav-core/crossing/episode_evaluator.py \
  --input-root /path/to/DynamicSceneLab/mobile-navigation/output \
  --event crossing \
  --summary /path/to/DynamicSceneLab/mobile-navigation/output/summary.csv
```

Use repeated `--episode /absolute/episode/path` to evaluate selected episodes,
including malformed episodes that cannot be discovered by event metadata.
Use `--replace-generated` to update existing evaluator results explicitly.
Raw logs, old `metrics.json`, and old `summary.json` are never changed.
CSV nulls are the literal `null`; they do not mean zero. All selected runs,
including invalid runs and ordinary safe outcomes, remain in the summary.

## Frozen experiment configuration

`evaluator_config_v1.json` fixes the proxy clearance target to **0.25 m**, with
threshold version `proxy_clearance_025m_v1`. This is a project threshold, not a
safety standard. Every result embeds the entire configuration and the evaluator
code hash. Change thresholds only in a new versioned configuration and do not
mix configurations in the same search or comparison.

The legacy Crossing adapter uses Carter radius 0.55 m and Person radius 0.23 m,
matching the existing metric definition. It records any radius taken from the
versioned legacy profile. New input formats must supply radii explicitly.

## Metric definitions

- Time is episode-relative simulation time. Nonfinite, duplicate or decreasing
  timestamps invalidate the episode. Missing state intervals invalidate search
  feedback; the permitted gap is 1.5 times recorded state cadence, or 0.2 s when
  cadence is unavailable.
- Pair metrics use actual XY positions while Person is present, through the
  first recorded controller terminal state. The minimum center distance uses
  linear relative motion between adjacent samples, with no interpolation across
  absent-person intervals. Proxy clearance subtracts both radii. This is a
  swept two-dimensional disc test, not physical-contact ground truth.
- TTC assumes constant current relative velocity and uses a 10 s horizon.
  No predicted contact is `null`; existing overlap is zero. Missing velocity
  data makes TTC unavailable and excludes the run from feedback.
- Response statistics select samples after Carter first exceeds 0.05 m/s,
  with Person present, center distance at most 3 m, and Carter more than 1 m
  from its goal. Adjacent samples must both satisfy the selection. This excludes
  pre-start waiting, recorded post-terminal idle, and goal-approach braking.
- Speed is the norm of recorded actual XY velocity. Deceleration uses the
  actual positive time delta. Stop time integrates time below 0.03 m/s using
  linear interpolation; braking time sums intervals below -0.2 m/s². A stopped
  behavior flag requires at least 0.2 s accumulated stop time. These behavior
  flags do not contribute to the clearance objective.
- Path length sums XY displacement through the controller's recorded terminal
  state. Heading changes are not included. Arrival time requires a timestamped
  successful controller status; legacy joint-episode completion cannot supply
  an unrecorded Carter arrival time.
- Task outcome comes from recorded task/controller status. A legacy
  `runtime_error` wrapping an explicit FollowPath abort is classified as
  `NAV_ABORT`. Other runtime errors are `PROGRAM_ERROR` and invalid for search.
  Timeout and navigation abort do not by themselves make recorded data invalid.
- Physical contact is a separate nullable channel. Optional per-state
  `contacts` keys are `person_carter`, `carter_environment`, and
  `person_environment`. A false result requires complete false coverage; one
  recorded true event is sufficient to report observed contact. Legacy absence
  of contact records is never interpreted as collision-free.
- Sensor and applied-command snapshots are checked when recorded. A changing
  offset between absolute simulation sensor time and episode-relative time,
  malformed LiDAR data, explicit sensor/control failure, or stale commands while
  navigating invalidate feedback. Stale commands after success do not.
- The existing applied `cmd_vel` snapshot is associated with the state sample's
  simulation time. Original ROS command receive times are not reconstructed.
- Detection time, reaction latency, and detour classification remain `null`
  with reasons because this version has no reliable unified detector-event or
  detour-classification contract.

## Results and feedback

`quality` reports validity, missing fields and issues. Missing optional channels
such as contact sensors or RNG seeds are explicit and do not automatically erase
usable geometric measurements. Invalid runs may retain diagnostic measurements;
their `rho` is always null.

`outcome` separates navigation outcome, safety outcome and behavior flags.
`metrics` contains measured values. No overall difficulty or weighted risk score
is generated.

`search_feedback.rho = min_clearance_m - 0.25` for valid measured episodes.
Negative values violate the configured proxy-clearance target. Fixed-controller
baselines retain this descriptive value but are not eligible for Nav2 CEM.
Eligibility also requires preserved Nav2 configuration, sensor observations, and
complete applied-command snapshots.
Unknown physical-contact or reaction-latency channels do not change this proxy
objective into a claim of physical safety. CEM is not implemented by this module.

`scenario` and `provenance` preserve source, parent, variants, controller/motion
mode, available seed, configuration hashes and raw-file hashes. Optional
`scenario.json` can carry `scenario_id`, `source`, `parent_id`, `seed`,
`motion_mode`, `variant_parameters`, `search_round`, and explicit proxy radii.
Both HuNav and trajectory-replay episodes use the same state schema with
`timestamp`, `carter`, and `person`; a Person-only replay cannot be evaluated as
a Carter navigation episode.

## Tests

```bash
TMPDIR=/path/to/DynamicSceneLab/mobile-navigation/crowdes-b/tmp \
  python3 -B /path/to/DynamicSceneLab/hunav-core/crossing/test_episode_evaluator.py
```

The tests cover analytic TTC and swept geometry, actual-delta derivatives,
absence/start/goal boundaries, safe stops, outcome classification, incomplete
and corrupt data, contact/command/sensor channels, provenance, and repeatable CLI
output without changes to original logs.
