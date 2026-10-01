# CartPole policy playback demo

The capacity comparison is at `demo-capacity/index.html`. It compares the saved
four-leaf baseline with the selected eight-leaf policy, with rescue, shared-success,
and remaining-failure examples. See [capacity results](CAPACITY.md).

The weighted follow-up has its own viewer at `demo-weighted/index.html`, with
an Original / Weighted candidate selector. See [weighting results](WEIGHTED_DAGGER.md).

The later dataset-aggregation trial has a separate viewer at
`demo-aggregation/index.html`. It compares the original tree with a worse
aggregated tree; see [the experiment report](DAGGER.md). The original demo below
has been preserved.

Open **demo/index.html** in your browser (double-click it in File Explorer).
It is self-contained: no server, network connection, Python process, or model
installation is needed for playback. Keep `trajectories.json` beside it if you
want the data download link to work.

## Using the viewer

- Choose **Both complete** or **Tree fails, teacher completes**.
- Use Play/Pause, Step +1, Restart, and the speed selector.
- Drag the step slider to inspect any point in the paired episode.
- The tree diagram highlights the exact rule path for the displayed state.
- "Next action" is chosen from that state. Advancing one step applies it.
- At an episode's final state there is no next action or active decision path.
  That panel freezes while the other policy continues.

Both policies start from the same reset seed, in independent environments, and
then follow their own actions. These are recorded trajectories, not browser-side
reinforcement learning. Canvas graphics are a schematic of the Gymnasium state.

## Recorded examples and aggregate evidence

| Reset seed | Teacher return | Tree return | Description |
|---|---:|---:|---|
| 9,000,001 | 500 | 500 | Both complete |
| 9,000,000 | 500 | 439 | Tree fails; teacher completes |

These examples were deliberately selected. They do not estimate reliability.
The generator independently reevaluates both policies on 100 reset seeds from
9,000,000 through 9,000,099. The tree reproduces **486.93 mean return and 67%
completion**. Complete episode returns, model checksums, and library versions
are included in `demo/trajectories.json`. No weights are updated.

The full-precision tree chooses actions in Python. The browser shows rounded
threshold labels but highlights the recorded exact path. A redundant split is
preserved rather than silently simplifying the saved policy.

## Rebuild the demo

From the project folder:

```powershell
.\.venv\Scripts\python.exe record_demo.py
```

This regenerates `demo/index.html` and `demo/trajectories.json` from the saved
models and `demo_template.html`. To record different example seeds in a separate
folder:

```powershell
.\.venv\Scripts\python.exe record_demo.py --seeds 9000002 9000075 --output demo-other
```

The viewer diagram is tailored to the current selected four-leaf tree. The seed
option changes examples, not the 100-episode aggregate evaluation.

## Verification

The recorder checks matching initial observations, step/return alignment, final
states, and every tree action/path against the saved model. `node test_demo.cjs`
checks playback controls, seeking, terminal freezing, and path clearing with a
mock DOM. It does not validate browser layout. Automated browser preview was
blocked by the browser tool's local-file URL policy, so visual verification in
a real browser remains outstanding.
