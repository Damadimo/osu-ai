# osu-ai

A small vision model that plays osu!lazer on macOS from screen capture. It passed Cold Green Eyes
[apple's Insane] (4.19★, never seen in training) with an S rank, 99.64% accuracy and no misses.

## How it works

- **Eyes and hands.** `hands_app/` is a macOS helper app (OsuHands). It captures osu!'s window
  with ScreenCaptureKit and posts mouse moves and Z/X presses with CGEvent, only while osu! is the
  frontmost app. It also draws a visible cursor and Z/X keys over osu! for people watching; the
  capture leaves that layer out, so the model never sees it.
- **Model.** `osu_ai/vision.py`: a 2.9M-parameter CNN (MLX). Input: the last 4 frames, 320x240
  grayscale. Output: where the next two objects are, how many ms until each must be hit, and
  whether a slider or spinner is active.
- **Policy.** `osu_ai/policy.py` turns that into cursor moves and key presses at the predicted
  times, sent early by the measured input delay (`--latency-ms`).
- **Game setup.** osu! runs offline, logged out, in a 1024x768 window with a plain grayscale
  skin, 100% background dim and no storyboard, HUD or hit animations, so it looks like the
  training frames.
- **Scoring.** After a run, lazer's own replay is exported. The hit offsets in it give the
  latency correction for the next run.

## Setup

Needs an Apple silicon Mac on macOS 26, [uv](https://docs.astral.sh/uv/), the Xcode Command Line
Tools and the .NET SDK (only for setting osu!'s ruleset options).

```bash
uv sync --extra train
brew install --cask osu                      # osu!lazer; start it once, then quit it

# skin: generate and import it, then quit osu! for the config steps
uv run --with pillow scripts/make_skin.py data/skin/osu-jev-minimal
open -a "osu!" data/skin/osu-jev-minimal.osk
REALM="$HOME/Library/Application Support/osu/client.realm"
dotnet run --project tools/lazer_realm -- "$REALM"   # lists skins with their GUIDs
uv run scripts/lazer_setup.py --skin <skin GUID>      # --restore puts the old config back
dotnet run --project tools/lazer_realm -- "$REALM" \
  --set HitAnimations=False --set SnakingInSliders=False --set SnakingOutSliders=False --set ShowCursorTrail=False

# OsuHands: build, then allow Screen Recording and Accessibility in System Settings
hands_app/build.sh
uv run python -m osu_ai.hands_client grant
uv run python -m osu_ai.hands_client status   # all three permissions should be True

# maps: download sets from osu.direct, then import copies (osu! deletes what it imports)
uv run scripts/fetch_osz.py download
uv run python -c "from pathlib import Path; from osu_ai import lazer; lazer.import_files(sorted(Path('data/osz').glob('*.osz')))"
```

## Play a map

```bash
uv run python -c "from osu_ai import lazer; lazer.launch()"
uv run --extra train scripts/play_real.py data/lazer_osu/827932.osu --latency-ms 9
```

- Don't touch the mouse or switch apps during a run. Switching apps (or Mission Control) cuts
  the run short; that is also how to stop it.
- In a new osu! session, the sign-in panel opens the first time you leave the start screen.
  Close it with Esc twice before running (the script doesn't handle it yet). Never sign in.
- Each run saves its frames and a report to `runs/real/play/`, and prints a suggested
  `--latency-ms` for next time.
- `scripts/score_lazer.py <map.osu>` plays a map in the simulator with lazer's scoring rules;
  `scripts/rehearse_real.py` runs the real play loop against a simulated game with screen and
  input delays.

## Training

Training frames are drawn on the fly by `osu_ai/lazer_frames.py`, which reproduces what lazer
shows with this skin at any game time and for any play: hits, misses, early and late clicks.

```bash
uv run --extra train scripts/train_lazer.py --steps 24000 --out models/osunet-lazer.safetensors
scripts/train_supervised.sh <out> <init> <start step> <total steps>   # same, restarting after crashes
```

- Maps: 1,563 for training, 82 for validation and 31 held out, including every map played in the
  real game (`data/real_maps.json`). Sources: `scripts/fetch_maps.py` and `scripts/fetch_osz.py`.
- Augmentation: random plays, small zoom and shift errors, brightness, gamma, blur and noise,
  jittered and skipped frames, and extra streams and stacks.
- The included model trained for 24,000 steps (about 2.5 hours on an M5 Pro). On the held-out
  maps it finds the next object 99.9% of the time, with a 0.18 osu!px aim error and a 3.5 ms
  median timing error.

## In progress

- **Map backgrounds.** The model has only seen black backgrounds. The plan is to draw training
  frames over map background images at random dim levels and retrain from the current model.
- **Harder maps.** Next real-game targets are the 4.5★ maps in `data/real_maps.json`
  (Koigokoro, Zen Zen Zense [Habi's Insane], Tower of Heaven).
- **Spinners.** One real run spun a spinner that osu! still judged a miss; not yet explained.
- **Sign-in panel.** Make `real.Driver` close it on its own.

Earlier experiments with text models (Jev, Laya) are in `scripts/watch.py`, `scripts/play.py` and
`scripts/train_lora.py`; they could not read the timing well enough to play.

Offline use only: never run this logged in or submit its scores.
