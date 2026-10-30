"""Configure osu!lazer for the vision model: a fixed window, a plain black playfield, no flashing.

  uv run scripts/lazer_setup.py            # osu! must not be running (it rewrites its config on exit)
  uv run scripts/lazer_setup.py --skin <GUID of the imported "osu-jev minimal" skin>
  uv run scripts/lazer_setup.py --dim 0.7  # show map backgrounds at 70% dim
  uv run scripts/lazer_setup.py --restore  # put the original config back

Everything set here is a normal player setting (background dim, storyboard off, ...).
The original files are kept as *.orig next to the config.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DATA = Path.home() / "Library/Application Support/osu"

FRAMEWORK = {
    "WindowMode": "Windowed",
    "WindowedSize": "1024x768",
    "WindowedPositionX": "0.5",
    "WindowedPositionY": "0.5",
    "ConfineMouseMode": "Never",
    "MinimiseOnFocusLossInFullscreen": "False",
}

GAME = {
    "ShowFirstRunSetup": "False",
    "DimLevel": "1",  # black background: the beatmap's picture would only be noise
    "BlurLevel": "0",
    "LightenDuringBreaks": "False",
    "ShowStoryboard": "False",
    "HitLighting": "False",
    "StarFountains": "False",
    "FadePlayfieldWhenHealthLow": "False",
    "BeatmapSkins": "False",  # every map drawn with the same skin
    "BeatmapColours": "False",
    "CursorRotation": "False",
    "GameplayLeaderboard": "False",
    "ReplaySettingsOverlay": "False",
    "ReplayPlaybackControlsExpanded": "False",
    "AutomaticallyDownloadMissingBeatmaps": "False",
    "DiscordRichPresence": "Off",
    "MenuVoice": "False",
    "MenuTips": "False",
    "ConfineMouseMode": "Never",
    "HUDVisibilityMode": "Never",  # score/HP widgets would sit inside the captured frame
    "KeyOverlay": "False",
    "ShowFpsDisplay": "False",
    "MenuParallaxScale": "0",
    "SeasonalBackgroundMode": "Never",
    "UIHoldActivationDelay": "0",  # quick retry / exit act immediately
    "AudioOffset": "0",
    "AutomaticallyAdjustBeatmapOffset": "False",
    "Scaling": "Off",
    "ComboColourNormalisationAmount": "0",
    "GameplayCursorSize": "1",
    "AutoCursorSize": "False",
    "Ruleset": "osu",
}

# "High precision mouse" (relative mode) is effectively on for this build on macOS, despite the
# framework meaning to force it off, and synthetic absolute moves would then drift. This file
# turns it off for real (osu-framework InputConfigManager).
INPUT_JSON = '{"InputHandlers":[{"$type":"osu.Framework.Input.Handlers.Mouse.MouseHandler, osu.Framework","UseRelativeMode":false}]}'


def running() -> bool:
    return subprocess.run(["pgrep", "-f", "osu!.app/Contents/MacOS"], capture_output=True).returncode == 0


def patch(path: Path, values: dict[str, str]) -> None:
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy(path, backup)
    lines = path.read_text().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in values:
            lines[i] = f"{key} = {values[key]}"
            seen.add(key)
    lines += [f"{k} = {v}" for k, v in values.items() if k not in seen]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--skin", help="skin GUID to select (list them: dotnet run --project tools/lazer_realm -c Release)")
    parser.add_argument("--dim", type=float, default=1.0,
                        help="background dim, 0-1 (1: black; lower shows the map background, which needs a model "
                             "trained with backgrounds)")
    args = parser.parse_args()
    if args.skin:
        GAME["Skin"] = args.skin
    GAME["DimLevel"] = f"{args.dim:g}"
    if running():
        raise SystemExit("quit osu! first: it writes its config back when it exits")
    for name, values in (("framework.ini", FRAMEWORK), ("game.ini", GAME)):
        path = DATA / name
        if args.restore:
            shutil.copy(path.with_suffix(".ini.orig"), path)
        else:
            patch(path, values)
        print(f"{'restored' if args.restore else 'configured'} {path}")
    input_json = DATA / "input.json"
    if args.restore:
        input_json.unlink(missing_ok=True)
    else:
        input_json.write_text(INPUT_JSON)
        print(f"configured {input_json}")


if __name__ == "__main__":
    main()
