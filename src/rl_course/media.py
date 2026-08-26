from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from racetrackgym.graphic import create_map

from rl_course.racetrack import make_env


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def annotate(frame: np.ndarray | Image.Image, text: str) -> Image.Image:
    image = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(20, 27, 34))
    draw.text((12, 10), text, fill=(255, 255, 255))
    return image


def write_video(frames: Iterable[Image.Image], output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to export videos")
    iterator = iter(frames)
    first = next(iterator)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = first.size
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in chain([first], iterator):
            if frame.size != (width, height):
                raise ValueError("all video frames must have the same dimensions")
            process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def save_poster(frame: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(path, "WEBP", quality=82, method=6)


def mountain_car_frames(capture: dict, stride: int = 4) -> Iterable[Image.Image]:
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    state, _ = env.reset(seed=capture["seed"])
    try:
        for step, expected in enumerate(capture["frames"]):
            if step:
                state, _, _, _, _ = env.step(
                    np.array([expected["action"]], dtype=np.float32)
                )
                if not np.allclose(
                    state,
                    [expected["position"], expected["velocity"]],
                    atol=1e-6,
                ):
                    raise RuntimeError(f"MountainCar replay diverged at step {step}")
            if step % stride == 0:
                yield annotate(
                    env.render(),
                    f"Random policy · step {step} · position {state[0]:.3f}",
                )
    finally:
        env.close()


def frozen_lake_frames(result: dict, seed: int, failure: bool) -> Iterable[Image.Image]:
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    captured = [(0, state, 0.0, env.render())]
    try:
        for step in range(1, 101):
            state, reward, terminated, truncated, _ = env.step(result["policy"][state])
            if not failure or step <= 12 or truncated:
                captured.append((step, state, reward, env.render()))
            if terminated or truncated:
                if failure and not truncated:
                    raise RuntimeError("expected the failure policy to time out")
                if not failure and reward != 1:
                    raise RuntimeError("expected the success policy to reach the goal")
                break
    finally:
        env.close()

    for step, state, reward, frame in captured:
        if reward:
            status = "goal reached"
        elif failure and step == 100:
            status = "timeout · policy stayed in the same tile"
        else:
            status = f"state {state}"
        image = annotate(frame, f"{'Premature-stop' if failure else 'Full'} policy · step {step} · {status}")
        repeats = 12 if reward or step == 100 else 3
        yield from (image,) * repeats


def racetrack_frames(capture: dict, outcome: str) -> Iterable[Image.Image]:
    expected = capture["trajectories"][outcome]
    env = make_env()
    env.reset(seed=capture["seed"])
    base = env.unwrapped.racetrack_env
    try:
        for index, frame in enumerate(expected["frames"]):
            base.position = np.array(frame["position"])
            base.velocity = np.array(frame["velocity"])
            base.path = [
                np.array(item["position"]) for item in expected["frames"][: index + 1]
            ]
            image = create_map(base, show_path=True).convert("RGB").resize((888, 336))
            draw = ImageDraw.Draw(image)
            row, column = frame["position"]
            velocity_row, velocity_column = frame["velocity"]
            center = (int((column + 0.5) * 24), int((row + 0.5) * 24))
            draw.ellipse((center[0] - 9, center[1] - 9, center[0] + 9, center[1] + 9), fill="#0072b2", outline="white", width=2)
            draw.line((center[0], center[1], center[0] + velocity_column * 7, center[1] + velocity_row * 7), fill="white", width=3)
            status = " · finish" if frame["reward"] > 0 else " · crash" if frame["reward"] < 0 else ""
            rendered = annotate(image, f"Greedy DQN · {outcome} · step {index} · velocity ({velocity_row}, {velocity_column}){status}")
            yield from (rendered,) * 3
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight videos for the RL article.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/media"))
    parser.add_argument("--capture-dir", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    mountain = load_json(args.capture_dir / "mountain-car.json")
    mountain_frames = mountain_car_frames(mountain)
    first_mountain = next(mountain_frames)
    write_video(chain([first_mountain], mountain_frames), args.output_dir / "mountain-car.mp4", fps=20)
    save_poster(first_mountain, args.output_dir / "mountain-car.webp")

    frozen_success = load_json(args.runs_dir / "rl-v1/frozen-lake/full/seed-21.json")
    success_frames = frozen_lake_frames(frozen_success, seed=22, failure=False)
    first_success = next(success_frames)
    write_video(chain([first_success], success_frames), args.output_dir / "frozen-lake-success.mp4", fps=6)
    save_poster(first_success, args.output_dir / "frozen-lake-success.webp")

    frozen_failure = load_json(args.runs_dir / "rl-v1/frozen-lake/naive/seed-87.json")
    failure_frames = frozen_lake_frames(frozen_failure, seed=88, failure=True)
    first_failure = next(failure_frames)
    write_video(chain([first_failure], failure_frames), args.output_dir / "frozen-lake-failure.mp4", fps=6)
    save_poster(first_failure, args.output_dir / "frozen-lake-failure.webp")

    racetrack = load_json(args.runs_dir / "rl-artifacts/racetrack/dqn/seed-87/trajectories.json")
    track_frames = racetrack_frames(racetrack, "success")
    first_track = next(track_frames)
    write_video(chain([first_track], track_frames), args.output_dir / "racetrack-success.mp4", fps=6)
    save_poster(first_track, args.output_dir / "racetrack-success.webp")

    crash_frames = racetrack_frames(racetrack, "crash")
    first_crash = next(crash_frames)
    write_video(chain([first_crash], crash_frames), args.output_dir / "racetrack-failure.mp4", fps=6)
    save_poster(first_crash, args.output_dir / "racetrack-failure.webp")

    print(json.dumps({"output_dir": str(args.output_dir), "videos": sorted(path.name for path in args.output_dir.glob("*.mp4"))}))


if __name__ == "__main__":
    main()
