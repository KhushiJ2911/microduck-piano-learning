#!/usr/bin/env python3
"""Render an honest two-duck lateral traversal of a continuous floor piano.

The VelStand-derived actor receives only the official policy's documented
lateral velocity command. Every key is a passive spring-loaded MuJoCo articulation.
Acceptance requires displacement, alternating one-foot support, both feet
landing on different keys, physical note on/off edges, and no fall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


CONTROL_HZ = 50
KEY_COUNT = 16
KEY_PITCH_M = 0.084
KEY_HALF_WIDTH_M = 0.041
DUCKS = ("duck0", "duck1")
SIDES = ("left", "right")
NOTES = (
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4",
    "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5",
)
PIANO_XML = Path(__file__).resolve().parents[1] / "assets" / "piano_keys.xml"
OFFICIAL_VELSTAND_SHA256 = (
    "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_y(index: int) -> float:
    return (index - (KEY_COUNT - 1) / 2.0) * KEY_PITCH_M


def two_key_bank_spec(bank: int, stiffness: float, damping: float):
    """Return two adjacent long floor keys in one articulated entity."""
    import mujoco

    if not PIANO_XML.is_file():
        raise FileNotFoundError(f"missing piano model: {PIANO_XML}")
    spec = mujoco.MjSpec.from_file(str(PIANO_XML))
    root = spec.body("piano_keys")
    root.pos[:] = (0.0, 0.0, 0.0)
    for local, side in enumerate(SIDES):
        index = 2 * bank + local
        body = spec.body(f"piano_key_{side}_body")
        body.pos[:] = (0.015, key_y(index), 0.010)
        body.gravcomp = 1.0
        geom = spec.geom(f"piano_key_{side}")
        geom.size[:] = (0.145, KEY_HALF_WIDTH_M, 0.002)
        geom.mass = 0.040
        geom.rgba[:] = (
            (0.12, 0.14, 0.18, 1.0)
            if index in (1, 3, 6, 8, 10, 13, 15)
            else (0.96, 0.94, 0.84, 1.0)
        )
        joint = spec.joint(f"piano_key_{side}_slide")
        # These are explicit experiment parameters: stiffness must keep an
        # even two-foot stance below note-on while permitting a one-foot impact;
        # damping suppresses keycap rebound after that impact.
        joint.stiffness[0] = stiffness
        joint.damping[0] = damping
    return spec


def two_foot_found(sensor, num_envs: int) -> torch.Tensor:
    found = torch.nan_to_num(sensor.data.found.float(), nan=0.0)
    return found.reshape(num_envs, 2, -1).amax(-1) > 0.5


def duet_last_action(env, action_start: int) -> torch.Tensor:
    """Return one duck's 14 previous actions from the combined 28-D action."""
    return env.action_manager.action[:, action_start : action_start + 14]


def remap_actor_group(group, robot_name: str, action_start: int):
    """Bind one official VelStand observation group to a named duet robot."""
    from mjlab.managers import ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    group = deepcopy(group)
    for term in group.terms.values():
        for value in term.params.values():
            if isinstance(value, SceneEntityCfg) and value.name == "robot":
                value.name = robot_name
    for name in ("base_ang_vel", "projected_gravity"):
        group.terms[name].params["asset_cfg"] = SceneEntityCfg(robot_name)
    group.terms["actions"] = ObservationTermCfg(
        func=duet_last_action,
        params={"action_start": action_start},
    )
    return group


def build_cfg(
    num_envs: int,
    seed: int,
    width: int,
    height: int,
    key_stiffness: float,
    key_damping: float,
):
    from mjlab.entity import EntityCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_STANDUP_ROBOT_CFG,
    )
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    # Begin from the official 61-D locomotion interface, then duplicate its
    # action/observation contracts around the same ground-contact robot model
    # used in the audited piano rollout. This keeps the repository evaluator
    # independent of the exploratory piano training modules used during R&D.
    cfg = make_microduck_velocity_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.scene.env_spacing = 0.0

    robot_template = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    cfg.scene.entities = {
        "duck0": deepcopy(robot_template),
        "duck1": deepcopy(robot_template),
    }
    for bank in range(KEY_COUNT // 2):
        cfg.scene.entities[f"bank{bank:02d}"] = EntityCfg(
            spec_fn=partial(two_key_bank_spec, bank, key_stiffness, key_damping)
        )

    cfg.commands["twist"].entity_name = "duck0"
    base_action = cfg.actions["joint_pos"]
    if not isinstance(base_action, JointPositionActionCfg):
        raise TypeError("official VelStand action contract changed")
    cfg.actions = {
        "duck0_joint_pos": deepcopy(base_action),
        "duck1_joint_pos": deepcopy(base_action),
    }
    cfg.actions["duck0_joint_pos"].entity_name = "duck0"
    cfg.actions["duck1_joint_pos"].entity_name = "duck1"

    original_actor = cfg.observations["actor"]
    cfg.observations = {
        "duck0_actor": remap_actor_group(original_actor, "duck0", 0),
        "duck1_actor": remap_actor_group(original_actor, "duck1", 14),
    }

    reset_base = deepcopy(cfg.events["reset_base"])
    reset_joints = deepcopy(cfg.events["reset_robot_joints"])
    cfg.events = {
        "expand_bam_friction_fields": cfg.events["expand_bam_friction_fields"],
    }
    for duck_index in range(2):
        base = deepcopy(reset_base)
        base.params["asset_cfg"] = SceneEntityCfg(f"duck{duck_index}")
        base.params["pose_range"]["x"] = (0.0, 0.0)
        base.params["pose_range"]["y"] = ((-0.168, 0.168)[duck_index],) * 2
        base.params["pose_range"]["yaw"] = (0.0, 0.0)
        cfg.events[f"reset_duck{duck_index}_base"] = base
        joints = deepcopy(reset_joints)
        joints.params["asset_cfg"] = SceneEntityCfg(
            f"duck{duck_index}", joint_names=(".*",)
        )
        joints.params["position_range"] = (0.0, 0.0)
        joints.params["velocity_range"] = (0.0, 0.0)
        cfg.events[f"reset_duck{duck_index}_joints"] = joints

    cfg.rewards.clear()
    cfg.metrics.clear()
    cfg.curriculum.clear()
    cfg.terminations.clear()

    # The native stance is approximately +/-42 mm about each root.  These
    # centers put all four initial feet near key centers instead of seams.
    sensors = []
    for duck_index, duck in enumerate(DUCKS):
        sensors.append(
            ContactSensorCfg(
                name=f"duck{duck_index}_ground",
                primary=ContactMatch(
                    mode="geom",
                    pattern=r"^(left_foot_collision|right_foot_collision)$",
                    entity=duck,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found",),
                reduce="none",
                num_slots=1,
            )
        )
        for key in range(KEY_COUNT):
            bank, side = divmod(key, 2)
            sensors.append(
                ContactSensorCfg(
                    name=f"duck{duck_index}_key{key:02d}",
                    primary=ContactMatch(
                        mode="geom",
                        pattern=r"^(left_foot_collision|right_foot_collision)$",
                        entity=duck,
                    ),
                    secondary=ContactMatch(
                        mode="geom",
                        pattern=rf"^piano_key_{SIDES[side]}$",
                        entity=f"bank{bank:02d}",
                    ),
                    fields=("found",),
                    reduce="none",
                    num_slots=1,
                )
            )
    cfg.scene.sensors = tuple(sensors)
    cfg.sim.nconmax = max(cfg.sim.nconmax, 512)

    cfg.viewer.width, cfg.viewer.height = width, height
    cfg.viewer.env_idx = 0
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.origin_type = type(cfg.viewer.origin_type).WORLD
    cfg.viewer.lookat = (0.025, 0.0, 0.075)
    cfg.viewer.distance = 1.62
    cfg.viewer.azimuth = 180.0
    cfg.viewer.elevation = -10.0
    return cfg


def command(step: int) -> tuple[tuple[float, float, float], str]:
    """Out-and-back lateral walk; both ducks retain their separation."""
    t = step / CONTROL_HZ
    if t < 1.0:
        return (0.0, 0.0, 0.0), "SETTLE"
    if t < 3.2:
        return (0.0, 0.22, 0.0), "WALK RIGHT"
    if t < 3.7:
        return (0.0, 0.0, 0.0), "BALANCE"
    if t < 7.9:
        return (0.0, -0.22, 0.0), "WALK LEFT"
    if t < 8.4:
        return (0.0, 0.0, 0.0), "BALANCE"
    if t < 10.6:
        return (0.0, 0.22, 0.0), "RETURN"
    return (0.0, 0.0, 0.0), "FINAL STAND"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--key-stiffness", type=float, default=1200.0)
    parser.add_argument("--key-damping", type=float, default=10.0)
    parser.add_argument("--key-on-mm", type=float, default=4.5)
    parser.add_argument("--key-off-mm", type=float, default=0.5)
    args = parser.parse_args()

    policy_path = args.policy.expanduser().resolve(strict=True)
    official_path = policy_path.with_name("velstand.onnx")
    if not official_path.is_file() or sha256(official_path) != OFFICIAL_VELSTAND_SHA256:
        raise RuntimeError("pinned official VelStand source is missing or changed")

    import mujoco
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cfg = build_cfg(
        args.num_envs,
        args.seed,
        args.width,
        args.height,
        args.key_stiffness,
        args.key_damping,
    )
    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=device,
        # Construct the renderer only after rollout selection.  Initializing
        # OpenGL before reset perturbs RNG consumption in the batched simulator,
        # making a rendered audit differ from the headless evaluation it is
        # supposed to reproduce.
        render_mode=None,
    )
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if session.get_inputs()[0].shape[-1] != 61 or session.get_outputs()[0].shape[-1] != 14:
        raise RuntimeError("unexpected official policy contract")

    n = args.num_envs
    horizon = round(args.seconds * CONTROL_HZ)
    note = torch.zeros((n, KEY_COUNT), dtype=torch.bool, device=device)
    on_count = torch.zeros((n, KEY_COUNT), dtype=torch.long, device=device)
    off_count = torch.zeros_like(on_count)
    walk_on_count = torch.zeros_like(on_count)
    walk_off_count = torch.zeros_like(on_count)
    seam_chord_frames = torch.zeros(n, dtype=torch.long, device=device)
    ever_key = torch.zeros((2, n, 2, KEY_COUNT), dtype=torch.bool, device=device)
    initial_key = torch.zeros_like(ever_key)
    new_key_landing = torch.zeros_like(ever_key)
    previous_support = torch.zeros((2, n, 2), dtype=torch.bool, device=device)
    touchdown_count = torch.zeros((2, n, 2), dtype=torch.long, device=device)
    left_only = torch.zeros((2, n), dtype=torch.long, device=device)
    right_only = torch.zeros_like(left_only)
    foot_air = torch.zeros((2, n, 2), dtype=torch.long, device=device)
    fell = torch.zeros((2, n), dtype=torch.bool, device=device)
    min_y = torch.full((2, n), float("inf"), device=device)
    max_y = torch.full((2, n), -float("inf"), device=device)
    min_root_z = torch.full((2, n), float("inf"), device=device)
    max_tilt = torch.zeros((2, n), device=device)
    events: list[dict] = []
    states: list[tuple[np.ndarray, np.ndarray, int, str, np.ndarray, np.ndarray]] = []

    try:
        env.reset(seed=args.seed)
        action_terms = [env.action_manager.get_term(f"duck{i}_joint_pos") for i in range(2)]
        target_names = [tuple(getattr(term, "_target_names")) for term in action_terms]
        metadata_names = tuple(session.get_modelmeta().custom_metadata_map["joint_names"].split(","))
        if target_names[0] != target_names[1] or target_names[0] != metadata_names:
            raise RuntimeError("policy/action joint order mismatch")

        for step in range(horizon):
            cmd, phase = command(step)
            actions = []
            for duck_index in range(2):
                obs = env.observation_manager.compute_group(f"duck{duck_index}_actor").clone()
                obs[:, 48:51] = torch.tensor(cmd, device=device)
                obs[:, 51:61] = 0.0
                out = session.run(
                    [output_name],
                    {input_name: obs.detach().cpu().numpy().astype(np.float32)},
                )[0]
                actions.append(torch.as_tensor(out, device=device, dtype=torch.float32))
            env.step(torch.cat(actions, dim=1).clamp(-1.0, 1.0))

            travel = torch.stack(
                [
                    env.scene[f"bank{bank:02d}"].data.joint_pos[:, side]
                    for bank in range(KEY_COUNT // 2)
                    for side in range(2)
                ],
                dim=1,
            ).clamp_min(0.0)
            contacts = torch.zeros((2, n, 2, KEY_COUNT), dtype=torch.bool, device=device)
            support = torch.zeros((2, n, 2), dtype=torch.bool, device=device)
            for duck_index, duck in enumerate(DUCKS):
                ground = two_foot_found(env.scene.sensors[f"duck{duck_index}_ground"], n)
                for key in range(KEY_COUNT):
                    contacts[duck_index, :, :, key] = two_foot_found(
                        env.scene.sensors[f"duck{duck_index}_key{key:02d}"], n
                    )
                support[duck_index] = ground | contacts[duck_index].any(2)
                ever_key[duck_index] |= contacts[duck_index]
                if step == CONTROL_HZ - 1:
                    initial_key[duck_index] = contacts[duck_index]
                if step >= CONTROL_HZ:
                    touchdown = (~previous_support[duck_index]) & support[duck_index]
                    touchdown_count[duck_index] += touchdown.long()
                    new_key_landing[duck_index] |= (
                        contacts[duck_index]
                        & touchdown[:, :, None]
                        & (~initial_key[duck_index])
                    )
                previous_support[duck_index] = support[duck_index]
                left_only[duck_index] += (support[duck_index, :, 0] & ~support[duck_index, :, 1]).long()
                right_only[duck_index] += (support[duck_index, :, 1] & ~support[duck_index, :, 0]).long()
                foot_air[duck_index] += (~support[duck_index]).long()

                root = env.scene[duck].data.root_link_pos_w
                min_y[duck_index] = torch.minimum(min_y[duck_index], root[:, 1])
                max_y[duck_index] = torch.maximum(max_y[duck_index], root[:, 1])
                min_root_z[duck_index] = torch.minimum(min_root_z[duck_index], root[:, 2])
                quat = env.scene[duck].data.root_link_quat_w
                up_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
                tilt = torch.rad2deg(torch.acos(up_z.clamp(-1.0, 1.0)))
                max_tilt[duck_index] = torch.maximum(max_tilt[duck_index], tilt)
                physical_fall = (~torch.isfinite(root[:, 2])) | (~torch.isfinite(tilt))
                physical_fall |= (root[:, 2] < 0.055) | (tilt > 70.0)
                fell[duck_index] |= physical_fall

            any_contact = contacts.any(0).any(1)
            new_note = torch.where(
                note,
                travel > args.key_off_mm / 1000.0,
                travel >= args.key_on_mm / 1000.0,
            )
            on_edge = (~note) & new_note & any_contact
            off_edge = note & (~new_note)
            on_count += on_edge.long()
            off_count += off_edge.long()
            if step >= CONTROL_HZ:
                walk_on_count += on_edge.long()
                walk_off_count += off_edge.long()
                # A foot bridging two neighboring keys is an accidental chord,
                # not a precise piano step.  Reject any performance frame in
                # which one physical foot activates more than one key.
                notes_by_foot = (contacts & on_edge[None, :, None, :]).sum(3)
                seam_chord_frames += (notes_by_foot > 1).any(dim=(0, 2)).long()
            for world, key in on_edge.nonzero(as_tuple=False).tolist():
                touching = contacts[:, world, :, key].nonzero(as_tuple=False).tolist()
                events.append(
                    {
                        "step": step,
                        "time_s": step / CONTROL_HZ,
                        "world": world,
                        "event": "note_on",
                        "key": key,
                        "note": NOTES[key],
                        "travel_mm": float(1000.0 * travel[world, key]),
                        "touching_feet": [f"duck{d}_{SIDES[foot]}" for d, foot in touching],
                    }
                )
            for world, key in off_edge.nonzero(as_tuple=False).tolist():
                events.append(
                    {
                        "step": step,
                        "time_s": step / CONTROL_HZ,
                        "world": world,
                        "event": "note_off",
                        "key": key,
                        "note": NOTES[key],
                        "travel_mm": float(1000.0 * travel[world, key]),
                    }
                )
            note = new_note

            if args.gif is not None and step % args.frame_stride == 0:
                states.append(
                    (
                        env.sim.data.qpos.detach().cpu().numpy().copy(),
                        env.sim.data.qvel.detach().cpu().numpy().copy(),
                        step,
                        phase,
                        support.detach().cpu().numpy().copy(),
                        travel.detach().cpu().numpy().copy(),
                    )
                )

        distinct_by_foot = ever_key.sum(3)
        new_landings_by_foot = new_key_landing.sum(3)
        y_range = max_y - min_y
        completed_keys = (walk_on_count > 0) & (walk_off_count > 0)
        walk_note_ons = walk_on_count.sum(1)
        total_touchdowns = touchdown_count.sum(dim=(0, 2))
        no_chatter = (walk_note_ons >= 4) & (walk_note_ons <= total_touchdowns)
        no_seam_chords = seam_chord_frames == 0
        movement_core = (
            (~fell).all(0)
            & (left_only >= 5).all(0)
            & (right_only >= 5).all(0)
            & (foot_air >= 5).all(dim=(0, 2))
            & (new_landings_by_foot >= 1).all(dim=(0, 2))
            & (y_range >= 0.08).all(0)
            & (completed_keys.sum(1) >= 4)
        )
        movement_gate = movement_core & no_chatter & no_seam_chords
        candidates = movement_gate.nonzero(as_tuple=False).flatten()
        score = (
            2.0 * new_landings_by_foot.float().sum((0, 2))
            + distinct_by_foot.float().sum((0, 2))
            + 20.0 * y_range.clamp_max(0.20).sum(0)
            + 0.1 * completed_keys.sum(1)
            - 100.0 * fell.any(0).float()
        )
        selected = int(candidates[0]) if candidates.numel() else int(score.argmax())

        all_selected_events = [event for event in events if event["world"] == selected]
        selected_events = [event for event in all_selected_events if event["step"] >= CONTROL_HZ]
        if args.gif is not None:
            from mjlab.viewer.offscreen_renderer import OffscreenRenderer

            renderer = OffscreenRenderer(
                model=env.sim.mj_model,
                cfg=cfg.viewer,
                scene=env.scene,
            )
            renderer.initialize()
            frames = []
            recent: list[str] = []
            event_cursor = 0
            for qpos, qvel, step, phase, support, travel in states:
                while event_cursor < len(selected_events) and selected_events[event_cursor]["step"] <= step:
                    event = selected_events[event_cursor]
                    if event["event"] == "note_on":
                        recent = (recent + [event["note"]])[-8:]
                    event_cursor += 1
                renderer._data.qpos[:] = qpos[selected]
                renderer._data.qvel[:] = qvel[selected]
                mujoco.mj_forward(renderer._model, renderer._data)
                renderer.renderer.update_scene(renderer._data, camera=renderer._cam)
                frame = Image.fromarray(np.asarray(renderer.render()).copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((8, 8, args.width - 8, 92), fill=(0, 0, 0))
                status = "PASS" if bool(movement_gate[selected]) else "DIAGNOSTIC / REJECTED"
                draw.text((18, 14), f"CONTINUOUS PHYSICAL KEYBOARD | {status} | t={step / CONTROL_HZ:4.1f}s", fill="white")
                draw.text((18, 34), f"VelStand-derived actor command: {phase}", fill=(165, 225, 255))
                for duck_index in range(2):
                    feet = support[duck_index, selected]
                    label = "LEFT" if feet[0] and not feet[1] else "RIGHT" if feet[1] and not feet[0] else "BOTH" if feet.all() else "AIR"
                    draw.text((18 + 320 * duck_index, 54), f"duck{duck_index}: support={label}", fill=(165, 225, 255))
                draw.text((18, 74), f"physical notes: {' '.join(recent) if recent else 'waiting'}", fill=(255, 220, 100))
                frames.append(frame)
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                args.gif,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=round(1000 * args.frame_stride / CONTROL_HZ),
                optimize=False,
            )
            renderer.close()

        report = {
            "claim_scope": "two VelStand-derived actors laterally traverse a continuous passive floor piano",
            "accepted": bool(movement_gate[selected]),
            "strict_successes": int(movement_gate.sum()),
            "movement_core_successes": int(movement_core.sum()),
            "num_worlds": n,
            "seed": args.seed,
            "selected_world": selected,
            "official_velstand_sha256": sha256(official_path),
            "evaluated_policy_sha256": sha256(policy_path),
            "key_count": KEY_COUNT,
            "key_pitch_m": KEY_PITCH_M,
            "key_stiffness_n_per_m": args.key_stiffness,
            "key_damping_n_s_per_m": args.key_damping,
            "key_on_mm": args.key_on_mm,
            "key_off_mm": args.key_off_mm,
            "selected_events": selected_events,
            "selected_settle_events": [
                event for event in all_selected_events if event["step"] < CONTROL_HZ
            ],
            "selected_completed_key_count": int(completed_keys[selected].sum()),
            "selected_walk_note_ons": int(walk_note_ons[selected]),
            "selected_foot_touchdowns": int(total_touchdowns[selected]),
            "selected_seam_chord_frames": int(seam_chord_frames[selected]),
            "population": {
                "falls": int(fell.any(0).sum()),
                "movement_core_successes": int(movement_core.sum()),
                "strict_no_chatter_successes": int(movement_gate.sum()),
                "mean_note_ons_per_touchdown_on_movement_core": (
                    float(
                        (
                            walk_note_ons[movement_core].float()
                            / total_touchdowns[movement_core].clamp_min(1).float()
                        ).mean()
                    )
                    if bool(movement_core.any())
                    else None
                ),
                "mean_lateral_range_m_by_duck": y_range.mean(1).tolist(),
            },
            "per_duck": {
                str(duck): {
                    "fell": bool(fell[duck, selected]),
                    "lateral_range_m": float(y_range[duck, selected]),
                    "left_only_support_frames": int(left_only[duck, selected]),
                    "right_only_support_frames": int(right_only[duck, selected]),
                    "left_foot_air_frames": int(foot_air[duck, selected, 0]),
                    "right_foot_air_frames": int(foot_air[duck, selected, 1]),
                    "left_foot_distinct_keys": int(distinct_by_foot[duck, selected, 0]),
                    "right_foot_distinct_keys": int(distinct_by_foot[duck, selected, 1]),
                    "left_foot_new_key_landings": int(new_landings_by_foot[duck, selected, 0]),
                    "right_foot_new_key_landings": int(new_landings_by_foot[duck, selected, 1]),
                    "left_foot_touchdowns": int(touchdown_count[duck, selected, 0]),
                    "right_foot_touchdowns": int(touchdown_count[duck, selected, 1]),
                    "min_root_z_m": float(min_root_z[duck, selected]),
                    "max_tilt_deg": float(max_tilt[duck, selected]),
                }
                for duck in range(2)
            },
            "acceptance": {
                "no_falls": not bool(fell[:, selected].any()),
                "both_ducks_move_at_least_8cm": bool((y_range[:, selected] >= 0.08).all()),
                "both_ducks_use_left_and_right_single_support": bool(
                    (left_only[:, selected] >= 5).all() and (right_only[:, selected] >= 5).all()
                ),
                "every_foot_has_airtime": bool((foot_air[:, selected] >= 5).all()),
                "every_foot_lands_on_a_new_key_after_airtime": bool(
                    (new_landings_by_foot[:, selected] >= 1).all()
                ),
                "four_physical_press_release_keys": bool(completed_keys[selected].sum() >= 4),
                "no_threshold_chatter": bool(no_chatter[selected]),
                "no_foot_spans_two_keys": bool(no_seam_chords[selected]),
            },
            "post_reset_root_writes": False,
            "post_reset_joint_or_foot_writes": False,
            "post_reset_key_writes": False,
            "external_forces": False,
            "welds_mocap_or_harness": False,
            "joint_trajectory_scripting": False,
            "gif": str(args.gif) if args.gif is not None else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "selected_events"}, indent=2))
        return 0 if report["accepted"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
