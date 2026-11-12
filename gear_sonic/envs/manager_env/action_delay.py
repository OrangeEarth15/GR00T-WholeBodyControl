"""Random action delay (``manager_env.config.action_delay=[min, max]``, in physics steps).

On the robot the joints follow an action some milliseconds after the observation it was computed from, and a policy
trained without that delay has little margin for it: one control step (20 ms) late costs the released checkpoint
about 5 points of eval success rate, two steps fail nearly every motion. With this switch the joints apply the
action the policy chose ``d`` physics steps earlier (5 ms each, ``decimation`` per control step), ``d`` drawn per
env uniformly from ``[min, max]`` at every episode start; the policy still observes its own latest action, as on
the robot. ``[4, 4]`` is a fixed one-control-step delay, e.g. for an eval; a fixed delay draws no random numbers.

The history, the draws and the episode-start resets run on the device under masks (an episode starts where
``episode_length_buf`` is 0), so the switch has no host syncs and works with ``sync_free_step`` and the
post-processing graph. With ``newton.graph_decimation`` the action is applied once per control step, so the delays
must then be whole control steps, and they are drawn as whole control steps.
"""

import torch


def enable(env, low: int, high: int):
    """Delay every action term of ``env`` by ``low`` to ``high`` physics steps."""
    decimation = env.cfg.decimation
    once_per_control_step = env._physics_handles_decimation  # noqa: SLF001
    if not 0 <= low <= high:
        raise ValueError(f"action_delay needs 0 <= min <= max, got [{low}, {high}]")
    if once_per_control_step and (low % decimation or high % decimation):
        raise ValueError(
            f"action_delay [{low}, {high}]: with newton.graph_decimation the action is applied once per control "
            f"step, so the delays must be multiples of {decimation}; or turn graph_decimation off"
        )
    unit = decimation if once_per_control_step else 1
    for term in env.action_manager._terms.values():  # noqa: SLF001
        _delay_term(env, term, low // unit, high // unit, unit)


def _delay_term(env, term, low, high, unit):
    """Delay ``term`` by ``unit`` times a draw from ``[low, high]`` physics steps."""
    decimation = env.cfg.decimation
    num_envs, dim = term.processed_actions.shape
    device = term.processed_actions.device
    # a ring of processed actions, one per control step: the newest at step % slots
    slots = (high * unit + decimation - 1) // decimation + 1
    history = torch.zeros((slots, num_envs, dim), device=device)
    step = torch.zeros((), dtype=torch.long, device=device)
    delay = torch.full((num_envs,), low * unit, dtype=torch.long, device=device)
    rows = torch.arange(num_envs, device=device)
    substep = 0
    process_actions, apply_actions = term.process_actions, term.apply_actions

    def delayed_process_actions(actions):
        nonlocal substep
        process_actions(actions)
        current = term.processed_actions
        start = env.episode_length_buf == 0
        if low < high:
            draw = torch.randint(low, high + 1, (num_envs,), device=device) * unit
            torch.where(start, draw, delay, out=delay)
        step.add_(1)
        history.index_copy_(0, (step % slots).view(1), current[None])
        # an episode's first steps fall back to its first action, not the previous episode's
        torch.where(start[None, :, None], current, history, out=history)
        substep = 0

    def delayed_apply_actions():
        nonlocal substep
        # at physics step `substep` of the control step: the newest action issued `delay` or more physics steps ago
        lag = torch.div(delay - substep + decimation - 1, decimation, rounding_mode="floor").clamp_(min=0)
        current = term._processed_actions  # noqa: SLF001
        term._processed_actions = history[(step - lag) % slots, rows]  # noqa: SLF001
        apply_actions()
        term._processed_actions = current  # noqa: SLF001
        substep += 1

    term.process_actions = delayed_process_actions
    term.apply_actions = delayed_apply_actions
