"""Random action delay (``manager_env.config.action_delay=[min, max]``, in physics steps).

On the robot the joints follow an action some milliseconds after the observation it was computed from, and a policy
trained without that delay has little margin for it: one control step (20 ms) late costs the released checkpoint
about 5 points of eval success rate, two steps fail nearly every motion. With this switch the joints apply the
action the policy chose ``d`` physics steps earlier (5 ms each, ``decimation`` per control step), ``d`` drawn per env
from ``[min, max]`` at every episode start; the policy still observes its own latest action, as on the robot.
``[4, 4]`` is a fixed one-control-step delay, e.g. for an eval.

The history, the draws and the episode-start resets run on the device under masks (an episode starts where
``episode_length_buf`` is 0), so the switch has no host syncs and works with ``sync_free_step`` and the
post-processing graph. With ``newton.graph_decimation`` the action is applied once per control step, so the delays
must then be whole control steps.
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
    for term in env.action_manager._terms.values():  # noqa: SLF001
        _delay_term(env, term, low, high, decimation)


def _delay_term(env, term, low, high, decimation):
    num_envs, dim = term.processed_actions.shape
    device = term.processed_actions.device
    # history[k]: the processed action of k control steps ago
    history = torch.zeros((high + decimation - 1) // decimation + 1, num_envs, dim, device=device)
    delay = torch.zeros(num_envs, dtype=torch.long, device=device)
    rows = torch.arange(num_envs, device=device)
    substep = [0]
    process_actions, apply_actions = term.process_actions, term.apply_actions

    def delayed_process_actions(actions):
        process_actions(actions)
        current = term.processed_actions
        start = env.episode_length_buf == 0
        delay.copy_(torch.where(start, torch.randint(low, high + 1, (num_envs,), device=device), delay))
        history.copy_(history.roll(1, 0))
        history[0] = current
        # an episode's first steps fall back to its first action, not the previous episode's
        history.copy_(torch.where(start[None, :, None], current, history))
        substep[0] = 0

    def delayed_apply_actions():
        # at physics step s of the control step, the newest action issued at least `delay` physics steps ago
        lag = torch.div(delay - substep[0] + decimation - 1, decimation, rounding_mode="floor").clamp_(min=0)
        current = term._processed_actions  # noqa: SLF001
        term._processed_actions = history[lag, rows]  # noqa: SLF001
        try:
            apply_actions()
        finally:
            term._processed_actions = current  # noqa: SLF001
        substep[0] += 1

    term.process_actions = delayed_process_actions
    term.apply_actions = delayed_apply_actions
