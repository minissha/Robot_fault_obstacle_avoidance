"""
anti_stall.py

The escape behaviour every controller falls back on when it runs out of room.

Purely reactive controllers deadlock. Jammed into a corner, the robot rotates
a few degrees towards whichever side looks better, which makes the other side
look better, so it rotates back -- and it sits there flipping between full
left and full right without ever moving. Measured on failed runs it was
doing this forty times in sixty steps with a mean speed of zero. Once
collisions were dealt with, this was the *only* remaining failure mode.

The fix is to decide once. Pick a direction when the way ahead first closes,
hold it until the way ahead opens, and let the robot rotate continuously
until it is facing the gap. Deciding late costs nothing; deciding twice
costs everything.

This lives here rather than inside a controller because all four of them use
it, on the same trigger, with the same parameters. What is being compared is
how they drive when they have room -- not who has the better panic button.

(This module used to hold a CycleBreaker that watched for repeating sensor
signatures and injected a random perturbation. Once the escape behaviour
below existed the breaker fired between zero and twenty-four times across
fifty episodes and removing it changed success rates by at most 0.02, so it
is gone.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TurnCommit:
    """Remembers which way the robot decided to turn while it is boxed in."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._direction: Optional[float] = None

    def direction(self, prefer_left: bool) -> float:
        """+1 to turn left, -1 to turn right. `prefer_left` is consulted only
        the first time, when the commitment is made."""
        if self._direction is None:
            self._direction = 1.0 if prefer_left else -1.0
        return self._direction

    def release(self) -> None:
        """Called once the way ahead is clear, so the next jam decides afresh."""
        self._direction = None


def escape_action(profile: np.ndarray, headings: np.ndarray, ahead: float,
                   commit: TurnCommit, max_steer: float,
                   speed_fn) -> Tuple[float, float]:
    """Full-lock turn towards the side with more room, at a speed that lets
    the robot make it. Returns (steering_deg, speed).

    Speed comes off the same ramp normal driving uses rather than being
    pinned low: pinned at a floor of zero the robot would stop dead and stay
    stopped, whereas on the ramp it stops while facing the obstacle and
    picks up again the moment the turn has opened the way ahead.
    """
    left_room = float(profile[headings > 0].max())
    right_room = float(profile[headings < 0].max())
    steer = max_steer * commit.direction(left_room >= right_room)
    return steer, speed_fn(ahead, max_steer)
