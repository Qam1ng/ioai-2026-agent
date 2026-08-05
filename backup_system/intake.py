"""Turn the operator's pasted Starter prompt into the few typed facts we need.

The rules give the operator exactly one lever: paste the exact Starter prompt.
Everything else the system must work out for itself. That splits cleanly in two:

* The **agents** need no help. The Starter prompt is handed to them verbatim, so
  whatever it says -- including instructions this module never anticipated, like
  the Report the submitted code has to carry -- reaches them unedited. Nothing
  here summarises or filters it.
* The **Broker** needs machine values. `kernels push --timeout` takes an integer;
  a competition slug has to match exactly. Those cannot be approximately
  understood, so one small agent extracts them and this module then refuses to
  believe it without a check.

Which facts have to come from the text was measured, not assumed. For these
private IOAI competitions the Kaggle API gives us very little:

    competitions_list(search=slug)         -> 0 rows (private comps are not searchable)
    competition_get_settings(slug)         -> 403 Forbidden
    competition_get_submission_limits(slug)-> OK: numAllowedNow / numTotal

So the deadline is NOT available from Kaggle and must be read from the task
text, while the submission budget is available and is therefore never taken on
the text's word.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: Kaggle's own hard ceiling for a kernel run. A task cannot ask for more, and a
#: parsed value beyond it means we misread the text.
KAGGLE_MAX_KERNEL_SECONDS = 12 * 3600
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")


class IntakeError(RuntimeError):
    """Raised when the Starter prompt cannot be turned into a runnable task."""


@dataclass(frozen=True)
class StarterTask:
    slug: str
    kernel_timeout_seconds: int
    deadline_epoch: float
    deadline_source: str          # "starter_prompt" | "operator_minutes"
    max_submissions: int
    submissions_used_today: int
    raw_prompt: str
    notes: str = ""

    def minutes_left(self, now: float | None = None) -> float:
        return max(0.0, (self.deadline_epoch - (now or time.time())) / 60)

    def summary(self) -> dict:
        return {
            "slug": self.slug,
            "kernel_timeout_seconds": self.kernel_timeout_seconds,
            "deadline_utc": datetime.fromtimestamp(
                self.deadline_epoch, UTC
            ).isoformat(),
            "deadline_source": self.deadline_source,
            "max_submissions": self.max_submissions,
            "submissions_used_today": self.submissions_used_today,
            "minutes_left": round(self.minutes_left(), 1),
            "notes": self.notes[:400],
        }


def extract_json(text: str) -> dict:
    """Pull the first JSON object carrying a `slug` out of a model's reply."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "slug" in value:
            return value
    raise IntakeError("the intake agent did not return a JSON object with a slug")


def parse_deadline(value: object) -> float | None:
    """Accept an ISO timestamp; anything ambiguous is refused rather than guessed."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        # A wall-clock time with no zone is not a deadline we can act on: being
        # wrong by a timezone means submitting after the competition closed.
        return None
    return stamp.timestamp()


def coerce_timeout(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return None
    if seconds <= 0 or seconds > KAGGLE_MAX_KERNEL_SECONDS:
        return None
    return seconds


def build_task(
    parsed: dict,
    *,
    raw_prompt: str,
    limits: dict,
    fallback_minutes: float | None,
    now: float | None = None,
) -> StarterTask:
    """Validate what the intake agent read, and prefer Kaggle wherever it can speak.

    Fail closed on every value the rules make load-bearing: a missing kernel
    timeout invalidates the submission, and a missing or unparseable deadline
    means we cannot know when to stop.
    """
    now = now or time.time()
    slug = str(parsed.get("slug", "")).strip().lower()
    if not _SLUG_RE.match(slug):
        raise IntakeError(
            f"the intake agent returned an implausible competition slug: {slug!r}"
        )

    timeout = coerce_timeout(parsed.get("kernel_timeout_seconds"))
    if timeout is None:
        raise IntakeError(
            "no usable kernel time limit was found in the Starter prompt. IOAI "
            "invalidates any solution pushed without --timeout, so refusing to "
            "start rather than guessing a value"
        )

    deadline = parse_deadline(parsed.get("deadline_iso"))
    source = "starter_prompt"
    if deadline is None or deadline <= now:
        if fallback_minutes is None or fallback_minutes <= 0:
            raise IntakeError(
                "the Starter prompt carried no usable future deadline and no "
                "--minutes fallback was given; refusing to start without "
                "knowing when submissions must stop"
            )
        deadline = now + fallback_minutes * 60
        source = "operator_minutes"

    # The budget is one of the few things Kaggle will tell us, so the text never
    # gets a vote on it.
    allowed = limits.get("num_allowed_now")
    if not isinstance(allowed, int) or allowed <= 0:
        raise IntakeError(
            f"Kaggle did not report a usable remaining-submission count for "
            f"{slug!r}; refusing to invent a budget"
        )
    used = limits.get("num_today")
    used = used if isinstance(used, int) and used >= 0 else 0

    return StarterTask(
        slug=slug,
        kernel_timeout_seconds=timeout,
        deadline_epoch=deadline,
        deadline_source=source,
        max_submissions=allowed,
        submissions_used_today=used,
        raw_prompt=raw_prompt,
        notes=str(parsed.get("notes", ""))[:1000],
    )


def read_starter_prompt(source: str) -> str:
    """`-` reads stdin, anything else is a path. The text is never edited."""
    if source == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(source).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        raise IntakeError("the Starter prompt is empty")
    return text


def offline_task(
    *, slug: str, kernel_timeout_seconds: int, minutes: float, raw_prompt: str = "",
    max_submissions: int = 50,
) -> StarterTask:
    """A task built without Kaggle, for rehearsals that must not touch the API."""
    timeout = coerce_timeout(kernel_timeout_seconds)
    if timeout is None:
        raise IntakeError("kernel_timeout_seconds must be a positive number of seconds")
    if not math.isfinite(minutes) or minutes <= 0:
        raise IntakeError("minutes must be positive")
    return StarterTask(
        slug=slug, kernel_timeout_seconds=timeout,
        deadline_epoch=time.time() + minutes * 60,
        deadline_source="operator_minutes", max_submissions=max_submissions,
        submissions_used_today=0, raw_prompt=raw_prompt,
        notes="offline rehearsal task",
    )
