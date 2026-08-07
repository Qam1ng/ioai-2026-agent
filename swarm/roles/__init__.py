"""The specialist roles, and the registry the pod builds them from.

Every role is short-lived, tool-restricted and ends by emitting one JSON report.
The registry keys match ``swarm.config.ROLES`` so that model routing
(``SwarmConfig.model_for(name)``) and role construction use the same names — a
mismatch there would silently route a role to the default model.
"""

from __future__ import annotations

from .aggregator import AggregatorRole
from .base import Role, RoleResult, extract_json, load_prompt
from .coder import CoderRole
from .compliance import ComplianceRole
from .designer import DesignerRole
from .manager import ManagerRole
from .prober import ProberRole
from .profiler import ProfilerRole
from .reporter import ReporterRole
from .tuner import TunerRole
from .verifier import VerifierRole

ROLE_CLASSES: dict[str, type[Role]] = {
    "profiler": ProfilerRole,
    "manager": ManagerRole,
    "designer": DesignerRole,
    "coder": CoderRole,
    "tuner": TunerRole,
    "verifier": VerifierRole,
    "prober": ProberRole,
    "compliance": ComplianceRole,
    "aggregator": AggregatorRole,
    "reporter": ReporterRole,
}


def get_role_class(name: str) -> type[Role]:
    """Look up a role class by name, failing loudly on a typo.

    A missing role must not fall back to a generic ``Role``: an unrestricted
    tool set is exactly what the role split exists to prevent.
    """
    try:
        return ROLE_CLASSES[name]
    except KeyError:
        raise KeyError(f"unknown role {name!r}; known: {sorted(ROLE_CLASSES)}") from None


__all__ = [
    "ROLE_CLASSES",
    "AggregatorRole",
    "CoderRole",
    "ComplianceRole",
    "DesignerRole",
    "ManagerRole",
    "ProberRole",
    "ProfilerRole",
    "ReporterRole",
    "Role",
    "RoleResult",
    "TunerRole",
    "VerifierRole",
    "extract_json",
    "get_role_class",
    "load_prompt",
]
