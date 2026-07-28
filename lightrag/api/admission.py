"""Admission ticket handed from the ASGI middleware to the route (LR2 §9.3).

The capacity decision has to happen before the first ``receive()`` — a request
that will be refused must never have its body read — but the reservation it
takes has to stay held until the documents are visible in ``doc_status``, which
happens in a background task long after the response is sent. The ticket is how
that one reservation travels across the three owners:

    ASGI middleware → endpoint adopt → managed background task → release

Exactly one owner is responsible at any moment, and each hands over explicitly:

* not adopted (rejected before the route, or the route never ran) — the
  middleware releases it in its ``finally``;
* adopted, not handed off (validation rejection, streaming failure) — the
  endpoint releases it in its ``finally``;
* handed off — the background task's ``finally`` / backstop releases it.

Returning the HTTP response must NOT release: the client has been told the
document was accepted, so the reservation stays charged until it lands.

This module is intentionally dependency-free so both the middleware and the
document routes can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Key under which the ticket is published in the ASGI ``scope["state"]`` dict.
# Starlette exposes that dict as ``request.state``, which is how the route reads
# it without the middleware having to know anything about FastAPI.
ADMISSION_STATE_KEY = "lightrag_admission_ticket"


@dataclass
class AdmissionTicket:
    """One pending-enqueue reservation in flight across layers."""

    token: str
    weight: int = 1
    adopted: bool = False


def publish_admission_ticket(scope: dict[str, Any], ticket: AdmissionTicket) -> None:
    """Put ``ticket`` where the route can find it."""
    scope.setdefault("state", {})[ADMISSION_STATE_KEY] = ticket


def adopt_admission_ticket(request: Any) -> Optional[AdmissionTicket]:
    """Take ownership of the middleware's reservation, if there is one.

    Returns the ticket (now marked adopted, so the middleware's ``finally``
    leaves it alone) or ``None`` when no middleware ran for this request — the
    route then takes its own reservation exactly as before, which is what keeps
    admission correct when the middleware is not installed (capacity disabled)
    or when the route is exercised directly by a test rig.
    """
    state = getattr(request, "state", None)
    ticket = getattr(state, ADMISSION_STATE_KEY, None) if state is not None else None
    if not isinstance(ticket, AdmissionTicket):
        return None
    ticket.adopted = True
    return ticket
