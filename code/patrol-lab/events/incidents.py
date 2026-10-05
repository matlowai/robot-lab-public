"""Operator-facing incident records built from events. Summaries are factual templates, never judgements."""

from events.rules import Event, _clock

_TEMPLATES = {
    "PERIMETER_BREACH": "{subject} crossed the perimeter outside a gate at {clock}.",
    "RESTRICTED_ZONE_ENTRY": "{subject} entered a restricted zone at {clock} without an authorized credential.",
    "PERSON_FALL": "{subject} has been prone since {clock}.",
    "VEHICLE_IN_PROHIBITED_ZONE": "{subject} entered a vehicle-prohibited zone at {clock}.",
    "GATE_LEFT_OPEN": "{subject} has been open since {clock}; expected closed.",
    "ROUTE_OBSTRUCTION": "{subject} has been blocking a patrol road since {clock}.",
}


def to_incident(event: Event, n: int) -> dict:
    return {
        "id": f"inc_{n:05d}",
        "type": event.type,
        "subject": event.subject,
        "severity": event.severity,
        "t": event.t,
        "t_detected": event.t_detected,
        "location": [round(v, 2) for v in event.location],
        "summary": _TEMPLATES[event.type].format(subject=event.subject, clock=_clock(event.t)),
        "evidence": list(event.evidence),
        "event_ids": [event.id],
    }
