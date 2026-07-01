# ---------------------------------------------------------------------------
# scanner/parsing.py — Serial parsing (pure functions)
#
# Turn a received telemetry line into numbers.  No Qt, no shared state — safe to
# unit-test in isolation.
# ---------------------------------------------------------------------------

import numpy as np


def parse_keyed_fields(line):
    """Parse a ``key:value>key:value|...`` line into a {key: float} dict.

    Anything after the first ``|`` is ignored, and segments without a numeric
    value are skipped.  Keys are lower-cased and stripped.
    """
    fields = {}
    payload = line.split("|", 1)[0]
    for segment in payload.split(">"):
        if not segment or ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        try:
            fields[key.strip().lower()] = float(value.strip())
        except ValueError:
            continue
    return fields


def parse_serial_line(line):
    """Parse one telemetry line into ``(t, rp, l, mag, width, crack_x, crack_size)``.

    Supports the keyed format (``t:..>l:..>rp:..>...``) and the legacy
    whitespace format (``t s1 s2``).  ``t`` and ``l`` are required; ``rp``
    defaults to 0.  Optional fields are ``None`` when absent/non-finite.
    Raises ``ValueError`` when the line cannot be parsed.
    """
    if not line:
        raise ValueError("Empty serial line")

    if ">" in line and ":" in line:
        fields = parse_keyed_fields(line)

        t = fields.get("t")
        l_val = fields.get("l")
        rp_val = fields.get("rp", 0.0)
        mag_val = fields.get("mag")
        width_val = fields.get("width")
        crack_x_val = fields.get("crack_x")
        crack_size_val = fields.get("crack_size")
        if t is None or l_val is None:
            raise ValueError("Missing required keyed fields")
        if not np.isfinite(t) or not np.isfinite(l_val):
            raise ValueError("Non-finite required keyed fields")
        if not np.isfinite(rp_val):
            rp_val = 0.0
        if mag_val is not None and not np.isfinite(mag_val):
            mag_val = None
        if width_val is not None and not np.isfinite(width_val):
            width_val = None
        if crack_x_val is not None and not np.isfinite(crack_x_val):
            crack_x_val = None
        if crack_size_val is not None and not np.isfinite(crack_size_val):
            crack_size_val = None
        return float(t), float(rp_val), float(l_val), mag_val, width_val, crack_x_val, crack_size_val

    t, s1, s2 = map(float, line.split())
    if not np.isfinite(t) or not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("Non-finite whitespace fields")
    return t, s1, s2, None, None, None, None


def extract_reject_reason(line):
    """Return the value following ``reject_reason=`` in a line, or ``None``.

    The value is trimmed at the first separator and stripped of brackets; a
    literal ``-`` (no reason) returns ``None``.
    """
    if not line:
        return None

    text = line.strip()
    marker = "reject_reason="
    idx = text.lower().find(marker)
    if idx < 0:
        return None

    value = text[idx + len(marker):].strip()
    if not value:
        return None

    for sep in ("|", ">", " ", "\t", ","):
        sep_index = value.find(sep)
        if sep_index >= 0:
            value = value[:sep_index]
            break

    parsed_value = value.strip().strip("[]")
    if parsed_value == "-":
        return None
    return parsed_value or None


def parse_crack_event(line, fallback_t):
    """Return ``(t, mag, crack_size)`` for a crack event line, or ``None``.

    A crack event is a line carrying a finite, non-zero ``mag``.  When the line
    has no ``t``, ``fallback_t`` (the most recent sample time) is used.
    """
    if ">" not in line or ":" not in line:
        return None

    fields = parse_keyed_fields(line)
    mag_val = fields.get("mag")
    if mag_val is None or not np.isfinite(mag_val) or float(mag_val) == 0.0:
        return None

    t_val = fields.get("t")
    if t_val is None:
        t_val = fallback_t
    if t_val is None or not np.isfinite(t_val):
        return None

    crack_size_val = fields.get("crack_size")
    if crack_size_val is None or not np.isfinite(crack_size_val):
        crack_size_val = None

    return float(t_val), float(mag_val), (float(crack_size_val) if crack_size_val is not None else None)
