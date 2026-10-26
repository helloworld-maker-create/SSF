"""Pure scheduling helpers shared by the controller and GPU monitor."""


def monitor_slots(rows, world, rank):
    """Equal collective calls on all ranks, without adding evaluation rows."""
    if not rows or world <= 0 or not 0 <= rank < world:
        raise ValueError('Invalid monitor rows/topology')
    for start in range(0, len(rows), world):
        slot = start + rank
        yield slot < len(rows), rows[slot % len(rows)]
