_CANONICAL_TF = {
    "15min":  "15Min",
    "30min":  "30Min",
    "1h":     "1Hour",  "1hour": "1Hour",
    "2h":     "2Hour",  "2hour": "2Hour",
    "4h":     "4Hour",  "4hour": "4Hour",
    "1d":     "1Day",   "1day":  "1Day",
}

def canonical_timeframe(tf: str) -> str:
    if not tf:
        return tf
    tf_clean = tf.strip().lower()
    return _CANONICAL_TF.get(tf_clean, tf)


def timeframe_to_code(tf: str) -> str:
    tf = canonical_timeframe(tf)
    mapping = {
        "15Min": "M15",
        "30Min": "M30",
        "1Hour": "H1",
        "2Hour": "H2",
        "4Hour": "H4",
        "1Day":  "D1",
    }
    return mapping.get(tf, tf)

def get_bars_per_day(tf: str) -> float:
    tf = canonical_timeframe(tf)
    mapping = {
        "15Min": 32,
        "30Min": 16,
        "1Hour":  8,
        "2Hour":  4,
        "4Hour":  2,
        "1Day":   1,
    }
    return mapping.get(tf, 1)