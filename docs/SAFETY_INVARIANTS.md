# Safety Invariants

The simulator must never allocate an unavailable charger, exceed a charger rating, exceed a station power cap, or continue charging after a session departs. Enforce invariants in environment code rather than relying on learned policy behavior. Test each invariant with adversarial action sequences.