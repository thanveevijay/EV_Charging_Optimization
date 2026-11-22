# API Input Contract

Validate timestamps, numeric ranges, station identifiers, connector types, and requested energy before scheduling.

Reject malformed payloads with a stable error code and a field-level explanation. Never coerce an invalid timestamp or silently replace a missing power limit with an unsafe default.