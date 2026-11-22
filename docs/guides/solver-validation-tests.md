# Solver Validation Tests

Test the optimizer with a single session, multiple compatible sessions, zero available power, a fully occupied station, and an infeasible energy request.

For every result, verify connector exclusivity, non-negative allocations, site power limits, and consistent session status. Include a deterministic small case with an independently calculated expected schedule.