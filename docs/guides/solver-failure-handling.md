# Solver Failure Handling

Define behavior for infeasible, timed-out, and interrupted optimization runs.

Return a structured status and preserve the last known feasible schedule when safe. Do not dispatch a partial solution without validating all operational constraints.