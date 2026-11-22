# Scheduling Policy Notes

The repository includes FCFS as a transparent baseline. Additional policies can be evaluated using the same simulator interface.

## Candidate policies
- Earliest departure first: prioritize vehicles with the least time remaining.
- Highest remaining energy first: prioritize sessions with larger remaining energy, subject to deadline feasibility.
- Weighted urgency: combine departure urgency and remaining energy.
- Forecast-aware scheduling: use predicted arrivals and station demand to avoid excessive peak load.

A policy is only a proposal mechanism. The environment must validate charger availability, compatibility, and station power limits before applying an action.
