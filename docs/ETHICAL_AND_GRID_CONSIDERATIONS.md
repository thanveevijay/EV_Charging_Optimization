# Operational and Grid Considerations

This project is a research simulator, not a production charging controller.

## Operational constraints to model
- Maximum power per charger.
- Station-level power limits.
- Vehicle departure deadlines.
- Charger compatibility.
- Emergency or priority charging requirements.
- Communication failures and stale station state.

## Responsible deployment
A real deployment would require hardware-in-the-loop testing, fail-safe fallback scheduling, operator override, cybersecurity review, and validation against local electrical regulations. Simulation results alone do not establish grid safety or real-world savings.
