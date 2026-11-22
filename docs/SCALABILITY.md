# Scalability Considerations

Measure simulation runtime as session count and charger count increase. Profile graph construction, state encoding, policy inference, and event processing separately. Avoid optimizing from intuition alone. Consider batching graph operations and using efficient event queues only after profiling identifies a bottleneck.