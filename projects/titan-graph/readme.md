# titan-graph

Core Graph IR contract for Titan (Living DXO `16` / phase-03):

- schema-versioned `Graph` / `Value` / `Node`
- deterministic `semantic_hash`
- debug JSON serialize / roundtrip
- Living `15` IR diagnostics (`DXO_IR_*`)
- pass declaration registry skeleton

Runtime `OpRequest` remains the eager dispatch bridge; compiled ExecutablePlan lands in later stages.
