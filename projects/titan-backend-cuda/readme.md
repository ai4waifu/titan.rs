# titan-backend-cuda

CUDA backend boundary for Titan.rs. Driver loading and target compilation are
kept behind the backend contract and are unavailable when no CUDA driver exists.
