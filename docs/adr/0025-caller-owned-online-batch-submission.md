# Caller-owned online batch submission

Status: Accepted. Supersedes ADR 0022 and the Import Session clauses of ADR 0006.
ADR 0006's searchable online topology contract remains in force.

Online bulk loading submits ordinary `Index::batch_mutate` operations. Callers
own batch size, bounded concurrency, and each result; in-repository loading
loops submit sequentially. The Import Session API, process-local Batch Tokens,
adaptive concurrency controller, Fixup Backlog gate, and session-only runtime
configuration and telemetry are removed. No aliases or replacement session
layer are retained for this unreleased API.

Every batch retains ordinary validation, atomicity, bounded whole-operation
retry, cancellation, and unknown-commit semantics. An unknown commit outcome
must be recovered through the ordinary mutation protocol, never blindly
retried. Runtime admission and demand-driven maintenance remain independently
bounded. There is no whole-load transaction, completion barrier, or retained
adaptive-throughput guarantee.

Resumable offline construction is a separate capability specified in
[bulk-build.md](../design/bulk-build.md), with its own feasibility gates. Its
durable task identities do not stand in for transaction identities. Removing
the process-local session does not itself implement that construction path or
demonstrate a load-time improvement.
