# ADR 0026: Reserve Bulk Builds through the Index lifecycle

Status: Accepted; format-version paragraph amended 2026-11-03.

## Context

Bulk Build needs a durable identity before loading serving data. A name must not
open incomplete data, and a recovered or cancelled job must not follow name reuse.
Source locators and construction options do not belong on serving read paths.

## Decision

Add Building to the Manifest lifecycle and store the immutable Build Descriptor
at a separate index-owned key. Allocate the ID, reserve its name, and persist both
values in one transaction. Ordinary operations require Active. Recovery checks
the attempted ID after an unknown commit; job handles remain bound to that ID.
Abort permits only unpublished identities and uses the existing Dropping cleanup.
Caller-owned source files survive abort. No worker-owned files are created until
a durable namespace cleanup ledger exists.

The persistent format includes lifecycle byte 2, descriptor key kind 0x05, and
value tag 0x0d. Index storage and bulk artifacts share the format version defined
in ADR 0018; unsupported versions fail closed. There is no released compatibility
requirement. This extends ADR 0017's lifecycle.

## Consequences

Ordinary Manifest reads remain small. Building data is hidden and cancellable.
The reservation API alone does not construct, load, validate, or publish an index;
worker fencing, durable cleanup, and sealed publication remain separate work.
