# InfinityDB independent developer validation program

## Objective

Recruit ten independent developers or teams to run a real workload against
InfinityDB and publish reproducible findings. This is a recruitment and
evidence plan, not a claim that ten participants have already joined.

## Participation criteria

Each participant should:
1. Use a non-trivial workload they control (agent, support bot, coding agent,
   research assistant, or multi-tenant application).
2. Record dataset shape without disclosing private user content.
3. Test ingestion, retrieval quality, latency, resource use and failure/recovery.
4. Include environment, commit, dependency versions and exact commands.
5. Publish a report, issue, or pull request with limitations and failures as
   well as successes.

## Ten validation slots

| Slot | Workload focus | Status |
|---|---|---|
| 1 | Personal assistant / long-term preferences | Open |
| 2 | Coding agent / repository knowledge | Open |
| 3 | Customer support / conversation memory | Open |
| 4 | RAG / document memory | Open |
| 5 | Multi-tenant SaaS agent | Open |
| 6 | Temporal facts / changing user state | Open |
| 7 | Local-first / edge deployment | Open |
| 8 | High-throughput ingestion | Open |
| 9 | Multi-agent shared memory | Open |
| 10 | Recovery, deletion and auditability | Open |

## Public report template

- Participant/project (optional anonymous)
- InfinityDB commit and install method
- Hardware, OS, storage and deployment topology
- Workload and dataset size (synthetic or safely described)
- Embedding model/provider and revision
- Ingest throughput and p50/p95 latency
- Query p50/p95 latency and recall methodology
- Peak process RSS and database/index size
- Tenant isolation, deletion and recovery tests
- What failed, what was missing, and what should change
- Reproduction commands and raw artifacts

## Outreach message

> I'm inviting developers to independently evaluate InfinityDB, an open-source
> memory engine for AI agents. The goal is an honest real-workload evaluation,
> not a promotional testimonial. We provide a reproducible benchmark harness
> and want reports that include limitations, failures, hardware, and commands.
> Would you be interested in trying it on one of your agent workloads and
> sharing an independent report?

## Acceptance and integrity

Do not count a participant until they have run a workload and shared a report
or reproducible issue. Do not edit negative findings out of reports. Ask
permission before using a participant's name, logo, quote or results in
marketing. Do not publish private prompts, user data, credentials or proprietary
datasets.
