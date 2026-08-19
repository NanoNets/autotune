# knowledge/ &mdash; the memory store

Cross-run memory, mirroring how Claude's own memory works: one **belief** per markdown file with
frontmatter, plus an always-loaded index (`beliefs.md`).

Each belief is an evidence-backed claim keyed by context (`{arch_family, hardware_class, dataset_shape}`)
so the next run retrieves what's relevant and starts smart. Authored and curated by the agent.

See the `Belief` contract in `loop/contracts.md`. *(Convention defined; population begins at the intelligence stage.)*
