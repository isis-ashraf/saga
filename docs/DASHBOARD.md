# S.A.G.A. Dashboard Guide

The main UI is [story_dashboard.py](/B:/Documents/PyCharm/graduationProject/story_dashboard.py).

## Purpose

The dashboard is no longer just a review tool. It is the primary operational surface for:

- ingesting books
- running the pipeline
- observing progress
- reviewing outputs
- exporting the current run
- querying indexed story data

## Main Controls

- Upload books
- Reorder books
- Select scene analysis model
- Select identity model
- Choose scene size
- Run pipeline
- Reset results
- Export JSON contract

## Key Tabs

- `Status`
  Run progress and execution timing
- `Books`
  Current ordered inputs
- `Chapters`
  Chapter extraction output
- `Scenes`
  Per-scene analysis output
- `Entity Registry`
  Tracked entities and mention counts
- `State Transitions`
  State change log and latest state
- `Canon Snapshot`
  State at the current point in reading order
- `Timeline`
  Ordered event timeline
- `Character Timelines`
  Per-character event grouping
- `Alias Map`
  Canonicals and aliases
- `Identity Decisions`
  Identity reasoning outcomes
- `Causal Graph`
  Causal events and links
- `Causal Metrics`
  Graph-level summary metrics
- `Story Search`
  Search over indexed outputs

## Export

Use the `Export JSON Contract` button in the sidebar after a run has produced chapters. The file is meant to be stable enough for handoff to downstream tools and integrations.
