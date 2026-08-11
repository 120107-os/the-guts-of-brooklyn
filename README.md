# Wet-run monomer folding

A terminal-only UNIX-style pipeline for budgeted ESMFold monomer prediction on
Modal. There is no frontend, web server, dimer workflow, or assembly workflow.

## Source and outputs

The source is the real wet-run MMseqs2 representative FASTA in Modal Volume
`viral-metagenome-data`:

```text
/data/wet_run_mmseqs_rep_seq.fasta
```

The pipeline writes only monomer artifacts:

```text
/data/wet_run_monomers/{protein_id}.pdb
/data/wet_run_monomers/{protein_id}.json
/data/monomer_plan_under_30.json
/data/monomer_cost_report.json
/data/monomer_run_summary.json
```

## Setup

```sh
uv sync
uv run modal setup
```

## Commands

Create a plan using a real three-fold dry run. This does not execute the wet
plan:

```sh
uv run modal run pipeline.py::plan --budget-usd 30 --sample-size 3
```

Execute or resume exactly the persisted plan:

```sh
uv run modal run pipeline.py::fold
```

Print JSON status to stdout:

```sh
uv run modal run pipeline.py::status
```

## Current dry-run estimate

The completed dry run sampled real monomers of 49, 219, and 800 residues.
Measured inference times were 2.921, 3.294, and 61.529 seconds, with median model
load time 75.269 seconds.

The resulting plan is:

```text
Source representatives:       35,941
Length eligible (<=800 aa):   34,546
Planned monomers:             12,514
MAGs represented:             64
Projected inference:          10.94 A100 GPU-hours
Projected compute cost:       $27.00
Budget reserve:               $3.00
User cap:                     $30.00
```

The estimate includes one A100 40 GB, two CPU cores, and 32 GiB memory at the
recorded Modal rates. Storage and taxes are excluded. The 10% reserve is held
for runtime variation and retries; this is a planning estimate rather than a
billing guarantee.

No wet monomer plan has been launched yet.
