"""Terminal-only, budgeted ESMFold monomer pipeline on Modal.

Commands:
  uv run modal run pipeline.py::plan --budget-usd 30
  uv run modal run pipeline.py::fold
  uv run modal run pipeline.py::status
"""

import modal

app = modal.App("wet-run-monomer-folding")
volume = modal.Volume.from_name("viral-metagenome-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

DATA_FASTA = "/data/wet_run_mmseqs_rep_seq.fasta"
OUTPUT_DIR = "/data/wet_run_monomers"
PLAN_PATH = "/data/monomer_plan_under_30.json"
REPORT_PATH = "/data/monomer_cost_report.json"
SUMMARY_PATH = "/data/monomer_run_summary.json"
MAX_RESIDUES = 800
MAX_CONTAINERS = 4
GPU_RATE = 0.000583       # Modal A100 40 GB, USD/s
CPU_RATE = 2 * 0.0000131  # two physical cores, USD/s
MEMORY_RATE = 32 * 0.00000222  # assumed 32 GiB, USD/s

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "biopython==1.88",
        "numpy<2",
        "torch==2.3.1",
        "transformers==4.41.2",
        "accelerate==0.31.0",
    )
)


@app.function(image=image, volumes={"/data": volume})
def inventory():
    """Return real FASTA records and completed monomer IDs."""
    import os
    from Bio import SeqIO

    volume.reload()
    records = []
    for record in SeqIO.parse(DATA_FASTA, "fasta"):
        sequence = str(record.seq).rstrip("*").upper()
        raw_bin = record.id.split("___", 1)[0]
        records.append({
            "id": record.id,
            "sequence": sequence,
            "length": len(sequence),
            "bin": raw_bin.replace("scaffoldssta_bin", "scaffolds.fasta_bin"),
        })
    completed = {
        name.removesuffix(".pdb")
        for name in os.listdir(OUTPUT_DIR)
        if name.endswith(".pdb")
    } if os.path.isdir(OUTPUT_DIR) else set()
    return records, sorted(completed)


@app.function(image=image, volumes={"/data": volume})
def write_artifact(path: str, payload):
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    volume.commit()
    return path


@app.function(image=image, volumes={"/data": volume})
def read_plan():
    import json
    import os
    volume.reload()
    with open(PLAN_PATH) as handle:
        plan = json.load(handle)
    completed = {
        name.removesuffix(".pdb")
        for name in os.listdir(OUTPUT_DIR)
        if name.endswith(".pdb")
    } if os.path.isdir(OUTPUT_DIR) else set()
    return plan, sorted(completed)


@app.cls(
    image=image,
    gpu="A100",
    timeout=1800,
    max_containers=MAX_CONTAINERS,
    secrets=[hf_secret],
    volumes={"/data": volume},
)
class MonomerFolder:
    @modal.enter()
    def load(self):
        import time
        started = time.perf_counter()
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").cuda()
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    @modal.method()
    def predict(self, record: dict, output_dir: str):
        import json
        import os
        import time

        os.makedirs(output_dir, exist_ok=True)
        pdb_path = os.path.join(output_dir, f"{record['id']}.pdb")
        json_path = os.path.join(output_dir, f"{record['id']}.json")
        if os.path.isfile(pdb_path):
            return {"id": record["id"], "status": "existing", "inference_seconds": 0.0,
                    "model_load_seconds": self.load_seconds}

        started = time.perf_counter()
        inputs = self.tokenizer(record["sequence"], add_special_tokens=False)
        input_ids = self.torch.tensor(inputs["input_ids"], device="cuda").unsqueeze(0)
        attention_mask = self.torch.tensor(inputs["attention_mask"], device="cuda").unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pdb = self.model.output_to_pdb(output)[0]
        with open(pdb_path, "w") as handle:
            handle.write(pdb)

        confidence = output.plddt.mean().item()
        if confidence <= 1:
            confidence *= 100
        result = {
            "id": record["id"],
            "length": record["length"],
            "bin": record["bin"],
            "model": "facebook/esmfold_v1",
            "mean_plddt": round(confidence, 2),
            "inference_seconds": round(time.perf_counter() - started, 3),
            "model_load_seconds": round(self.load_seconds, 3),
            "status": "completed",
            "pdb_path": pdb_path,
        }
        with open(json_path, "w") as handle:
            json.dump(result, handle, indent=2)
        return result


def _balanced_budget_plan(candidates, seconds_per_l2, load_seconds, budget_usd):
    """Round-robin shortest-first selection across MAGs under a compute ceiling."""
    from collections import defaultdict, deque

    rate = GPU_RATE + CPU_RATE + MEMORY_RATE
    planning_ceiling = budget_usd * 0.90
    startup = load_seconds * min(MAX_CONTAINERS, len(candidates))
    allowed_inference = max(0.0, planning_ceiling / rate - startup)
    groups = defaultdict(list)
    for record in candidates:
        groups[record["bin"]].append(record)
    queues = {
        key: deque(sorted(items, key=lambda row: (row["length"], row["id"])))
        for key, items in sorted(groups.items())
    }
    selected, inference = [], 0.0
    while queues:
        progressed = False
        for key in list(queues):
            queue = queues[key]
            if not queue:
                del queues[key]
                continue
            record = queue[0]
            seconds = seconds_per_l2 * record["length"] ** 2
            if inference + seconds <= allowed_inference:
                selected.append(queue.popleft())
                inference += seconds
                progressed = True
            else:
                del queues[key]
        if not progressed:
            break
    estimated_cost = (inference + startup) * rate
    return selected, inference, estimated_cost, planning_ceiling


@app.local_entrypoint()
def plan(budget_usd: float = 30.0, sample_size: int = 3):
    """Dry-run monomers, estimate cost, and persist a plan below the cap."""
    import json
    import statistics

    if not 0 < budget_usd <= 30:
        raise ValueError("budget-usd must be positive and no greater than 30")
    if sample_size < 3:
        raise ValueError("sample-size must be at least 3")

    records, completed = inventory.remote()
    eligible = [
        row for row in records
        if 1 <= row["length"] <= MAX_RESIDUES and row["id"] not in completed
    ]
    ordered = sorted(eligible, key=lambda row: (row["length"], row["id"]))
    indexes = [round(i * (len(ordered) - 1) / (sample_size - 1)) for i in range(sample_size)]
    sample = [ordered[index] for index in indexes]
    results = list(MonomerFolder().predict.map(
        sample, kwargs={"output_dir": "/data/dry_run_monomers"}
    ))
    coefficients = [
        result["inference_seconds"] / record["length"] ** 2
        for record, result in zip(sample, results)
    ]
    seconds_per_l2 = statistics.median(coefficients)
    model_load = statistics.median(result["model_load_seconds"] for result in results)
    selected, inference, cost, ceiling = _balanced_budget_plan(
        eligible, seconds_per_l2, model_load, budget_usd
    )
    plan_rows = [{k: value for k, value in row.items() if k != "sequence"} for row in selected]
    report = {
        "budget_cap_usd": budget_usd,
        "planning_ceiling_usd": round(ceiling, 2),
        "reserve_usd": round(budget_usd - ceiling, 2),
        "source_records": len(records),
        "already_completed": len(completed),
        "length_eligible_remaining": len(eligible),
        "maximum_residues": MAX_RESIDUES,
        "dry_run_lengths": [row["length"] for row in sample],
        "dry_run_inference_seconds": [row["inference_seconds"] for row in results],
        "median_model_load_seconds": round(model_load, 3),
        "seconds_per_length_squared": seconds_per_l2,
        "planned_monomers": len(plan_rows),
        "represented_mags": len({row["bin"] for row in selected}),
        "projected_inference_gpu_hours": round(inference / 3600, 2),
        "projected_compute_cost_usd": round(cost, 2),
        "rates": {"gpu": GPU_RATE, "cpu": CPU_RATE, "memory": MEMORY_RATE},
        "excludes": "storage and taxes; 10% reserve retained for runtime variance/retries",
    }
    write_artifact.remote(PLAN_PATH, plan_rows)
    write_artifact.remote(REPORT_PATH, report)
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def fold():
    """Resume and execute only the persisted budgeted monomer plan."""
    import json

    records, _ = inventory.remote()
    by_id = {row["id"]: row for row in records}
    plan_rows, completed = read_plan.remote()
    selected = [by_id[row["id"]] for row in plan_rows if row["id"] not in completed]
    print(f"planned={len(plan_rows)} completed={len(completed)} remaining={len(selected)}")
    if not selected:
        return
    outputs = list(MonomerFolder().predict.map(
        selected, kwargs={"output_dir": OUTPUT_DIR}, return_exceptions=True
    ))
    failures = [str(row) for row in outputs if isinstance(row, Exception)]
    completed_rows = [row for row in outputs if isinstance(row, dict)]
    summary = {
        "planned": len(plan_rows),
        "completed_before_run": len(completed),
        "attempted": len(outputs),
        "completed_this_run": sum(row["status"] == "completed" for row in completed_rows),
        "failures": failures,
    }
    write_artifact.remote(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError("Some folds failed; rerun to resume")


@app.local_entrypoint()
def status():
    """Print machine-readable monomer status to stdout."""
    import json
    import os

    records, completed = inventory.remote()
    try:
        plan_rows, _ = read_plan.remote()
    except Exception:
        plan_rows = []
    print(json.dumps({
        "source_records": len(records),
        "planned": len(plan_rows),
        "completed": len(completed),
        "remaining": max(0, len(plan_rows) - len(set(completed) & {row['id'] for row in plan_rows})),
        "output_dir": OUTPUT_DIR,
    }, indent=2))
