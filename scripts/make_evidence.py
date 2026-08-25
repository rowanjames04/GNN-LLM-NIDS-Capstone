"""Phase 6a: turn detections into evidence packs -- the GNN/LLM contract.

Loads a trained checkpoint, scores windows of the test split, and emits one
[[Evidence Pack Schema]] document per detection: the flow in real units, the
model's confidence and full class distribution, why it fired (integrated
gradients, fusion weights, neighbour occlusion), what the hosts involved were
doing, and static per-family guidance (D10).

**This is the ordering constraint of Stage B.** Phase 7's prompts and
groundedness checker are written against these packs, so one real pack has to
exist before Phase 7 is built -- otherwise Phase 7 gets written against an
imagined schema and then written again.

Usage:
    python scripts/make_evidence.py --smoke
    python scripts/make_evidence.py --checkpoint results/checkpoints/NF-ToN-IoT-v2_full_seed0.pt -n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.explain.attribution import (  # noqa: E402
    integrated_gradients, neighbour_influence, top_features,
)
from gnnids.explain.evidence import (  # noqa: E402
    build_pack, config_hash, host_context, load_guidance, validate_pack,
)
from gnnids.explain.units import UnitRestorer, feature_names  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.models.dual_channel import DualChannelGNN  # noqa: E402
from gnnids.training.loop import pick_device  # noqa: E402


def load_model(ckpt_path: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DualChannelGNN(
        edge_dim=ckpt["edge_dim"], n_classes=ckpt["n_classes"],
        **ckpt["model_cfg"], **ckpt["ablation"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def pack_for_detection(
    *, edge, batch, model, restorer, ip_lookup, families_inv, guidance,
    attributions, alpha, scores, class_probs, threshold, ckpt, provenance_extra,
    feature_names, raw_cols, min_neighbour_importance,
) -> dict:
    """One detection -> one pack."""
    src_local = int(batch.edge_index[0, edge])
    dst_local = int(batch.edge_index[1, edge])

    z_row = batch.edge_attr[edge].detach().cpu().numpy()
    restored = restorer.restore_row(z_row)

    # The flow, in the units it was measured in. Values the model never saw
    # (the IPs) come from the metadata; values it did see are inverted back.
    flow = {
        "src_ip": ip_lookup(src_local),
        "dst_ip": ip_lookup(dst_local),
        **{name: restored[name]["value"] for name in feature_names[:14]
           if name in restored},
        **raw_cols,
    }
    lower_bounds = [n for n, v in restored.items() if not v.get("exact", True)]
    if lower_bounds:
        flow["_values_at_measurement_ceiling"] = lower_bounds

    probs = {families_inv[i]: round(float(p), 5) for i, p in enumerate(class_probs)}
    predicted = max(probs, key=probs.get)

    detection = {
        "score": round(float(scores[edge]), 5),
        "threshold": round(float(threshold), 5),
        "flagged": bool(scores[edge] >= threshold),
        "predicted_class": predicted,
        "class_confidence": probs[predicted],
        # Full distribution, not the argmax -- see the module docstring.
        "class_distribution": probs,
    }

    attribution = {
        "channel_weights": (
            {"attribute": round(float(alpha[edge, 0]), 4),
             "topological": round(float(alpha[edge, 1]), 4)}
            if alpha is not None else None),
        "top_features": [
            {**f, "value": restored.get(f["name"], {}).get("value", f["value"])}
            for f in top_features(attributions[edge], feature_names, z_row)
        ],
        "influential_neighbours": [],
    }

    # Flows sharing an endpoint with this one: the candidates worth occluding.
    ei = batch.edge_index.cpu().numpy()
    touching = np.flatnonzero((ei[0] == src_local) | (ei[1] == dst_local)
                              | (ei[1] == src_local) | (ei[0] == dst_local))
    for n in neighbour_influence(model, batch.x, batch.edge_index,
                                 batch.edge_attr, edge, touching):
        # Only neighbours that actually moved the score. Occluding one edge in
        # a 10,000-edge window typically moves it by ~1e-4, and listing five
        # such flows under the heading "influential neighbours" would hand the
        # LLM five pieces of non-evidence that it would faithfully write up.
        if abs(n["importance"]) < min_neighbour_importance:
            continue
        attribution["influential_neighbours"].append({
            "src_ip": ip_lookup(int(ei[0, n["edge"]])),
            "dst_ip": ip_lookup(int(ei[1, n["edge"]])),
            "importance": n["importance"],
            "summary": (
                f"removing this flow moves the score to {n['score_without']:.3f}"),
        })
    if not attribution["influential_neighbours"]:
        # Stated, not left as an empty list. Absence of a finding and absence of
        # a measurement look identical in JSON, and the LLM must not narrate the
        # first as though it were the second.
        attribution["neighbour_note"] = (
            f"No single neighbouring flow changed this score by more than "
            f"{min_neighbour_importance}. The topological contribution here is "
            f"diffuse across the neighbourhood rather than traceable to one flow.")

    ports = np.array([restorer.restore("L4_DST_PORT", float(r[1]))["value"]
                      for r in batch.edge_attr.detach().cpu().numpy()])
    byts = np.array([restorer.restore("IN_BYTES", float(r[2]))["value"]
                     for r in batch.edge_attr.detach().cpu().numpy()])
    context = {
        "window_flow_count": int(batch.edge_index.shape[1]),
        "window_host_count": int(batch.x.shape[0]),
        "src_host": host_context(ei[0], ei[1], ports, byts, src_local),
        "dst_host": host_context(ei[1], ei[0], ports, byts, dst_local),
    }

    fam_guidance = guidance.get(predicted, guidance.get("Benign", {}))
    return build_pack(
        flow=flow, detection=detection, attribution=attribution,
        context=context,
        guidance={"attack_family": predicted, **fam_guidance},
        provenance={"checkpoint": ckpt_name(ckpt), **provenance_extra},
    )


def ckpt_name(ckpt: dict) -> str:
    return f"seed{ckpt['seed']}_valPRAUC{ckpt['val_pr_auc_at_target']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidence.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("-n", "--n-packs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())

    n_packs = args.n_packs or (cfg["smoke"]["n_packs"] if args.smoke else cfg["n_packs"])
    window = cfg["smoke"]["window_size"] if args.smoke else pre_cfg["graph"]["window_size"]
    max_windows = cfg["smoke"]["max_windows"] if args.smoke else cfg["max_windows"]
    ig_steps = cfg["smoke"]["ig_steps"] if args.smoke else cfg["integrated_gradient_steps"]

    ckpt_path = args.checkpoint or (REPO_ROOT / cfg["checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Train one first.")

    proc = REPO_ROOT / pre_cfg["output"]["dir"]
    print(f"dataset: {ds_cfg['name']}   checkpoint: {ckpt_path.name}"
          f"{'   [SMOKE]' if args.smoke else ''}")
    inputs = load_graph_inputs(proc, ds_cfg["name"])
    pipeline_meta = json.loads((proc / "pipeline.json").read_text())
    restorer = UnitRestorer(pipeline_meta)
    all_feature_names = feature_names(pipeline_meta)
    guidance = load_guidance(REPO_ROOT / cfg["guidance"])
    families_inv = {v: k for k, v in inputs.families.items()}

    device = pick_device(gnn_cfg["train"]["device"])
    model, ckpt = load_model(ckpt_path, device)
    threshold = ckpt["threshold"]
    print(f"device: {device}   threshold {threshold:.4f} (from the checkpoint, "
          f"chosen on validation)\n")

    import pandas as pd
    meta = pd.read_parquet(proc / "meta.parquet",
                           columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])

    s = inputs.splits["test"]
    ds = SnapshotDataset(inputs.src, inputs.dst, inputs.edge_features,
                         inputs.y, inputs.y_multiclass,
                         Split("test", s["start"], s["stop"]), window,
                         max_windows=max_windows)

    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    packs, checked = [], 0

    for wi in range(len(ds)):
        if len(packs) >= n_packs:
            break
        batch = ds[wi].to(device)
        lo, hi = ds.windows[wi]
        rows = np.arange(lo, hi) if ds.row_index is None else ds.row_index[lo:hi]

        with torch.no_grad():
            logit_b, logit_m, alpha = model(batch.x, batch.edge_index, batch.edge_attr)
            scores = torch.sigmoid(logit_b).cpu().numpy()
            probs = torch.softmax(logit_m, dim=-1).cpu().numpy()
        alpha_np = alpha.cpu().numpy() if alpha is not None else None

        flagged = np.flatnonzero(scores >= threshold)
        if not len(flagged):
            continue
        take = flagged[:n_packs - len(packs)]

        attributions = integrated_gradients(
            model, batch.x, batch.edge_index, batch.edge_attr,
            torch.as_tensor(take, device=device), steps=ig_steps,
        ).cpu().numpy()

        for edge in take:
            row = int(rows[edge])
            pack = pack_for_detection(
                edge=int(edge), batch=batch, model=model, restorer=restorer,
                ip_lookup=lambda local, b=batch, r=rows: _ip_for(local, b, r, meta),
                families_inv=families_inv, guidance=guidance,
                attributions=attributions, alpha=alpha_np, scores=scores,
                class_probs=probs[edge], threshold=threshold, ckpt=ckpt,
                provenance_extra={
                    "dataset": ds_cfg["name"], "split": "test",
                    "row_index": row, "window_index": wi,
                    "config_hash": config_hash(gnn_cfg),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                feature_names=all_feature_names,
                min_neighbour_importance=cfg["min_neighbour_importance"],
                raw_cols={"true_label": families_inv[int(inputs.y_multiclass[row])]},
            )
            validate_pack(pack)
            checked += 1
            packs.append(pack)

    name = f"{'smoke_' if args.smoke else ''}evidence_{ds_cfg['name']}.json"
    (out_dir / name).write_text(json.dumps(packs, indent=2))

    print(f"{len(packs)} packs, all {checked} validated against schema "
          f"{packs[0]['schema_version'] if packs else 'n/a'}")
    if packs:
        p = packs[0]
        print(f"\nexample: {p['flow']['src_ip']} -> {p['flow']['dst_ip']}  "
              f"score {p['detection']['score']:.4f}  "
              f"predicted {p['detection']['predicted_class']} "
              f"(true {p['flow'].get('true_label')})")
        cw = p["attribution"]["channel_weights"]
        if cw:
            print(f"  channels: attribute {cw['attribute']:.3f} / "
                  f"topological {cw['topological']:.3f}")
        for f in p["attribution"]["top_features"][:3]:
            print(f"  {f['name']:<28} {f['value']:>14}  {f['contribution']:+.4f}  "
                  f"{f['direction']}")
        print(f"  neighbours considered: {len(p['attribution']['influential_neighbours'])}")
    print(f"\nwritten -> {(out_dir / name).relative_to(REPO_ROOT)}\n")


def _ip_for(local: int, batch, rows: np.ndarray, meta) -> str:
    """Local node index -> the IP string it stands for.

    build_snapshot renumbers each window's hosts 0..N-1, so recovering the
    address means finding any edge with that endpoint and reading the original
    row. Without this the pack would carry node indices, which mean nothing
    outside the window and would be exactly the model-internal detail the
    schema forbids.
    """
    ei = batch.edge_index.cpu().numpy()
    hit = np.flatnonzero(ei[0] == local)
    if len(hit):
        return str(meta.IPV4_SRC_ADDR.iloc[int(rows[hit[0]])])
    hit = np.flatnonzero(ei[1] == local)
    if len(hit):
        return str(meta.IPV4_DST_ADDR.iloc[int(rows[hit[0]])])
    return "unknown"


if __name__ == "__main__":
    main()
