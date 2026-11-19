"""
MCP (stdio) server for the YOLO dataset curator (CleanCrop).

Lets Claude run the analysis, read the report, record review decisions (same review_state.json the
dashboard uses) and edit YOLO label files with backups.

    claude mcp add --scope user curator -- python "C:\\path\\to\\mcp_server.py"
"""
import os
import sys
import json
import time
import uuid
import socket
import subprocess
from collections import Counter
from datetime import datetime, timezone

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(SCRIPT_DIR, "dataset_curator_yolo.py")
JOBS_DIR = os.path.join(SCRIPT_DIR, "curation_output", "_mcp_jobs")
JOBS = {}  # job_id -> {"proc", "log", "cmd", "started"}

mcp = FastMCP("curator")


# ---------------------------------------------------------------- helpers

def _out_dir(dataset_dir, output_dir=None):
    """Same rule as main(): curation_output/<dataset folder name> next to the CLI script."""
    if output_dir:
        return os.path.abspath(output_dir)
    root = os.environ.get("CURATOR_OUTPUT_ROOT") or os.path.join(SCRIPT_DIR, "curation_output")  # override for tests
    return os.path.abspath(os.path.join(root, os.path.basename(os.path.normpath(dataset_dir))))


def _report(dataset_dir, output_dir=None):
    path = os.path.join(_out_dir(dataset_dir, output_dir), "curation_report.json")
    if not os.path.exists(path):
        raise ValueError(f"No curation_report.json in {os.path.dirname(path)}. Run run_analysis first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _state_path(dataset_dir, output_dir=None):
    return os.path.join(_out_dir(dataset_dir, output_dir), "review_state.json")


def _load_state(dataset_dir, output_dir=None):
    p = _state_path(dataset_dir, output_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}


def _save_state(dataset_dir, state, output_dir=None):
    p = _state_path(dataset_dir, output_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, p)


def _params(report):
    p = {"dup_threshold": 0.97, "lof_threshold": 1.5, "mislabel_conf": 0.7}
    p.update(report.get("params") or {})
    return p


def _live(report):
    return [it for it in report["items"] if not it.get("removed")]


def _is_mislabel(it, p):
    return bool(it.get("pred")) and it["pred"] != it["label"] and (it.get("pred_conf") or 0) >= p["mislabel_conf"]


def _is_outlier(it, p):
    return (it.get("lof") or 0) >= p["lof_threshold"]


def _dup_pairs(report, p):
    items = report["items"]
    out = []
    for pair in report.get("duplicates") or []:
        i, j, sim = int(pair[0]), int(pair[1]), float(pair[2])
        if sim >= p["dup_threshold"] and i < len(items) and j < len(items):
            if not items[i].get("removed") and not items[j].get("removed"):
                out.append((items[i], items[j], sim))
    return out


def _brief(it, out_dir):
    return {"id": it["id"], "image": it.get("image"), "split": it.get("split"), "label": it.get("label"),
            "pred": it.get("pred"), "conf": it.get("pred_conf"), "lof": it.get("lof"),
            "crop": os.path.normpath(os.path.join(out_dir, it["path"])) if it.get("path") else None}


def _dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _run_cli(args, timeout=3600):
    cmd = [sys.executable, CLI] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, cwd=SCRIPT_DIR)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
    return {"returncode": r.returncode, "output_tail": tail}


def _server(dataset_dir, output_dir=None):
    sys.path.insert(0, SCRIPT_DIR)
    from dataset_curator_yolo import CurationServer  # heavy imports (numpy, PIL) only when editing
    return CurationServer(_out_dir(dataset_dir, output_dir))


def _edit_box(dataset_dir, item_id, new_class_id, output_dir=None):
    """Rewrite all boxes of the item's image via CurationServer.edit_image (backup + report re-sync)."""
    srv = _server(dataset_dir, output_dir)
    item = srv.by_id.get(item_id)
    if item is None or item.get("removed"):
        raise ValueError(f"Unknown or already removed item: {item_id}")
    boxes = srv.image_labels(item_id)["boxes"]
    idx = next((n for n, b in enumerate(boxes) if b["class_id"] == item["class_id"]
                and max(abs(a - c) for a, c in zip(b["bbox"], item["bbox"])) < 2e-6), None)
    if idx is None:
        raise ValueError(f"Box of {item_id} not found in {item['label_path']} (file changed since analysis?)")
    if new_class_id is None:
        boxes.pop(idx)
    else:
        boxes[idx]["class_id"] = new_class_id
    result = srv.edit_image(item_id, boxes)
    if new_class_id is not None:  # the relabeled box now is a new line; keep the report entry pointing at it
        it = srv.by_id[item_id]
        it.update({"class_id": new_class_id, "label": srv.report["class_names"][new_class_id], "line": idx})
        it.pop("removed", None)
        from dataset_curator_yolo import save_report
        save_report(srv.report, srv.output_dir)
    return result


# ---------------------------------------------------------------- tools

@mcp.tool()
def run_analysis(dataset_dir: str, model: str = "dinov2", no_cache: bool = False, background: bool = True,
                 golden_split: float | None = None) -> str:
    """Run the curation analysis on a YOLO dataset (folder with data.yaml). Can take ~25 min on CPU the first
    time; with background=True returns a job id to poll with job_status."""
    args = [os.path.abspath(dataset_dir), "--model", model, "--no-serve", "--no-browser"]
    if no_cache:
        args.append("--no-cache")
    if golden_split is not None:
        args += ["--golden-split", str(golden_split)]
    if not background:
        r = _run_cli(args, timeout=4 * 3600)
        r["output_dir"] = _out_dir(dataset_dir)
        return _dumps(r)
    os.makedirs(JOBS_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    log = os.path.join(JOBS_DIR, f"{job_id}.log")
    lf = open(log, "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, CLI] + args, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=SCRIPT_DIR, env=env)
    JOBS[job_id] = {"proc": proc, "log": log, "started": time.time(), "output_dir": _out_dir(dataset_dir)}
    return _dumps({"job_id": job_id, "log": log, "output_dir": _out_dir(dataset_dir)})


@mcp.tool()
def job_status(job_id: str, tail: int = 20) -> str:
    """Status of a background job (running / exit code) plus the last lines of its log."""
    job = JOBS.get(job_id)
    log = job["log"] if job else os.path.join(JOBS_DIR, f"{job_id}.log")
    if not os.path.exists(log):
        return _dumps({"error": f"unknown job {job_id}"})
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        # tqdm writes \r progress updates; keep the latest one per line
        lines = [l.split("\r")[-1] for l in f.read().splitlines() if l.strip()]
    rc = job["proc"].poll() if job else None
    status = "unknown (server restarted)" if not job else ("running" if rc is None else ("done" if rc == 0 else "failed"))
    out = {"job_id": job_id, "status": status, "returncode": rc, "log_tail": "\n".join(lines[-tail:])}
    if job:
        out["elapsed_s"] = round(time.time() - job["started"])
        out["output_dir"] = job.get("output_dir")
    return _dumps(out)


@mcp.tool()
def get_summary(dataset_dir: str) -> str:
    """Counts per class, images, backgrounds, flagged mislabels/outliers/duplicates at the report thresholds,
    and reviewed vs unresolved counts from review_state.json."""
    out_dir = _out_dir(dataset_dir)
    rep = _report(dataset_dir)
    p = _params(rep)
    live = _live(rep)
    st = _load_state(dataset_dir)
    deleted, kept, relabeled = set(st.get("deleted") or []), set(st.get("kept") or []), dict(st.get("relabeled") or {})
    reviewed = deleted | kept | set(relabeled)
    mis = [it for it in live if _is_mislabel(it, p)]
    outl = [it for it in live if _is_outlier(it, p)]
    dups = _dup_pairs(rep, p)
    flagged_ids = {it["id"] for it in mis + outl} | {it["id"] for it in live if it.get("issues")}
    stats = rep.get("stats") or {}
    bg = rep.get("backgrounds")
    out = {
        "output_dir": out_dir, "model": rep.get("model"), "created": rep.get("created"),
        "class_names": rep.get("class_names"),
        "images": stats.get("images"), "boxes": len(live),
        "per_class": dict(Counter(it["label"] for it in live)),
        "background_images": (len(bg) if isinstance(bg, list) else bg) if bg is not None else stats.get("background_images"),
        "thresholds": {k: p[k] for k in ("dup_threshold", "lof_threshold", "mislabel_conf")},
        "mislabels": len(mis), "outliers": len(outl), "duplicate_pairs": len(dups),
        "boxes_with_issues": sum(1 for it in live if it.get("issues")),
        "review": {"deleted": len(deleted), "kept": len(kept), "relabeled": len(relabeled),
                   "ignored_duplicates": len(st.get("ignoredDuplicates") or []),
                   "flagged_reviewed": len(flagged_ids & reviewed),
                   "flagged_unresolved": len(flagged_ids - reviewed)},
    }
    return _dumps(out)


@mcp.tool()
def list_flagged(dataset_dir: str, kind: str = "mislabel", limit: int = 50) -> str:
    """List flagged boxes. kind: mislabel | outlier | duplicate | issue. Returns image, label, pred, conf, crop path."""
    out_dir = _out_dir(dataset_dir)
    rep = _report(dataset_dir)
    p = _params(rep)
    live = _live(rep)
    if kind == "mislabel":
        rows = sorted((it for it in live if _is_mislabel(it, p)), key=lambda it: -(it.get("pred_conf") or 0))
        res = [_brief(it, out_dir) for it in rows]
    elif kind == "outlier":
        rows = sorted((it for it in live if _is_outlier(it, p)), key=lambda it: -(it.get("lof") or 0))
        res = [_brief(it, out_dir) for it in rows]
    elif kind == "duplicate":
        res = [{"similarity": s, "a": _brief(a, out_dir), "b": _brief(b, out_dir), "pair_key": f"{a['id']}|{b['id']}"}
               for a, b, s in _dup_pairs(rep, p)]
    elif kind == "issue":
        res = [dict(_brief(it, out_dir), issues=it["issues"]) for it in live if it.get("issues")]
    else:
        return _dumps({"error": "kind must be mislabel, outlier, duplicate or issue"})
    return _dumps({"kind": kind, "total": len(res), "items": res[:max(0, int(limit))]})


@mcp.tool()
def get_item(dataset_dir: str, item_id: str) -> str:
    """Full details of one box: label, prediction, scores, issues, image/label/crop paths, neighbours, review decision."""
    out_dir = _out_dir(dataset_dir)
    rep = _report(dataset_dir)
    items = rep["items"]
    it = next((x for x in items if x["id"] == item_id), None)
    if it is None:
        return _dumps({"error": f"unknown item {item_id}"})
    d = {k: v for k, v in it.items() if k not in ("nbrs", "pca", "map2d")}
    d["crop"] = os.path.normpath(os.path.join(out_dir, it["path"])) if it.get("path") else None
    d["neighbors"] = [{"id": items[j]["id"], "label": items[j]["label"], "sim": s}
                      for j, s in (it.get("nbrs") or [])[:8] if j < len(items)]
    st = _load_state(dataset_dir)
    d["review"] = ("deleted" if item_id in (st.get("deleted") or []) else
                   "relabeled:" + st["relabeled"][item_id] if item_id in (st.get("relabeled") or {}) else
                   "kept" if item_id in (st.get("kept") or []) else None)
    return _dumps(d)


@mcp.tool()
def decide(dataset_dir: str, item_ids: list[str], action: str, new_label: str | None = None) -> str:
    """Record review decisions in review_state.json (same format as the dashboard; nothing is written to labels).
    action: approve | exclude | relabel (needs new_label). Apply them later with the dashboard's write-back,
    or edit labels directly with relabel_box / remove_box."""
    rep = _report(dataset_dir)
    by_id = {it["id"]: it for it in rep["items"]}
    if action not in ("approve", "exclude", "relabel"):
        return _dumps({"error": "action must be approve, exclude or relabel"})
    if action == "relabel" and new_label not in rep["class_names"]:
        return _dumps({"error": f"new_label must be one of {rep['class_names']}"})
    st = _load_state(dataset_dir)
    deleted = list(st.get("deleted") or [])
    kept = list(st.get("kept") or [])
    relabeled = dict(st.get("relabeled") or {})
    log = list(st.get("log") or [])
    done, unknown = [], []
    for iid in item_ids:
        it = by_id.get(iid)
        if it is None:
            unknown.append(iid)
            continue
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if action == "exclude":
            if iid not in deleted:
                deleted.append(iid)
            kept = [k for k in kept if k != iid]
            act = {"type": "delete", "id": iid, "description": f"Deleted {iid}."}
        elif action == "approve":
            if iid not in kept:
                kept.append(iid)
            act = {"type": "keep", "id": iid, "description": f"Confirmed {iid}."}
        else:
            old = relabeled.get(iid, it["label"])
            if new_label == it["label"]:
                relabeled.pop(iid, None)
            else:
                relabeled[iid] = new_label
            act = {"type": "relabel", "id": iid, "old_label": old, "new_label": new_label,
                   "description": f"Relabeled {iid}: '{old}' \u2192 '{new_label}'."}
        act["time"] = now
        log = [a for a in log if a.get("id") != iid]  # latest decision per crop wins (registerAction)
        log.append(act)
        done.append(iid)
    st.update({"deleted": deleted, "relabeled": relabeled, "kept": kept,
               "ignoredDuplicates": list(st.get("ignoredDuplicates") or []), "log": log})
    st.setdefault("thresholds", {"dup": _params(rep)["dup_threshold"], "lof": _params(rep)["lof_threshold"],
                                 "mis": _params(rep)["mislabel_conf"]})
    _save_state(dataset_dir, st)
    return _dumps({"updated": len(done), "unknown": unknown, "state_file": _state_path(dataset_dir),
                   "note": "If the dashboard is open, reload it so it does not overwrite these decisions."})


@mcp.tool()
def relabel_box(dataset_dir: str, item_id: str, new_label: str) -> str:
    """Change the class of one box directly in its YOLO label file (timestamped backup in label_backups)."""
    names = _report(dataset_dir)["class_names"]
    if new_label not in names:
        return _dumps({"error": f"new_label must be one of {names}"})
    return _dumps(_edit_box(dataset_dir, item_id, names.index(new_label)))


@mcp.tool()
def remove_box(dataset_dir: str, item_id: str) -> str:
    """Delete one box from its YOLO label file (timestamped backup in label_backups)."""
    return _dumps(_edit_box(dataset_dir, item_id, None))


@mcp.tool()
def export_for_labeling(dataset_dir: str) -> str:
    """Copy flagged images + labels (with X-AnyLabeling JSON) into <output>/for_labeling."""
    r = _run_cli([os.path.abspath(dataset_dir), "--export-labeling"])
    r["folder"] = os.path.join(_out_dir(dataset_dir), "for_labeling")
    return _dumps(r)


@mcp.tool()
def sync_labels(dataset_dir: str) -> str:
    """Copy labels edited in <output>/for_labeling back into the dataset (with backups)."""
    return _dumps(_run_cli([os.path.abspath(dataset_dir), "--sync-labels"]))


@mcp.tool()
def get_distribution(dataset_dir: str) -> str:
    """Class distribution: report['distribution'] if the analysis produced it, else per-class / per-split counts."""
    rep = _report(dataset_dir)
    if rep.get("distribution") is not None:
        return _dumps(rep["distribution"])
    live = _live(rep)
    per_img = Counter(it["image_path"] for it in live)
    by_split = {}
    for it in live:
        by_split.setdefault(it.get("split"), Counter())[it["label"]] += 1
    hist = Counter(per_img.values())
    return _dumps({"per_class": dict(Counter(it["label"] for it in live)),
                   "per_split": {k: dict(v) for k, v in by_split.items()},
                   "images_with_boxes": len(per_img),
                   "background_images": (rep.get("stats") or {}).get("background_images"),
                   "boxes_per_image_histogram": {str(k): hist[k] for k in sorted(hist)}})


@mcp.tool()
def open_dashboard(dataset_dir: str, port: int = 8000) -> str:
    """Start the review dashboard (serve-only) in the background and return its URL."""
    _report(dataset_dir)
    url = f"http://localhost:{port}/"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return _dumps({"url": url, "note": f"port {port} already in use; a dashboard may already be running"})
    os.makedirs(JOBS_DIR, exist_ok=True)
    log = os.path.join(JOBS_DIR, f"dashboard_{port}.log")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([sys.executable, CLI, os.path.abspath(dataset_dir), "--serve-only", "--no-browser",
                             "--port", str(port)], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=SCRIPT_DIR, creationflags=flags)
    time.sleep(2)
    if proc.poll() is not None:
        return _dumps({"error": "dashboard exited", "log": open(log, encoding="utf-8", errors="replace").read()[-1500:]})
    return _dumps({"url": url, "pid": proc.pid, "log": log})


if __name__ == "__main__":
    mcp.run()
