from pathlib import Path

from .routing import analyze
from .shared import hash_file, read, save, signature


def figure(path, curves):
    colors = {"gain":"#00644f","difficulty":"#245ca6","entropy":"#894d00","random":"#626779","oracle":"#674090"}
    levels = [r["patient_dice"] for points in curves.values() for r in points]
    low, high = max(0,min(levels)-.01), min(1,max(levels)+.01)
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="380" viewBox="0 0 700 380">',
                '<rect width="700" height="380" fill="white"/>',
                '<text x="55" y="28">Patient-macro lesion Dice vs. retry budget</text>']
    for step in range(5):
        height = 300-60*step
        elements.append(f'<path d="M70 {height}H625" stroke="#ddd"/><text x="7" y="{height}">{low+(high-low)*step/4:.3f}</text>')
    for index,(name,points) in enumerate(curves.items()):
        coords = " ".join(f'{70+550*r["budget"]:.1f},{300-240*(r["patient_dice"]-low)/(high-low):.1f}' for r in points)
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
        elements.append(f'<text x="{70+115*index}" y="357" fill="{colors[name]}">{name}</text>')
    elements.extend(['<text x="65" y="325">0%</text><text x="171" y="325">20%</text><text x="590" y="325">100% retried</text>', '</svg>'])
    path.write_text("\n".join(elements))


def report(run):
    run = Path(run)
    pilot = read(run/"pilot"/"gate.json")
    out = run/"report"
    out.mkdir(exist_ok=True)
    title = "Synthetic engineering demonstration — not study evidence" if pilot["demo"] else "Recoverable errors in breast ultrasound segmentation"
    lines = ["# "+title,"","## Pilot", "",
             f'Oracle advantage over entropy: {pilot["oracle_minus_entropy"]:+.4f} absolute Dice. Gate passed: {pilot["passed"]}.',
             f'First-pass, entropy and oracle Dice: {pilot["scores"]["never"]:.4f}, {pilot["scores"]["entropy"]:.4f}, {pilot["scores"]["oracle"]:.4f}.',
             f'Refining every positive image helped {pilot["improved_fraction"]:.1%} and harmed {pilot["harmed_fraction"]:.1%}.']
    if not (run/"evaluation"/"complete.json").exists():
        lines.append("External cohorts not evaluated. This is a pilot-only report.")
    else:
        marker = read(run/"evaluation"/"complete.json")
        frozen = read(run/"freeze.json")
        if marker["freeze"] != signature(frozen):
            raise ValueError("Frozen evaluation signature mismatch")
        for name in ("routing.py","report.py"):
            if hash_file(Path(__file__).with_name(name)) != frozen["source"][name]:
                raise ValueError("Analysis code differs from the frozen version")
        threshold = read(run/"router.json")["threshold"]
        if hash_file(run/"router.json") != read(run/"freeze.json")["files"]["router.json"]:
            raise ValueError("Frozen router changed")
        lines += ["","## Endpoint", "", "Paired patient-bootstrap 95% intervals, conditional on fixed retry decisions. BrEaST is primary; other cohorts are not pooled.",
                  "", "| Cohort | Positive patients | Gain − entropy Dice | 95% interval |", "|---|---:|---:|---|"]
        details = []
        results = {}
        for cohort in ("breast","uclm","internal"):
            folder = run/"evaluation"/cohort
            if hash_file(folder/"predictions.json") != marker["predictions"][cohort]:
                raise ValueError("Saved evaluation predictions changed")
            if hash_file(folder/"timing.json") != marker["timing"][cohort]:
                raise ValueError("Saved timing changed")
            metrics = analyze(read(folder/"predictions.json"), threshold, 2000)
            metrics["stream_timing"] = read(folder/"timing.json")
            results[cohort] = metrics
            low, high = metrics["interval"]
            lines.append(f'| {cohort} | {metrics["positive_patients"]} | {metrics["delta"]:+.4f} | [{low:+.4f}, {high:+.4f}] |')
            details += ["",f"### {cohort}", "", "| Policy | Retries / scans | Patient Dice | Missed positives | Normal FP rate | Replay ms/image |",
                        "|---|---:|---:|---:|---:|---:|"]
            for name, rec in metrics["policies"].items():
                fpr = "n/a" if rec["normal_fp_rate"] is None else f'{rec["normal_fp_rate"]:.3f}'
                details.append(f'| {name} | {rec["retries"]}/{rec["scans"]} | {rec["patient_dice"]:.4f} | {rec["missed_positive"]} | {fpr} | {rec["replay_mean_ms"]:.2f} |')
            details.append(f'\n![Quality versus retry budget]({cohort}.svg)')
            secondary = metrics["gain_vs_difficulty"]
            lower, upper = secondary["interval"]
            details.append(f'\nSecondary comparison, gain minus difficulty: {secondary["delta"]:+.4f} Dice '
                           f'(paired 95% interval [{lower:+.4f}, {upper:+.4f}]). '
                           f'Both retried {secondary["common"]} of the same scans; '
                           f'{secondary["gain_only"]} gain-only and {secondary["difficulty_only"]} difficulty-only.')
            timing = metrics["stream_timing"]
            details.append(f'\nMeasured streaming policy, including decoding: {timing["retries"]}/{timing["count"]} retries; median {timing["p50_ms"]:.2f} ms, p95 {timing["p95_ms"]:.2f} ms.')
            figure(out/f"{cohort}.svg", metrics["curves"])
        if not pilot["demo"]:
            conclusion = ("The primary comparison supports superiority at the prespecified budget."
                          if results["breast"]["interval"][0] > 0 else
                          "The primary comparison does not establish superiority at the prespecified budget.")
            lines += ["",conclusion]
        lines += details
        save(out/"metrics.json", results)
    lines += ["","## Interpretation", "", "Oracle uses reference masks and cannot be deployed. Replay costs sum measured stages; threshold timing in metrics.json is a separate batch-one measurement including image decoding. Missing normal denominators are not zero. No diagnostic or clinical utility claim."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    return out/"REPORT.md"
