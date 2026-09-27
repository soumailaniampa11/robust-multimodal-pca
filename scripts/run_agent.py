"""Run the decision-support agent on one case or on all held-out demo cases.

Examples:
    python scripts/run_agent.py --demo                  # all held-out demo cases, deterministic agent
    python scripts/run_agent.py --case 10636_1000652    # one case
    python scripts/run_agent.py --case 10636_1000652 --drop radiology   # simulate a missing modality
    python scripts/run_agent.py --case 10636_1000652 --llm              # Claude-orchestrated (needs API key)

Reports (Markdown) and full tool traces (JSON) are written to outputs/reports/.
The reference-standard label is never given to the agent; it is only printed
afterwards for the demo evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config  # noqa: E402
from pcafusion.agent import DecisionSupportAgent, ToolBox  # noqa: E402
from pcafusion.agent.tools import dumps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case")
    g.add_argument("--demo", action="store_true")
    ap.add_argument("--llm", action="store_true", help="let Claude orchestrate the tools")
    ap.add_argument("--drop", nargs="*", default=[], choices=list(config.MODALITIES),
                    help="simulate unavailable modalities for this run")
    args = ap.parse_args()

    tools = ToolBox()
    for m in args.drop:  # simulate missing modalities by blanking them in the case record
        tools.df.loc[:, config.MODALITIES[m]] = np.nan
    if args.llm:
        from pcafusion.agent.llm_agent import LLMDecisionSupportAgent
        agent = LLMDecisionSupportAgent(tools)
    else:
        agent = DecisionSupportAgent(tools)

    cases = tools.demo_ids if args.demo else [args.case]
    if args.case and args.case not in tools.demo_ids:
        print(f"note: {args.case} is not a held-out demo case; the model may have seen it during training.\n")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_llm" if args.llm else "") + ("_no-" + "-".join(args.drop) if args.drop else "")
    rows = []
    for cid in cases:
        res = agent.run(cid)
        (config.REPORTS_DIR / f"{cid}{suffix}.md").write_text(res.report_md)
        (config.REPORTS_DIR / f"{cid}{suffix}_trace.json").write_text(
            json.dumps({"case_id": cid, "flag": res.flag, "citation_check": res.citation_check,
                        "trace": json.loads(dumps(res.trace))}, indent=2))
        truth = int(tools.df.loc[cid, config.TARGET])
        rows.append({"case_id": cid, "flag": res.flag, "tool_calls": len(res.trace),
                     "citations_ok": res.citation_check["passed"], "reference_cspca": truth})
        if not args.demo:
            print(res.report_md)
            print(f"\n[tool calls: {len(res.trace)} | citation check: {res.citation_check}]")
            print(f"[reference standard, hidden from the agent: {'csPCa' if truth else 'no csPCa'}]")

    if args.demo:
        df = pd.DataFrame(rows)
        tab = pd.crosstab(df.flag, df.reference_cspca.map({0: "no csPCa", 1: "csPCa"}), margins=True)
        print(tab.to_string())
        print(f"\nAll citation checks passed: {df.citations_ok.all()}")
        tab.to_csv(config.RESULTS_DIR / f"agent_demo_flags{suffix}.csv")
    print(f"\nReports written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
