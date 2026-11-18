import argparse
import json
from pathlib import Path

from .data import breast, busbra, prepare, uclm
from .snapshot import create as snapshot
from .demo import synthetic
from .report import report
from .shared import read
from .study import Study


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recoverable ultrasound mask errors: research workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare","profile","train","pilot","crossfit","fit-router","freeze","evaluate","report","demo"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name=="prepare":
            command.add_argument("--busbra",type=Path,required=True)
            command.add_argument("--breast",type=Path)
            command.add_argument("--breast-sheet",type=Path)
            command.add_argument("--uclm-images",type=Path)
            command.add_argument("--uclm-masks",type=Path)
            command.add_argument("--uclm-csv",type=Path)
            command.add_argument("--uclm-map",type=Path)
            command.add_argument("--decisions",type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command == "prepare":
        rows, excluded = busbra(args.busbra), []
        if bool(args.breast) != bool(args.breast_sheet):
            parser.error("BrEaST needs both --breast and --breast-sheet")
        if args.breast:
            rows.extend(breast(args.breast,args.breast_sheet))
        uclm_args = [args.uclm_images,args.uclm_masks,args.uclm_csv]
        if any(uclm_args) and not all(uclm_args):
            parser.error("BUS-UCLM needs --uclm-images, --uclm-masks and --uclm-csv")
        if all(uclm_args):
            external, excluded = uclm(*uclm_args, read(args.uclm_map) if args.uclm_map else None)
            rows.extend(external)
        prepare(rows,run,excluded,read(args.decisions) if args.decisions else {})
        print(f"Prepared {run/'manifest.json'}")
    elif args.command == "demo":
        if run.exists() and any(run.iterdir()):
            parser.error("Demo requires a fresh output directory")
        prepare(synthetic(run/"synthetic"),run,demo=True)
        snapshot(run)
        study = Study(run)
        for step in (study.train,study.pilot,study.crossfit,study.router,study.freeze,study.evaluate):
            step()
        print(report(run))
    elif args.command == "report":
        print(report(run))
    else:
        study = Study(run)
        action = study.router if args.command=="fit-router" else getattr(study,args.command)
        result = action()
        print(json.dumps(result,indent=2) if result is not None else f"Completed {args.command}")


if __name__ == "__main__":
    main()
