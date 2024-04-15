import collections as c
import typing as t
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import TypeAlias

Record: TypeAlias = t.Mapping[str, t.Any]

MODEL_CODE_TO_NAME: dict[int, str] = {
    0: "DebugNet",
    1: "KyleNet",
    3: "SqueezeNet",
    18: "ResNet-18",
    50: "ResNet-50",
    152: "ResNet-152",
}

MODEL_NAME_TO_CODE: dict[str, int] = {
    name: code
    for code, name in MODEL_CODE_TO_NAME.items()
}

###################################################


def _filter_parsl_results(
    data_dirs: Path | t.Iterable[Path]
) -> t.Iterator[Path]:
    if not isinstance(data_dirs, c.abc.Iterable):
        data_dirs = [data_dirs]

    mask = "❯ Finished in"
    for d in data_dirs:
        for file in d.glob("*.stdout"):
            with open(file, "r") as f:
                text = f.read()
                if mask in text:
                    yield file


def _parsel_parsl_result_to_record(
    file: Path,
    include_start: bool = True,
    include_dfk: bool = False,
    include_priming: bool = True,
) -> Record:
    record = {}

    # Infer the experiment setup from the filename.
    stem = file.stem
    stem = stem.split("_")[-1]
    model_num, num_workers = stem.split(".")
    model_num, num_workers = int(model_num), int(num_workers)
    record["model"] = MODEL_CODE_TO_NAME[model_num]
    record["num_workers"] = num_workers

    # Extract the results from the logged stdout.
    with open(file, "r") as f:
        for line in f.readlines():
            if include_start and line.startswith("start:"):
                value = line.split(":")[-1]
                record["start"] = float(value)

            elif include_dfk and line.startswith("dfk_start_done:"):
                value = line.split(":")[-1]
                record["dfk_start_done"] = float(value)

            elif include_priming and line.startswith("priming_done:"):
                value = line.split(":")[-1]
                record["priming_done"] = float(value)

            elif line.startswith("end:"):
                value = line.split(":")[-1]
                record["end"] = float(value)

    return record


def parse_parsl_results(
    data_dirs: Path | t.Iterable[Path],
    include_start: bool = True,
    include_dfk: bool = False,
    include_priming: bool = True,
) -> pd.DataFrame:
    if not isinstance(data_dirs, c.abc.Iterable):
        data_dirs = [data_dirs]

    records: list[Record] = [
        _parsel_parsl_result_to_record(f, include_start, include_dfk, include_priming)
        for d in data_dirs
        for f in _filter_parsl_results(d)
    ]

    df = pd.DataFrame.from_records(records)

    if include_start:
        df["start2end"] = df.end - df.start
    if include_dfk:
        df["dfk2end"] = df.end - df.dfk_start_done
    if include_priming:
        df["prime2end"] = df.end - df.priming_done

    return df


###################################################

@dataclass
class FlwrTime:
    start: float = field(default=None)
    end: float = field(default=None)


# NOTE: We need to divide by 1e9 because the times are recorded in nanoseconds.
FLWR_TIME_NORMALIZER = 1e9


def _flower_params_from_path(path: Path) -> dict[str, t.Any]:
    if "single_node" in str(path):
        # pattern: 'flower_time_<model>_<workers>_*.txt'
        kind = "single_node"
        stem = path.stem
        parts = stem.split("_")
        model_code = int(parts[2])
        workers = int(parts[3])
        is_server = "exp" not in stem

    elif "multi_node" in str(path):
        # pattern: 'flower_time_<model>_<workers-per-node>_<nodes>_*.txt'
        kind = "multi_node"
        stem = path.stem
        is_server = "exp" not in stem
        parts = stem.split("_")

        if len(parts) == 4:
            is_server = True
            model_code = int(parts[2])
            workers = int(parts[3])

        elif len(parts) == 6:
            is_server = False
            model_code = int(parts[2])
            workers_per_node = int(parts[3])
            num_nodes = int(parts[4])
            assert workers_per_node == 128
            workers = workers_per_node * num_nodes

        else:
            raise ValueError

    else:
        raise ValueError("Illegal type of path.")

    return {
        "kind": kind,
        "model": MODEL_CODE_TO_NAME[model_code],
        "workers": workers,
        "is_server": is_server,
    }

def _server_result_mask(filename: str) -> bool:
    return "exp-" not in filename


def parse_flower_results(data_dirs: t.Iterable[Path] | Path):
    if not isinstance(data_dirs, c.abc.Iterable):
        data_dirs = [data_dirs]

    records = []
    times = c.defaultdict(lambda: dict())
    for d in data_dirs:
        for filename in d.glob("*.txt"):
            _time = FlwrTime()
            params = _flower_params_from_path(filename)
            with open(filename, "r") as f:
                lines = f.readlines()
                assert len(lines) <= 2

                value = lines[0]
                value = float(value)
                value /= FLWR_TIME_NORMALIZER

                key = params["model"], params["workers"]
                if params["is_server"]:
                    times[key]["end"] = value
                elif "start" not in times[key]:
                    times[key]["start"] = [value]
                else:
                    times[key]["start"].append(value)

    for (model, workers) in times:
        if "start" not in times[model, workers]:
            continue
        value = times[model, workers]["start"]
        times[model, workers]["start"] = min(value)

    for (model, workers), t in times.items():
        end, start = t.get("end"), t.get("start")
        if end is None or start is None:
            continue
        makespan = end - start
        records.append({"model": model, "workers": workers, "time": makespan})

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    from pathlib import Path

    df = parse_flower_results([
        Path("flower_results/multi_node/"),
    ])
    print(df.query("workers > 128").head())
    