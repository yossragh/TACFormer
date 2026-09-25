from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from utils import ensure_dir, read_jsonl, write_json


def collect_unique_pois_from_jsonls(
    input_jsonls: list[str | Path],
    target_uniprot_field: str,
    poi_field: str,
) -> list[dict[str, str]]:
    poi_by_uniprot: dict[str, str] = {}

    for jsonl_path in input_jsonls:
        for record in read_jsonl(jsonl_path):
            target_uniprot = record[target_uniprot_field]
            poi_norm = record[poi_field]

            existing_poi = poi_by_uniprot.get(
                target_uniprot
            )

            if (
                existing_poi is not None
                and existing_poi != poi_norm
            ):
                raise ValueError(
                    f"{target_uniprot} maps to both "
                    f"{existing_poi!r} and {poi_norm!r}."
                )

            poi_by_uniprot[
                target_uniprot
            ] = poi_norm

    return [
        {
            "target_uniprot": target_uniprot,
            "poi_norm": poi_norm,
        }
        for target_uniprot, poi_norm in sorted(
            poi_by_uniprot.items()
        )
    ]


def _request_json(
    url: str,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "TACFormer-poi-metadata/1.0",
            },
        )

        try:
            with urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                return json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except HTTPError as exc:
            last_error = exc

            if exc.code == 404:
                raise

            if exc.code not in {
                408,
                429,
                500,
                502,
                503,
                504,
            }:
                raise

        except URLError as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(
                sleep_seconds * attempt
            )

    assert last_error is not None
    raise last_error


def fetch_uniprot_metadata(
    target_uniprot: str,
    base_url: str,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    url = (
        f"{base_url.rstrip('/')}"
        f"/uniprotkb/{target_uniprot}.json"
    )

    try:
        entry = _request_json(
            url=url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )

    except HTTPError as exc:
        if exc.code == 404:
            return {
                "target_uniprot": target_uniprot,
                "fetch_status": "not_found",
                "entry_name": None,
                "reviewed": None,
                "organism_name": None,
                "organism_id": None,
                "sequence": None,
                "sequence_length": None,
            }

        raise

    sequence = entry.get(
        "sequence",
        {},
    ).get(
        "value"
    )

    organism = entry.get(
        "organism",
        {},
    )

    entry_type = entry.get(
        "entryType"
    )

    reviewed = (
        entry_type
        == "UniProtKB reviewed (Swiss-Prot)"
        if entry_type is not None
        else None
    )

    return {
        "target_uniprot": target_uniprot,
        "fetch_status": "ok",
        "entry_name": entry.get(
            "uniProtkbId"
        ),
        "reviewed": reviewed,
        "organism_name": organism.get(
            "scientificName"
        ),
        "organism_id": organism.get(
            "taxonId"
        ),
        "sequence": sequence,
        "sequence_length": (
            len(sequence)
            if sequence is not None
            else None
        ),
    }


def build_poi_metadata(
    input_jsonls: list[str | Path],
    output_csv: str | Path,
    stats_json: str | Path,
    target_uniprot_field: str,
    poi_field: str,
    base_url: str,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
    fail_on_missing_sequence: bool,
) -> dict[str, Any]:
    poi_rows = collect_unique_pois_from_jsonls(
        input_jsonls=input_jsonls,
        target_uniprot_field=target_uniprot_field,
        poi_field=poi_field,
    )

    metadata_rows: list[
        dict[str, Any]
    ] = []

    for index, poi_row in enumerate(
        poi_rows,
        start=1,
    ):
        target_uniprot = poi_row[
            "target_uniprot"
        ]

        print(
            f"[INFO] Fetching UniProt "
            f"{index}/{len(poi_rows)}: "
            f"{target_uniprot} "
            f"({poi_row['poi_norm']})"
        )

        uniprot_metadata = fetch_uniprot_metadata(
            target_uniprot=target_uniprot,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )

        metadata_rows.append(
            {
                "target_uniprot": (
                    target_uniprot
                ),
                "poi_norm": (
                    poi_row["poi_norm"]
                ),
                "fetch_status": (
                    uniprot_metadata[
                        "fetch_status"
                    ]
                ),
                "entry_name": (
                    uniprot_metadata[
                        "entry_name"
                    ]
                ),
                "reviewed": (
                    uniprot_metadata[
                        "reviewed"
                    ]
                ),
                "organism_name": (
                    uniprot_metadata[
                        "organism_name"
                    ]
                ),
                "organism_id": (
                    uniprot_metadata[
                        "organism_id"
                    ]
                ),
                "sequence": (
                    uniprot_metadata[
                        "sequence"
                    ]
                ),
                "sequence_length": (
                    uniprot_metadata[
                        "sequence_length"
                    ]
                ),
            }
        )

        if index < len(poi_rows):
            time.sleep(
                sleep_seconds
            )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    output_csv = Path(
        output_csv
    )

    stats_json = Path(
        stats_json
    )

    ensure_dir(
        output_csv.parent
    )

    metadata_df.to_csv(
        output_csv,
        index=False,
    )

    missing_sequence_df = metadata_df[
        metadata_df[
            "sequence"
        ].isna()
    ]

    sequence_lengths = metadata_df[
        "sequence_length"
    ].dropna()

    stats = {
        "input_jsonls": [
            str(path)
            for path in input_jsonls
        ],
        "output_csv": str(
            output_csv
        ),
        "stats_json": str(
            stats_json
        ),
        "num_unique_uniprots": int(
            len(metadata_df)
        ),
        "num_unique_pois": int(
            metadata_df[
                "poi_norm"
            ].nunique()
        ),
        "num_ok": int(
            (
                metadata_df[
                    "fetch_status"
                ]
                == "ok"
            ).sum()
        ),
        "num_not_found": int(
            (
                metadata_df[
                    "fetch_status"
                ]
                == "not_found"
            ).sum()
        ),
        "num_missing_sequence": int(
            len(
                missing_sequence_df
            )
        ),
        "missing_sequence_uniprots": (
            missing_sequence_df[
                "target_uniprot"
            ].tolist()
        ),
        "min_sequence_length": (
            int(
                sequence_lengths.min()
            )
            if not sequence_lengths.empty
            else None
        ),
        "median_sequence_length": (
            float(
                sequence_lengths.median()
            )
            if not sequence_lengths.empty
            else None
        ),
        "max_sequence_length": (
            int(
                sequence_lengths.max()
            )
            if not sequence_lengths.empty
            else None
        ),
    }

    write_json(
        stats,
        stats_json,
    )

    if (
        fail_on_missing_sequence
        and len(
            missing_sequence_df
        ) > 0
    ):
        raise ValueError(
            "Missing UniProt sequences for "
            f"{len(missing_sequence_df)} targets: "
            f"{missing_sequence_df['target_uniprot'].tolist()}"
        )

    return stats