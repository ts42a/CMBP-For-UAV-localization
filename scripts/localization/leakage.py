"""Development vs held-out region guards."""
from __future__ import annotations

ALL_REGION_IDS = {f"{i:02d}" for i in range(1, 12)}


def dev_regions_from_config(cfg: dict) -> set[str]:
    return set(cfg["leakage_and_splits"]["matcher_tuning_regions"])


def held_out_regions(cfg: dict) -> set[str]:
    return ALL_REGION_IDS - dev_regions_from_config(cfg)


def assert_dev_only(region: str, cfg: dict, context: str = "") -> None:
    allowed = dev_regions_from_config(cfg)
    if region not in allowed:
        raise RuntimeError(
            f"Leakage guard: region {region} is not in matcher_tuning_regions {sorted(allowed)}. "
            f"Held-out geography must not be used for matcher development. {context}"
        )


def assert_dev_query_list(regions: set[str], cfg: dict) -> None:
    for r in regions:
        assert_dev_only(r, cfg, "query batch")
