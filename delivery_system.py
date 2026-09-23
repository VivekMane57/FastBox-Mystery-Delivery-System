"""
FastBox Mystery Delivery System
--------------------------------
Simulates one day of delivery operations for a fictional logistics
company (FastBox). Reads warehouse / agent / package data from a JSON
file, assigns each package to the nearest available agent, simulates
the pickups + deliveries, and writes a report.json with per-agent
stats and the best (most efficient) agent.

Run:
    python delivery_system.py data.json
    python delivery_system.py data.json --report report.json --ascii --csv

Tests:
    pytest test_delivery_system.py -v

Author: Vivek
"""

from __future__ import annotations

import json
import csv
import math
import random
import argparse
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

Point = Tuple[float, float]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fastbox")


# ---------------------------------------------------------------------------
# Custom exceptions -- so callers (and tests) can catch specific failure
# modes instead of a bare Exception / raw traceback.
# ---------------------------------------------------------------------------

class DeliveryDataError(ValueError):
    """Raised when the input JSON is missing required fields or malformed."""


# ---------------------------------------------------------------------------
# 1. JSON PARSING
# ---------------------------------------------------------------------------
# ASSUMPTION (documented, since the assignment PDF example and the actual
# test_case_*.json files use two DIFFERENT schemas):
#   Schema A (PDF example / base_case.json):
#       "warehouses": {"W1": [0, 0], ...}                 (dict-style)
#       or [{"id": "W1", "location": [0, 0]}, ...]         (list-style)
#       "packages": [{"id": "P1", "warehouse_id": "W1", "destination": [x,y]}]
#   Schema B (every provided test_case file):
#       "warehouses": {"W1": [x, y], ...}
#       "agents": {"A1": [x, y], ...}
#       "packages": [{"id": "P1", "warehouse": "W1", "destination": [x, y]}]
#
# Rather than assuming one fixed shape, the loader below normalizes BOTH
# styles into simple dicts, and validates the result so bad input fails
# fast with a clear message instead of a raw traceback deep in the sim.
# ---------------------------------------------------------------------------

def _normalize_locations(raw: Any, field_name: str) -> "OrderedDict[str, Point]":
    """
    Accepts either:
        {"W1": [x, y], "W2": [x, y], ...}
    or
        [{"id": "W1", "location": [x, y]}, ...]
    Returns an OrderedDict: {id: (x, y)} preserving input order
    (order matters later for deterministic tie-breaking).
    Raises DeliveryDataError on malformed entries.
    """
    result: "OrderedDict[str, Point]" = OrderedDict()

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        try:
            items = [(entry["id"], entry["location"]) for entry in raw]
        except (KeyError, TypeError) as exc:
            raise DeliveryDataError(
                f"'{field_name}' list entries must have 'id' and 'location' keys"
            ) from exc
    else:
        raise DeliveryDataError(
            f"'{field_name}' must be a dict or a list, got {type(raw).__name__}"
        )

    for entry_id, coords in items:
        try:
            x, y = coords
            result[entry_id] = (float(x), float(y))
        except (TypeError, ValueError) as exc:
            raise DeliveryDataError(
                f"'{field_name}' entry '{entry_id}' has invalid coordinates: {coords!r}"
            ) from exc

    return result


def load_data(path: str) -> Dict[str, Any]:
    """
    Reads and parses the input JSON file, then normalizes it into a
    predictable internal structure:
        {
            "warehouses": {id: (x, y), ...},
            "agents":     {id: (x, y), ...},
            "packages":   [{"id": .., "warehouse": .., "destination": (x, y)}, ...]
        }
    Raises:
        FileNotFoundError    -- if `path` doesn't exist (clear message, not a traceback)
        DeliveryDataError    -- if the JSON is valid but the schema is wrong/incomplete
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeliveryDataError(f"'{path}' is not valid JSON: {exc}") from exc

    for required in ("warehouses", "agents", "packages"):
        if required not in raw:
            raise DeliveryDataError(f"Input JSON is missing required key: '{required}'")

    warehouses = _normalize_locations(raw["warehouses"], "warehouses")
    agents = _normalize_locations(raw["agents"], "agents")

    if not isinstance(raw["packages"], list):
        raise DeliveryDataError("'packages' must be a list")

    packages: List[Dict[str, Any]] = []
    seen_ids = set()
    for i, pkg in enumerate(raw["packages"]):
        pkg_id = pkg.get("id")
        if not pkg_id:
            raise DeliveryDataError(f"Package at index {i} is missing an 'id'")
        if pkg_id in seen_ids:
            raise DeliveryDataError(f"Duplicate package id: '{pkg_id}'")
        seen_ids.add(pkg_id)

        # Support both "warehouse" and "warehouse_id" keys.
        wh_id = pkg.get("warehouse", pkg.get("warehouse_id"))
        if wh_id is None:
            raise DeliveryDataError(f"Package '{pkg_id}' has no warehouse reference")
        if wh_id not in warehouses:
            raise DeliveryDataError(f"Package '{pkg_id}' references unknown warehouse '{wh_id}'")

        dest = pkg.get("destination")
        try:
            dest_point: Point = (float(dest[0]), float(dest[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise DeliveryDataError(f"Package '{pkg_id}' has invalid destination: {dest!r}") from exc

        packages.append({"id": pkg_id, "warehouse": wh_id, "destination": dest_point})

    logger.info(
        "Loaded %s: %d warehouses, %d agents, %d packages",
        path, len(warehouses), len(agents), len(packages),
    )
    return {"warehouses": warehouses, "agents": agents, "packages": packages}


# ---------------------------------------------------------------------------
# 2. DISTANCE
# ---------------------------------------------------------------------------

def euclidean(p1: Point, p2: Point) -> float:
    """Straight-line distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# 3. AGENT <-> PACKAGE ASSIGNMENT
# ---------------------------------------------------------------------------
# ASSUMPTION: "nearest agent" is measured from each agent's STARTING
# location to the package's WAREHOUSE (as literally described in the
# assignment: "distance from agent to warehouse"). Agents are not removed
# from the pool after being assigned a package -- a single agent can be
# responsible for several packages (this matches the sample report, where
# A1 and A2 each deliver 2 packages).
#
# TIE-BREAK ASSUMPTION: if two or more agents are exactly equidistant from
# a warehouse, the agent that appears EARLIEST in the input "agents" data
# wins. This keeps the assignment deterministic and reproducible.
# ---------------------------------------------------------------------------

def assign_packages(data: Dict[str, Any]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """
    Returns an OrderedDict: {agent_id: [package, package, ...]}
    Packages keep the order they appeared in the input file, which also
    becomes the pickup/delivery order used during simulation.
    """
    agents = data["agents"]
    warehouses = data["warehouses"]
    assignment: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict(
        (agent_id, []) for agent_id in agents
    )

    if not agents:
        logger.warning("No agents available -- no packages can be assigned")
        return assignment

    agent_order = list(agents.keys())
    for pkg in data["packages"]:
        wh_loc = warehouses[pkg["warehouse"]]
        nearest_agent = min(
            agent_order,
            key=lambda aid: (euclidean(agents[aid], wh_loc), agent_order.index(aid)),
        )
        assignment[nearest_agent].append(pkg)

    return assignment


# ---------------------------------------------------------------------------
# 4. SIMULATION
# ---------------------------------------------------------------------------
# ASSUMPTION: each agent starts at their own starting location. For every
# package assigned to them (processed in the order above), the agent:
#   1. Travels from their CURRENT position to the package's warehouse
#      (pickup leg).
#   2. Travels from the warehouse to the package's destination
#      (delivery leg).
#   3. Their "current position" becomes that destination, ready for the
#      next pickup -- agents do NOT teleport back to their starting point
#      between deliveries, which models "one continuous day of operations".
# ---------------------------------------------------------------------------

def simulate(
    data: Dict[str, Any],
    assignment: "OrderedDict[str, List[Dict[str, Any]]]",
    delays: bool = False,
    seed: int = 42,
) -> "OrderedDict[str, Dict[str, Any]]":
    """
    Runs the simulation and returns:
        {agent_id: {"packages_delivered", "total_distance", "efficiency", "route", [+"delays"]}}
    ("route" is a list of (label, point) tuples, used by the ASCII bonus.)
    """
    warehouses = data["warehouses"]
    rng = random.Random(seed)
    results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for agent_id, packages in assignment.items():
        current_pos = data["agents"][agent_id]
        route: List[Tuple[str, Point]] = [("start", current_pos)]
        total_distance = 0.0
        delivered = 0
        delay_log: List[Dict[str, Any]] = []

        for pkg in packages:
            wh_loc = warehouses[pkg["warehouse"]]
            dest_loc: Point = pkg["destination"]

            total_distance += euclidean(current_pos, wh_loc)
            route.append((f"pickup {pkg['id']} @ {pkg['warehouse']}", wh_loc))

            total_distance += euclidean(wh_loc, dest_loc)
            route.append((f"deliver {pkg['id']}", dest_loc))

            # BONUS: random delivery delay (minutes). Informational only --
            # does NOT affect distance/efficiency, only the report log.
            if delays:
                delay_min = rng.choice([0, 0, 0, 5, 10, 15])
                if delay_min:
                    delay_log.append({"package": pkg["id"], "delay_minutes": delay_min})

            current_pos = dest_loc
            delivered += 1

        efficiency = round(total_distance / delivered, 2) if delivered else 0.0

        results[agent_id] = {
            "packages_delivered": delivered,
            "total_distance": round(total_distance, 2),
            "efficiency": efficiency,
            "route": route,
        }
        if delays:
            results[agent_id]["delays"] = delay_log

    return results


# ---------------------------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------------------------

def build_report(results: "OrderedDict[str, Dict[str, Any]]") -> Dict[str, Any]:
    """
    Builds the final report dict, e.g.:
        {"A1": {"packages_delivered": 2, "total_distance": 85.32,
                 "efficiency": 42.66}, ..., "best_agent": "A1"}
    "best_agent" = the agent with the LOWEST efficiency (lowest average
    distance travelled per package) among agents who delivered at least
    one package. Agents who delivered nothing are still listed (0s) for
    transparency.
    """
    report: Dict[str, Any] = {}
    best_agent: Optional[str] = None
    best_efficiency: Optional[float] = None

    for agent_id, r in results.items():
        report[agent_id] = {
            "packages_delivered": r["packages_delivered"],
            "total_distance": r["total_distance"],
            "efficiency": r["efficiency"],
        }
        if r["packages_delivered"] > 0:
            if best_efficiency is None or r["efficiency"] < best_efficiency:
                best_efficiency = r["efficiency"]
                best_agent = agent_id

    report["best_agent"] = best_agent
    return report


def sanity_check(data: Dict[str, Any], report: Dict[str, Any]) -> bool:
    """Bonus safety net requested in the notes: total delivered == total packages."""
    total_delivered = sum(v["packages_delivered"] for k, v in report.items() if k != "best_agent")
    total_packages = len(data["packages"])
    if total_delivered != total_packages:
        logger.warning(
            "Packages delivered (%d) != total packages (%d). "
            "Check for an empty agents list or unreachable packages.",
            total_delivered, total_packages,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# 6. BONUS: ASCII route visualization
# ---------------------------------------------------------------------------

def ascii_visualize(data: Dict[str, Any], width: int = 60, height: int = 25) -> None:
    """Prints a crude ASCII map: warehouses (W), agent starts (A), and
    package destinations (.) scaled to a fixed-size grid."""
    all_points = (
        list(data["warehouses"].values())
        + list(data["agents"].values())
        + [p["destination"] for p in data["packages"]]
    )
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1
    span_y = (max_y - min_y) or 1

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def place(point: Point, ch: str) -> None:
        col = int((point[0] - min_x) / span_x * (width - 1))
        row = int((point[1] - min_y) / span_y * (height - 1))
        row = height - 1 - row  # flip so higher y is up
        grid[row][col] = ch

    for loc in data["warehouses"].values():
        place(loc, "W")
    for loc in data["agents"].values():
        place(loc, "A")
    for pkg in data["packages"]:
        place(pkg["destination"], ".")

    print("\nASCII route map  (W = warehouse, A = agent start, . = delivery)")
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")


# ---------------------------------------------------------------------------
# 7. BONUS: export top performer to CSV
# ---------------------------------------------------------------------------

def export_top_performer_csv(
    report: Dict[str, Any],
    results: "OrderedDict[str, Dict[str, Any]]",
    out_path: str = "top_performer.csv",
) -> Optional[str]:
    best = report.get("best_agent")
    if not best:
        return None
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "packages_delivered", "total_distance", "efficiency"])
        writer.writerow([
            best,
            report[best]["packages_delivered"],
            report[best]["total_distance"],
            report[best]["efficiency"],
        ])
        writer.writerow([])
        writer.writerow(["route_step", "x", "y"])
        for label, point in results[best]["route"]:
            writer.writerow([label, point[0], point[1]])
    return out_path


# ---------------------------------------------------------------------------
# 8. BONUS: new agent joining mid-day
# ---------------------------------------------------------------------------
# Simulates a fresh agent joining after some packages are already assigned.
# Packages from the SECOND HALF of the package list (mid-day onward) are
# re-assigned considering the new agent too. A simple, explainable model
# rather than a full re-optimization.
# ---------------------------------------------------------------------------

def simulate_with_midday_join(
    data: Dict[str, Any],
    new_agent_id: str,
    new_agent_location: Point,
    seed: int = 42,
) -> "OrderedDict[str, Dict[str, Any]]":
    packages = data["packages"]
    midpoint = len(packages) // 2
    morning_packages = packages[:midpoint]
    afternoon_packages = packages[midpoint:]

    morning_data = {**data, "packages": morning_packages}
    morning_assignment = assign_packages(morning_data)
    morning_results = simulate(morning_data, morning_assignment, seed=seed)

    agents_with_new = OrderedDict(data["agents"])
    agents_with_new[new_agent_id] = new_agent_location
    afternoon_data = {**data, "agents": agents_with_new, "packages": afternoon_packages}
    afternoon_assignment = assign_packages(afternoon_data)
    afternoon_results = simulate(afternoon_data, afternoon_assignment, seed=seed)

    combined: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for agent_id in agents_with_new:
        m = morning_results.get(agent_id, {"packages_delivered": 0, "total_distance": 0.0, "route": []})
        a = afternoon_results.get(agent_id, {"packages_delivered": 0, "total_distance": 0.0, "route": []})
        delivered = m["packages_delivered"] + a["packages_delivered"]
        distance = m["total_distance"] + a["total_distance"]
        combined[agent_id] = {
            "packages_delivered": delivered,
            "total_distance": round(distance, 2),
            "efficiency": round(distance / delivered, 2) if delivered else 0.0,
            "route": m["route"] + a["route"],
        }
    return combined


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    report_path: str = "report.json",
    show_ascii: bool = False,
    export_csv: bool = False,
    with_delays: bool = False,
) -> Dict[str, Any]:
    data = load_data(input_path)
    assignment = assign_packages(data)
    results = simulate(data, assignment, delays=with_delays)
    report = build_report(results)
    sanity_check(data, report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report for %s -> %s", input_path, report_path)
    print(json.dumps(report, indent=2))

    if show_ascii:
        ascii_visualize(data)

    if export_csv:
        csv_path = export_top_performer_csv(report, results)
        if csv_path:
            logger.info("Top performer exported -> %s", csv_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="FastBox Mystery Delivery System simulator")
    parser.add_argument("input", nargs="?", default="data.json", help="Path to input JSON file")
    parser.add_argument("--report", default="report.json", help="Path to write report.json")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII route visualization")
    parser.add_argument("--csv", action="store_true", help="Export top performer to CSV")
    parser.add_argument("--delays", action="store_true", help="Simulate random delivery delays")
    args = parser.parse_args()

    try:
        run(args.input, args.report, show_ascii=args.ascii, export_csv=args.csv,
            with_delays=args.delays)
    except (FileNotFoundError, DeliveryDataError) as exc:
        # Clean, actionable error instead of a raw traceback.
        logger.error(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
