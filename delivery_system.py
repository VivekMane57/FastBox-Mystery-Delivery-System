"""
FastBox Mystery Delivery System

Simulates one day of delivery operations: assigns packages to the
nearest agent, simulates pickup/delivery routes, and writes a report.

Usage:
    python delivery_system.py data.json
    python delivery_system.py data.json --report report.json --ascii --csv --delays

Tests:
    pytest test_delivery_system.py -v
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


class DeliveryDataError(ValueError):
    """Raised when input JSON is missing fields or malformed."""


# --------------------------------------------------------------------------
# JSON parsing
# --------------------------------------------------------------------------

def _normalize_locations(raw: Any, field_name: str) -> "OrderedDict[str, Point]":
    """Normalizes {'id': [x, y]} or [{'id':.., 'location':[x, y]}] into an
    OrderedDict of id -> (x, y). Order is preserved for tie-breaking."""
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
    """Reads and validates the input JSON, normalizing both supported
    schemas (dict-style and list-style warehouses/agents; 'warehouse'
    or 'warehouse_id' package key) into a single internal structure."""
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


# --------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------

def euclidean(p1: Point, p2: Point) -> float:
    """Straight-line distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------

def assign_packages(data: Dict[str, Any]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """Assigns each package to the agent nearest its warehouse. Agents
    can receive multiple packages; ties go to the earlier-listed agent."""
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


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(
    data: Dict[str, Any],
    assignment: "OrderedDict[str, List[Dict[str, Any]]]",
    delays: bool = False,
    seed: int = 42,
) -> "OrderedDict[str, Dict[str, Any]]":
    """Runs each agent through their assigned packages in order: current
    position -> warehouse -> destination, updating position after each
    delivery. Returns per-agent stats plus the route (for the ASCII bonus)."""
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


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(results: "OrderedDict[str, Dict[str, Any]]") -> Dict[str, Any]:
    """Builds the final report dict; best_agent = lowest efficiency
    (avg distance per package) among agents who delivered at least one."""
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
    """Verifies total packages delivered matches total packages."""
    total_delivered = sum(v["packages_delivered"] for k, v in report.items() if k != "best_agent")
    total_packages = len(data["packages"])
    if total_delivered != total_packages:
        logger.warning(
            "Packages delivered (%d) != total packages (%d)",
            total_delivered, total_packages,
        )
        return False
    return True


# --------------------------------------------------------------------------
# Bonus: ASCII route visualization
# --------------------------------------------------------------------------

def ascii_visualize(data: Dict[str, Any], width: int = 60, height: int = 25) -> None:
    """Prints warehouses (W), agent starts (A), and destinations (.) on a grid."""
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
        row = height - 1 - row
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


# --------------------------------------------------------------------------
# Bonus: export top performer to CSV
# --------------------------------------------------------------------------

def export_top_performer_csv(
    report: Dict[str, Any],
    results: "OrderedDict[str, Dict[str, Any]]",
    out_path: str = "top_performer.csv",
) -> Optional[str]:
    """Writes the best agent's stats and full route to a CSV file."""
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


# --------------------------------------------------------------------------
# Bonus: new agent joining mid-day
# --------------------------------------------------------------------------

def simulate_with_midday_join(
    data: Dict[str, Any],
    new_agent_id: str,
    new_agent_location: Point,
    seed: int = 42,
) -> "OrderedDict[str, Dict[str, Any]]":
    """Splits packages in half; runs the morning half normally, then adds
    the new agent for the afternoon half and merges per-agent stats."""
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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
        logger.error(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()