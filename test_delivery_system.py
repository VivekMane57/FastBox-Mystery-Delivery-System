"""
Unit tests for the FastBox Mystery Delivery System.

Run with:
    pytest test_delivery_system.py -v
"""

import json
import math
import pytest

from delivery_system import (
    euclidean,
    load_data,
    assign_packages,
    simulate,
    build_report,
    sanity_check,
    DeliveryDataError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """The example dataset from the assignment PDF."""
    return {
        "warehouses": {"W1": (0.0, 0.0), "W2": (50.0, 75.0), "W3": (100.0, 25.0)},
        "agents": {"A1": (5.0, 5.0), "A2": (60.0, 60.0), "A3": (95.0, 30.0)},
        "packages": [
            {"id": "P1", "warehouse": "W1", "destination": (30.0, 40.0)},
            {"id": "P2", "warehouse": "W2", "destination": (70.0, 90.0)},
            {"id": "P3", "warehouse": "W3", "destination": (105.0, 20.0)},
            {"id": "P4", "warehouse": "W1", "destination": (10.0, 10.0)},
            {"id": "P5", "warehouse": "W2", "destination": (40.0, 80.0)},
        ],
    }


@pytest.fixture
def write_json(tmp_path):
    """Helper: writes a dict to a temp JSON file and returns its path."""
    def _write(name, content):
        path = tmp_path / name
        path.write_text(json.dumps(content), encoding="utf-8")
        return str(path)
    return _write


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

def test_euclidean_basic():
    assert euclidean((0, 0), (3, 4)) == 5.0


def test_euclidean_zero_distance():
    assert euclidean((7, 2), (7, 2)) == 0.0


def test_euclidean_matches_math_hypot():
    p1, p2 = (1.5, -2.5), (-3.0, 4.0)
    expected = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    assert euclidean(p1, p2) == expected


# ---------------------------------------------------------------------------
# Agent <-> package assignment
# ---------------------------------------------------------------------------

def test_assignment_matches_expected_counts(sample_data):
    """A1 and A2 should each get 2 packages, A3 gets 1 -- matches the
    package counts in the assignment PDF's sample report."""
    assignment = assign_packages(sample_data)
    assert len(assignment["A1"]) == 2
    assert len(assignment["A2"]) == 2
    assert len(assignment["A3"]) == 1


def test_assignment_covers_every_package(sample_data):
    assignment = assign_packages(sample_data)
    total_assigned = sum(len(pkgs) for pkgs in assignment.values())
    assert total_assigned == len(sample_data["packages"])


def test_assignment_with_no_agents_returns_empty_groups(sample_data):
    sample_data["agents"] = {}
    assignment = assign_packages(sample_data)
    assert assignment == {}


def test_assignment_tie_break_prefers_earlier_agent():
    """Two agents equidistant from the warehouse -> the one listed first wins."""
    data = {
        "warehouses": {"W1": (0.0, 0.0)},
        "agents": {"A1": (10.0, 0.0), "A2": (-10.0, 0.0)},  # both dist=10 from W1
        "packages": [{"id": "P1", "warehouse": "W1", "destination": (5.0, 5.0)}],
    }
    assignment = assign_packages(data)
    assert len(assignment["A1"]) == 1
    assert len(assignment["A2"]) == 0


# ---------------------------------------------------------------------------
# Simulation + report
# ---------------------------------------------------------------------------

def test_simulate_single_package_distance():
    """Agent starts AT the warehouse -> total distance is just warehouse->dest."""
    data = {
        "warehouses": {"W1": (0.0, 0.0)},
        "agents": {"A1": (0.0, 0.0)},
        "packages": [{"id": "P1", "warehouse": "W1", "destination": (3.0, 4.0)}],
    }
    assignment = assign_packages(data)
    results = simulate(data, assignment)
    assert results["A1"]["packages_delivered"] == 1
    assert results["A1"]["total_distance"] == 5.0
    assert results["A1"]["efficiency"] == 5.0


def test_efficiency_is_average_distance_per_package(sample_data):
    """efficiency should equal total_distance / packages_delivered, within
    rounding tolerance (efficiency is computed from the raw pre-rounded
    distance, while total_distance itself is rounded separately for display)."""
    assignment = assign_packages(sample_data)
    results = simulate(sample_data, assignment)
    for agent_id, r in results.items():
        if r["packages_delivered"] > 0:
            expected = r["total_distance"] / r["packages_delivered"]
            assert r["efficiency"] == pytest.approx(expected, abs=0.01)


def test_best_agent_has_lowest_efficiency(sample_data):
    assignment = assign_packages(sample_data)
    results = simulate(sample_data, assignment)
    report = build_report(results)
    delivering_agents = {k: v for k, v in report.items() if k != "best_agent" and v["packages_delivered"] > 0}
    best_efficiency = min(v["efficiency"] for v in delivering_agents.values())
    assert report[report["best_agent"]]["efficiency"] == best_efficiency


def test_agent_with_zero_packages_has_zero_stats(sample_data):
    sample_data["agents"]["A4"] = (500.0, 500.0)  # far away, gets nothing
    assignment = assign_packages(sample_data)
    results = simulate(sample_data, assignment)
    report = build_report(results)
    assert report["A4"]["packages_delivered"] == 0
    assert report["A4"]["total_distance"] == 0.0
    assert report["A4"]["efficiency"] == 0.0


def test_sanity_check_passes_when_all_packages_delivered(sample_data):
    assignment = assign_packages(sample_data)
    results = simulate(sample_data, assignment)
    report = build_report(results)
    assert sanity_check(sample_data, report) is True


def test_sanity_check_fails_with_no_agents(sample_data):
    sample_data["agents"] = {}
    assignment = assign_packages(sample_data)
    results = simulate(sample_data, assignment)
    report = build_report(results)
    assert sanity_check(sample_data, report) is False


# ---------------------------------------------------------------------------
# JSON loading: both schemas + validation errors
# ---------------------------------------------------------------------------

def test_load_data_dict_style_schema(write_json):
    path = write_json("data.json", {
        "warehouses": {"W1": [0, 0]},
        "agents": {"A1": [1, 1]},
        "packages": [{"id": "P1", "warehouse": "W1", "destination": [2, 2]}],
    })
    data = load_data(path)
    assert data["warehouses"]["W1"] == (0.0, 0.0)
    assert data["packages"][0]["warehouse"] == "W1"


def test_load_data_list_style_schema(write_json):
    path = write_json("data.json", {
        "warehouses": [{"id": "W1", "location": [0, 0]}],
        "agents": [{"id": "A1", "location": [1, 1]}],
        "packages": [{"id": "P1", "warehouse_id": "W1", "destination": [2, 2]}],
    })
    data = load_data(path)
    assert data["warehouses"]["W1"] == (0.0, 0.0)
    assert data["packages"][0]["warehouse"] == "W1"


def test_load_data_missing_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_data("this_file_does_not_exist.json")


def test_load_data_missing_top_level_key_raises(write_json):
    path = write_json("bad.json", {"warehouses": {}, "packages": []})  # no "agents"
    with pytest.raises(DeliveryDataError, match="agents"):
        load_data(path)


def test_load_data_unknown_warehouse_reference_raises(write_json):
    path = write_json("bad.json", {
        "warehouses": {"W1": [0, 0]},
        "agents": {"A1": [1, 1]},
        "packages": [{"id": "P1", "warehouse": "W_GHOST", "destination": [2, 2]}],
    })
    with pytest.raises(DeliveryDataError, match="unknown warehouse"):
        load_data(path)


def test_load_data_duplicate_package_id_raises(write_json):
    path = write_json("bad.json", {
        "warehouses": {"W1": [0, 0]},
        "agents": {"A1": [1, 1]},
        "packages": [
            {"id": "P1", "warehouse": "W1", "destination": [2, 2]},
            {"id": "P1", "warehouse": "W1", "destination": [3, 3]},
        ],
    })
    with pytest.raises(DeliveryDataError, match="Duplicate package id"):
        load_data(path)


def test_load_data_invalid_coordinates_raises(write_json):
    path = write_json("bad.json", {
        "warehouses": {"W1": ["not", "numbers"]},
        "agents": {"A1": [1, 1]},
        "packages": [],
    })
    with pytest.raises(DeliveryDataError):
        load_data(path)
