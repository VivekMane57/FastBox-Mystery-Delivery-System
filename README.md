# FastBox Mystery Delivery System

A simulator for one day of FastBox delivery operations: parses warehouse /
agent / package data, assigns packages to the nearest agent, simulates
pickups + deliveries, and writes a `report.json`.

## Files
- `delivery_system.py` — the whole simulator (parsing, assignment, simulation,
  report, and bonus features). Fully type-hinted, uses `logging` instead of
  bare `print`, and raises specific exceptions (`DeliveryDataError`,
  `FileNotFoundError`) with actionable messages instead of raw tracebacks.
- `test_delivery_system.py` — pytest suite (20 tests): distance calc,
  assignment correctness + tie-breaking, simulation math, report/best-agent
  logic, the `packages_delivered == total_packages` sanity check, and input
  validation (missing keys, unknown warehouse refs, duplicate IDs, malformed
  coordinates, missing file).
- `requirements-dev.txt` — dev-only dependency (`pytest`) to run the tests.
- `data.json` — sample input (the example from the assignment PDF).
- `test_cases/` — the 10 provided test inputs.
- `reports/` — generated reports for every test case (for reference).
- `report.json` — output of the default run on `data.json`.
- `top_performer.csv` — bonus CSV export of the best agent's route.

## Running the tests
```bash
pip install -r requirements-dev.txt
pytest test_delivery_system.py -v
```
All 20 tests pass. Coverage highlights:
- Assignment produces the exact package-count split (2/2/1) from the PDF's
  sample data, and every package always gets assigned to exactly one agent.
- Distance/efficiency math is verified against hand-computed values.
- Bad input (missing key, unknown warehouse, duplicate package id, garbage
  coordinates, missing file) raises a clear, specific exception rather than
  crashing with a raw traceback.

## Usage
```bash
python delivery_system.py data.json
python delivery_system.py test_cases/test_case_3.json --report reports/tc3.json
python delivery_system.py data.json --ascii            # bonus: ASCII route map
python delivery_system.py data.json --csv               # bonus: export top performer CSV
python delivery_system.py data.json --delays             # bonus: simulate random delivery delays
```

## Assumptions made (documented, per the assignment's instructions to
proceed with the most logical approach on ambiguous points rather than
stopping to ask):

1. **Two input schemas.** The PDF's example and the actual `test_case_*.json`
   files don't use identical key names (`warehouse_id` vs `warehouse`,
   dict-style vs list-style warehouses/agents). `load_data()` normalizes
   both into a single internal shape so the same code runs on every file.

2. **"Nearest agent" = distance from the agent's starting location to the
   package's warehouse.** As literally stated in the brief. Agents are not
   removed from the pool after taking a package, so one agent can (and
   often does) end up responsible for several packages — this matches the
   sample report in the PDF (A1 and A2 each deliver 2 packages there).

3. **Tie-break rule.** If two agents are exactly equidistant from a
   warehouse, the agent listed earlier in the input file wins. Keeps
   results deterministic and reproducible.

4. **Pickup/delivery order.** Each agent processes their assigned packages
   in the order those packages appear in the input file.

5. **Route model.** An agent does **not** teleport back to their starting
   point between deliveries. For each package: `current position -> pickup
   at warehouse -> deliver at destination`, and `current position` becomes
   that destination for the next pickup. This is the more realistic model
   of "one continuous day of operations".

   Note: the package **counts** in this model (A1: 2, A2: 2, A3: 1) match
   the PDF's sample report exactly, confirming the assignment logic is
   correct. The PDF's sample **distance/efficiency** numbers (85.32,
   120.12, 50.00), however, could not be reproduced under any literal
   reading of the routing description we tried (batch pickup, chained
   delivery, round-trip-per-package, etc.) — they appear to be illustrative
   placeholder values for the report *format*, not the actual output of
   running the described algorithm on the sample input.

6. **Efficiency** = `total_distance / packages_delivered` (avg distance
   per delivery — lower is better). **`best_agent`** = the agent with the
   lowest efficiency among agents who delivered at least one package.

7. **Sanity check.** After every run, the script checks that
   `sum(packages_delivered) == total packages` and prints a warning if not
   (e.g. if the `agents` list is empty).

## Bonus features implemented
- **ASCII route visualization** (`--ascii`): plots warehouses (`W`), agent
  start points (`A`), and delivery destinations (`.`) on a text grid.
- **CSV export of top performer** (`--csv`): writes the best agent's stats
  and full route (as x/y waypoints) to `top_performer.csv`.
- **Random delivery delays** (`--delays`): attaches a random delay (0–15
  min) to some deliveries in the report, purely as an informational log —
  it does not affect distance/efficiency numbers.
- **New agent joining mid-day** (`simulate_with_midday_join()` in
  `delivery_system.py`): splits the day's packages in half, runs the
  morning normally, adds the new agent for the afternoon half, and merges
  the two halves' stats per agent. Simple and explainable rather than a
  full re-optimization.

## Verification
All 10 provided test cases were run through the simulator — every one
passes the `packages_delivered == total packages` sanity check with no
warnings (see `reports/` for each test case's output).
