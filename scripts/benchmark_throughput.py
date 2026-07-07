"""
Task 2: Benchmark Pipeline Throughput
======================================
Times the full 15-layer DQS pipeline across different batch sizes.
Runs 3 iterations per batch size and reports median timing.

Run: .venv\\Scripts\\python scripts/benchmark_throughput.py
"""
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_benchmark():
    print("=" * 60)
    print("  PIPELINE THROUGHPUT BENCHMARK")
    print("=" * 60)

    from src.data_generator import generate_visa_transactions
    from src.dqs_engine import DQSEngine

    BATCH_SIZES = [10, 50, 100, 200, 500]
    ITERATIONS = 3  # Median of 3 runs per batch size

    results = []

    for n in BATCH_SIZES:
        print(f"\n[Batch={n}] Running {ITERATIONS} iterations...")

        transactions = generate_visa_transactions(
            n_transactions=n,
            anomaly_rate=0.10,
            random_seed=42,
        )

        timings = []
        for i in range(ITERATIONS):
            engine = DQSEngine(use_ai=False)
            t0 = time.perf_counter()
            result = engine.run(transactions)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            timings.append(elapsed_ms)
            print(f"    Run {i+1}: {elapsed_ms:.1f}ms")

        median_ms = statistics.median(timings)
        ms_per_record = median_ms / n

        results.append({
            "batch_size": n,
            "median_ms": median_ms,
            "ms_per_record": ms_per_record,
            "min_ms": min(timings),
            "max_ms": max(timings),
        })

        print(f"    Median: {median_ms:.1f}ms | Per record: {ms_per_record:.2f}ms")

    # Print summary table
    print("\n")
    print("=" * 60)
    print("  THROUGHPUT RESULTS")
    print("=" * 60)
    print(f"  {'Batch':>6}  {'Median(ms)':>12}  {'ms/record':>10}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r['batch_size']:>6}  {r['median_ms']:>12.1f}  {r['ms_per_record']:>10.2f}  {r['min_ms']:>8.1f}  {r['max_ms']:>8.1f}")

    # Pick the 500-record result for resume bullet
    r500 = next(r for r in results if r["batch_size"] == 500)

    print(f"\n  Resume bullet:")
    print(f"  \"15-layer pipeline processes 500 transactions in under {r500['median_ms']:.0f}ms")
    print(f"   (avg {r500['ms_per_record']:.1f}ms/record end-to-end)\"")

    # Append to BENCHMARKS.md
    benchmarks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "BENCHMARKS.md"
    )

    throughput_section = f"""
## Pipeline Throughput
*Full 15-layer pipeline, median of {ITERATIONS} runs, no AI (use_ai=False)*

| Batch Size | Median (ms) | ms / record | Min (ms) | Max (ms) |
|------------|-------------|-------------|----------|----------|
"""
    for r in results:
        throughput_section += f"| {r['batch_size']} | {r['median_ms']:.1f} | {r['ms_per_record']:.2f} | {r['min_ms']:.1f} | {r['max_ms']:.1f} |\n"

    throughput_section += f"""
**Resume bullet:**
> "15-layer pipeline processes 500 transactions in under {r500['median_ms']:.0f}ms ({r500['ms_per_record']:.1f}ms/record end-to-end)"
"""

    with open(benchmarks_path, "a", encoding="utf-8") as f:
        f.write(throughput_section)

    print(f"\n  Appended to: {benchmarks_path}")
    print("\n" + "=" * 60)
    print("  TASK 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
