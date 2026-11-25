"""Load test for the content service.

Drives POST /summarize at increasing concurrency and reports sustained throughput and
latency. Run the service against the instant mock LLM (bench/mock_llm.py) so the numbers
reflect the service itself, not a remote provider.

  cached   - a small pool of texts is warmed first, then every request is a Redis cache hit
  uncached - every request has unique text, so it misses the cache and goes through
             token batching to the (mock) LLM

Usage:
  python bench/load_test.py --url http://localhost:8000 --levels 1,10,50,100,200 --duration 10
"""

import argparse
import asyncio
import statistics
import time
import uuid

import aiohttp

POOL_SIZE = 50


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def make_text(i: int, nonce: str) -> str:
    return (f"Document {i} ({nonce}). Async Python services use an event loop to overlap I/O. "
            "Caching repeated LLM responses in Redis avoids paying model latency twice.")


async def run_level(client: aiohttp.ClientSession, mode: str, concurrency: int, duration: float):
    nonce = uuid.uuid4().hex[:8]
    latencies: list[float] = []
    lookups: list[float] = []
    errors = 0
    counter = 0
    deadline = time.perf_counter() + duration

    async def worker():
        nonlocal errors, counter
        while time.perf_counter() < deadline:
            counter += 1
            i = counter % POOL_SIZE if mode == "cached" else counter
            text = make_text(i, "warm" if mode == "cached" else nonce)
            start = time.perf_counter()
            try:
                async with client.post("/summarize", json={"text": text}) as r:
                    await r.read()
                elapsed = (time.perf_counter() - start) * 1000
                if r.status != 200:
                    errors += 1
                    continue
                latencies.append(elapsed)
                if r.headers.get("x-cache") == "HIT":
                    lookups.append(float(r.headers["x-cache-lookup-ms"]))
            except (aiohttp.ClientError, asyncio.TimeoutError):
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "rps": len(latencies) / wall,
        "p50": pct(latencies, 50),
        "p99": pct(latencies, 99),
        "errors": errors,
        "hit_rate": len(lookups) / len(latencies) if latencies else 0.0,
        "lookup_p50": statistics.median(lookups) if lookups else float("nan"),
        "lookup_p99": pct(lookups, 99),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--levels", default="1,10,50,100,200")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds per concurrency level")
    ap.add_argument("--modes", default="cached,uncached")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    modes = args.modes.split(",")

    connector = aiohttp.TCPConnector(limit=max(levels))
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(args.url, connector=connector, timeout=timeout) as client:
        async with client.get("/health") as r:
            r.raise_for_status()
        if "cached" in modes:  # warm the cache pool
            for i in range(POOL_SIZE):
                async with client.post("/summarize", json={"text": make_text(i, "warm")}) as r:
                    r.raise_for_status()

        for mode in modes:
            print(f"\nmode={mode}  duration={args.duration:.0f}s/level")
            print(f"{'conc':>5} {'requests':>9} {'req/s':>9} {'p50 ms':>8} {'p99 ms':>8} "
                  f"{'errors':>6} {'hit%':>5} {'redis p50/p99 ms':>17}")
            for c in levels:
                r = await run_level(client, mode, c, args.duration)
                print(f"{r['concurrency']:>5} {r['requests']:>9} {r['rps']:>9.0f} {r['p50']:>8.2f} "
                      f"{r['p99']:>8.2f} {r['errors']:>6} {r['hit_rate'] * 100:>5.0f} "
                      f"{r['lookup_p50']:>8.3f}/{r['lookup_p99']:.3f}")
        async with client.get("/stats") as r:
            stats = await r.json()
        s = stats["summarize"]
        if s["items"]:
            print(f"\ntoken batching: {s['items']} LLM-bound items sent in {s['llm_calls']} LLM calls "
                  f"({s['items'] / max(1, s['llm_calls']):.1f} items/call)")


if __name__ == "__main__":
    asyncio.run(main())
