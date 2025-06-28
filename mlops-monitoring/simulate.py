import argparse
import random
import time
import requests
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate Iris model inference with optional drift injection."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000/predict",
        help="Inference endpoint URL",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between requests",
    )
    parser.add_argument(
        "--drift-prob",
        type=float,
        default=0.3,
        help="Probability of sending a drifted sample (0-1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of requests to send (0=infinite)",
    )
    args = parser.parse_args()
    if not 0 <= args.drift_prob <= 1:
        parser.error("--drift-prob must be between 0 and 1")
    return args

def normal_sample():
    # Typical training-range values
    return [
        random.uniform(4.5, 7.5),
        random.uniform(2.5, 4.5),
        random.uniform(1.0, 3.0),
        random.uniform(0.1, 1.0),
    ]

def drifted_sample():
    # Out-of-distribution values to simulate drift
    return [
        random.uniform(9.0, 12.0),
        random.uniform(0.5, 1.5),
        random.uniform(4.5, 7.5),
        random.uniform(2.5, 4.5),
    ]

def main():
    args = parse_args()
    sent = 0
    while True:
        if args.count and sent >= args.count:
            break

        if random.random() < args.drift_prob:
            features = drifted_sample()
            tag = "DRIFT"
        else:
            features = normal_sample()
            tag = "NORMAL"

        payload = {"features": features}
        start = time.time()
        try:
            r = requests.post(args.url, json=payload, timeout=5)
            status = r.status_code
            latency = time.time() - start
        except Exception as e:
            print(f"[ERROR] Request failed: {e}", file=sys.stderr)
            status = None
            latency = None

        print(
            f"{tag} | Features: {[round(f, 2) for f in features]} | "
            f"Status: {status} | Latency: {latency if latency is None else f'{latency:.3f}s'}"
        )

        sent += 1
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
