import argparse
import random
import time
import requests
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Aggressive drift simulation for Iris API")
    parser.add_argument("--url", type=str, default="http://localhost:5000/predict",
                        help="Inference endpoint URL")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between requests")
    parser.add_argument("--ramp-time", type=int, default=60,
                        help="Duration in seconds to ramp drift from 0% to 100%")
    parser.add_argument("--hold-time", type=int, default=60,
                        help="Duration in seconds to hold 100% drift traffic")
    args = parser.parse_args()
    return args

# Sample generators
def normal_sample():
    return [random.uniform(4.5, 7.5), random.uniform(2.5, 4.5), random.uniform(1.0, 3.0), random.uniform(0.1, 1.0)]

def drifted_sample():
    return [random.uniform(9.0, 12.0), random.uniform(0.5, 1.5), random.uniform(4.5, 7.5), random.uniform(2.5, 4.5)]

if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    total = args.ramp_time + args.hold_time

    while True:
        elapsed = time.time() - start
        # compute drift proportion: ramp from 0 to 1 over ramp_time, then stay at 1
        if elapsed < args.ramp_time:
            p = elapsed / args.ramp_time
        elif elapsed < total:
            p = 1.0
        else:
            # reset cycle
            start = time.time()
            continue

        # decide sample type
        if random.random() < p:
            features = drifted_sample()
            tag = 'DRIFT'
        else:
            features = normal_sample()
            tag = 'NORMAL'

        payload = {'features': features}
        try:
            r = requests.post(args.url, json=payload, timeout=5)
            status = r.status_code
        except Exception as e:
            print(f"[ERROR] Request failed: {e}", file=sys.stderr)
            status = None

        print(f"{tag} | p_drift={p:.2f} | features={['{:.2f}'.format(f) for f in features]} | status={status}")
        time.sleep(args.interval)