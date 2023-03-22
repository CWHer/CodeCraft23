import argparse
import re
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    scores = []
    pattern = re.compile(r'{"status":"Successful","score":(\d+)}')
    for map_id in range(1, 5):
        output = subprocess.check_output([
            "python", "main.py",
            "--map-id", f"maps/{map_id}.txt",
            "--seed", str(args.seed),
            "--no-statistics"
        ])
        result = pattern.search(output.decode('utf-8'))
        if not result:
            print(f"[ERROR]: Failed to parse output for map {map_id}")
            exit(1)
        scores.append(int(result.group(1)))
    print(f"\n\n[INFO]: {scores}")
    print(f"[INFO]: Total score {sum(scores)}")
