import argparse

def load_completion_time(file) -> float:
    with open(file, "r") as f:
        lines = f.readlines()
        n1, n2 = lines[0], lines[1]
        n1, n2 = float(n1), float(n2)
        nano = 1e9
        n1 /= nano
        n2 /= nano
        return n2 - n1
        

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file", "-f", type=str, required=True)
    args = args.parse_args()
    time = load_completion_time(args.file)
    print("Completion time:", time)
    
