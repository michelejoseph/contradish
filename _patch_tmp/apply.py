import sys

def apply_patch(target_path, old_path, new_path):
    with open(target_path, encoding="utf-8") as f:
        target = f.read()
    with open(old_path, encoding="utf-8") as f:
        old = f.read()
    with open(new_path, encoding="utf-8") as f:
        new = f.read()
    if old not in target:
        sys.exit(f"ANCHOR NOT FOUND for {target_path} <- {old_path}")
    if target.count(old) != 1:
        sys.exit(f"ANCHOR NOT UNIQUE ({target.count(old)}x) for {target_path} <- {old_path}")
    target = target.replace(old, new, 1)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(target)
    print(f"patched {target_path} using {old_path} -> {new_path}")

apply_patch("../contradish/cli.py", "old_cmd_compare.txt", "new_cmd_compare.txt")
apply_patch("../contradish/cli.py", "old_cmp_args.txt", "new_cmp_args.txt")
apply_patch("../contradish/distinction.py", "old_distinction_tail.txt", "new_distinction_tail.txt")
