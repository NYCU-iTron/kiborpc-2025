#!/usr/bin/env python3
from pathlib import Path

HOSTS_PATH = "/etc/hosts"
MARK = "#kiborpc"
HOSTNAMES = ["hlp", "mlp", "llp"]

def readfile(path):
    return Path(path).read_text(encoding="utf-8").splitlines()

def writefile(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

def is_our_entry(line, hostname):
    parts = line.strip().split()
    return (
        len(parts) >= 2 and
        parts[0] == "127.0.0.1" and
        parts[1] == hostname and
        parts[-1] == MARK
    )

def is_unmarked_entry(line, hostname):
    line = line.strip()
    if line.startswith("#"):
        return False
    parts = line.split()
    return (
        len(parts) >= 2 and
        parts[1] == hostname and
        MARK not in line
    )

def process_lines(lines):
    updated_lines = []
    already_handled = set()

    # comment out existing entries
    for line in lines:
        handled = False
        for hostname in HOSTNAMES:
            if is_our_entry(line, hostname):
                handled = True
                already_handled.add(hostname)
                updated_lines.append(line)
                break
            elif is_unmarked_entry(line, hostname):
                # Mark the entry
                updated_lines.append(f"# {line}")
                handled = True
                break
        if not handled:
            updated_lines.append(line)

    if len(already_handled) != len(HOSTNAMES):
        updated_lines.append("")
    for hostname in HOSTNAMES:
        if hostname not in already_handled:
            updated_lines.append(f"127.0.0.1 {hostname} {MARK}")

    return updated_lines

def main():
    path = Path(HOSTS_PATH)

    if not path.exists():
        print(f"Error: {HOSTS_PATH} does not exist")
        return 1
    
    try:
        lines = readfile(path)
        updated = process_lines(lines)
        writefile(path, updated)
        print(f"Updated {HOSTS_PATH} successfully.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()