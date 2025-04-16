#!/usr/bin/env python3
from pathlib import Path

HOSTS_PATH = "/etc/hosts"
MARK = "#kiborpc"
HOSTNAMES = ["hlp", "mlp", "llp"]

def readfile(path):
    return Path(path).read_text(encoding="utf-8").splitlines()

def writefile(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

def is_our_entry(line):
    parts = line.strip().split()
    return (
        len(parts) >= 2 and
        parts[0] == "127.0.0.1" and
        parts[1] in HOSTNAMES and
        parts[-1] == MARK
    )

def is_commented_original(line, hostname):
    stripped = line.strip()
    return (
        stripped.startswith("#") and
        not stripped.endswith(MARK) and
        any(part == hostname for part in stripped.lstrip("#").strip().split())
    )

def uncomment(line):
    return line.lstrip("#").strip()

def restore_lines(lines):
    restored = []
    for line in lines:
        if is_our_entry(line):
            continue
        
        uncommented = False
        for hostname in HOSTNAMES:
            if is_commented_original(line, hostname):
                new_line = uncomment(line)
                restored.append(new_line)
                uncommented = True
                break

        if not uncommented:
            restored.append(line)

    # remove trailing empty lines
    while restored and not restored[-1].strip():
        restored.pop()

    return restored

def main():
    path = Path(HOSTS_PATH)

    if not path.exists():
        print(f"File {HOSTS_PATH} does not exist.")
        return
    
    try:
        lines = readfile(path)
        updated = restore_lines(lines)
        writefile(path, updated)
        print(f"Updated {HOSTS_PATH} successfully.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()