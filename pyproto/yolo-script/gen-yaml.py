from pathlib import Path

DATASET_DIR = Path("../../assets/dataset")
CLASSES_TXT = DATASET_DIR / "classes.txt"
DATA_YAML = DATASET_DIR / "data.yaml"

if DATA_YAML.exists():
    print("YAML file already exists.")
    exit(0)

with open(CLASSES_TXT, "r") as f:
    classes = [line.strip() for line in f.readlines()]

dataYamlContent = f"""
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
"""

for i, name in enumerate(classes):
    dataYamlContent += f"  {i}: {name}\n"

with open(DATA_YAML, "w") as f:
    f.write(dataYamlContent)

print(f"YAML file generated at {DATA_YAML.resolve()}")