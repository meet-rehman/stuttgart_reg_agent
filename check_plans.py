# check_plans.py
from pathlib import Path

plans_dir = Path("data/raw/Landuse Plans")
plans = []

for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
    plans.extend(plans_dir.rglob(f"images/{ext}"))
    plans.extend(plans_dir.rglob(ext))

plans = list(set(plans))

print(f"Found {len(plans)} plan images:\n")
for i, plan in enumerate(sorted(plans), 1):
    print(f"{i:2}. {plan.name}")

# Check which might contain plot info
print("\n\nPlans that might contain plot numbers:")
for plan in sorted(plans):
    name = plan.name.lower()
    if any(keyword in name for keyword in ['stgt', 'nord', 'plan', 'flur', 'bebauung']):
        print(f"  → {plan.name}")