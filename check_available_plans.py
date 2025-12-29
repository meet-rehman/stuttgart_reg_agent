# check_available_plans.py
from pathlib import Path

plans_dir = Path("data/raw/Landuse Plans")

print("="*70)
print("📁 AVAILABLE PLANS IN SYSTEM")
print("="*70)
print()

# Check PDFs
pdfs = list(plans_dir.rglob("*.pdf"))
print(f"📄 PDF files: {len(pdfs)}")
for pdf in sorted(pdfs):
    print(f"   - {pdf.name}")

print()

# Check PNGs (what vision agent uses)
pngs = list(plans_dir.rglob("*.png"))
print(f"🖼️  PNG files: {len(pngs)}")

# Group by plan
plan_groups = {}
for png in pngs:
    base_name = png.stem.rsplit('_page_', 1)[0]  # Remove _page_X
    if base_name not in plan_groups:
        plan_groups[base_name] = []
    plan_groups[base_name].append(png.name)

for plan_name, pages in sorted(plan_groups.items()):
    print(f"\n   📋 {plan_name}")
    print(f"      Pages: {len(pages)}")
    if '286' in plan_name or '18' in plan_name or 'stgt' in plan_name.lower():
        print(f"      ⭐ RELEVANT for Stgt 286/Plot 18A")

print()
print("="*70)
print("🔍 SEARCH FOR SPECIFIC FILES")
print("="*70)

# Search for Stgt 286
stgt_286_files = [f for f in pdfs if '286' in f.name]
print(f"\n📌 Files with '286': {len(stgt_286_files)}")
for f in stgt_286_files:
    print(f"   PDF: {f.name}")

# Check if converted to PNG
stgt_286_pngs = [f for f in pngs if '286' in f.name]
print(f"\n📌 PNG images with '286': {len(stgt_286_pngs)}")
for f in stgt_286_pngs[:5]:  # Show first 5
    print(f"   PNG: {f.name}")

# Search for Plot 18A references
plot_18_files = [f for f in pngs if '18' in f.name.lower()]
print(f"\n📌 Files mentioning '18': {len(plot_18_files)}")
for f in plot_18_files[:5]:
    print(f"   - {f.name}")