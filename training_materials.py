import requests
import json
import time

# Lista materiałów (często stosowane w inżynierii)
MATERIALS = [
    "steel", "aluminum", "copper", "iron", "titanium", "brass", "bronze", "nickel",
    "carbon", "graphite", "plastic", "polycarbonate", "polyethylene", "rubber",
    "glass", "water", "air", "gold", "silver", "silicon", "carbon_fiber", "epoxy", "wood"
]

# Właściwości fizyczne (gęstość w kg/m³, lepkość w Pa·s – przybliżone lub szacunkowe)
PROPERTIES = {
    "steel": {"density": 7850, "viscosity": 0.0016},
    "aluminum": {"density": 2700, "viscosity": 0.0012},
    "copper": {"density": 8960, "viscosity": 0.0017},
    "iron": {"density": 7874, "viscosity": 0.0015},
    "titanium": {"density": 4500, "viscosity": 0.0013},
    "brass": {"density": 8500, "viscosity": 0.0016},
    "bronze": {"density": 8800, "viscosity": 0.0017},
    "nickel": {"density": 8900, "viscosity": 0.0017},
    "carbon": {"density": 2267, "viscosity": 0.0025},
    "graphite": {"density": 2200, "viscosity": 0.0024},
    "plastic": {"density": 950, "viscosity": 0.003},
    "polycarbonate": {"density": 1200, "viscosity": 0.004},
    "polyethylene": {"density": 930, "viscosity": 0.0035},
    "rubber": {"density": 1522, "viscosity": 0.005},
    "glass": {"density": 2500, "viscosity": 10},  # bardzo lepki w temp. pokojowej
    "water": {"density": 1000, "viscosity": 0.001},
    "air": {"density": 1.225, "viscosity": 0.0000181},
    "gold": {"density": 19300, "viscosity": 0.0024},
    "silver": {"density": 10490, "viscosity": 0.0019},
    "silicon": {"density": 2329, "viscosity": 0.003},
    "carbon_fiber": {"density": 1600, "viscosity": 0.0025},
    "epoxy": {"density": 1200, "viscosity": 10},
    "wood": {"density": 600, "viscosity": 0.002}
}

def fetch_description(name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "Brak opisu"), url
    else:
        print(f"⚠️ Brak opisu dla: {name}")
        return "Brak opisu", url

def build_material_entry(name):
    description, source = fetch_description(name)
    props = PROPERTIES.get(name)
    if not props:
        print(f"⚠️ Brak danych fizycznych dla: {name}, pomijam.")
        return None
    return {
        "name": name,
        "description": description,
        "source": source,
        "density": props["density"],
        "viscosity": props["viscosity"]
    }

def generate_materials():
    results = {}
    for name in MATERIALS:
        print(f"📥 Przetwarzam: {name}")
        entry = build_material_entry(name)
        if entry:
            results[name] = entry
        time.sleep(0.5)  # mały delay, żeby nie przeciążyć API
    return results

if __name__ == "__main__":
    data = generate_materials()
    with open("materials.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("✅ materials.json zapisany pomyślnie.")
