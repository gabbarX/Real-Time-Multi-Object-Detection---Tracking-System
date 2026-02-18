#!/usr/bin/env python3
"""
Download a handful of COCO val images for the web-app "Try a Sample" feature.
These are public-domain images from the COCO 2017 validation set.
"""

from pathlib import Path
from urllib.request import urlretrieve

SAMPLES = {
    "cats_on_couch.jpg":      "http://images.cocodataset.org/val2017/000000039769.jpg",
    "restaurant_kitchen.jpg":      "http://images.cocodataset.org/val2017/000000397133.jpg",
    "people_outdoors.jpg":    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "skating_ring.jpg":   "http://images.cocodataset.org/val2017/000000087038.jpg",
    "cool_bike_.jpg":  "http://images.cocodataset.org/val2017/000000007386.jpg",
    "kitchen_counter.jpg":    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "Tennis_player.jpg":"http://farm4.staticflickr.com/3158/2996782298_bf54af224d_z.jpg"
    "Food.jpg":"http://farm6.staticflickr.com/5084/5256361266_faa0448b7e_z.jpg"
}

def main():
    dest = Path(__file__).resolve().parent.parent / "web" / "static" / "samples"
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        out = dest / name
        if out.exists():
            print(f"  ✓ {name} (cached)")
            continue
        print(f"  ↓ {name} … ", end="", flush=True)
        urlretrieve(url, str(out))
        print("done")
    print(f"\n{len(SAMPLES)} sample images ready in {dest}")

if __name__ == "__main__":
    main()
