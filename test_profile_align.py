"""Quick local test: align a reference image and save preview."""
import sys

from PIL import Image

from profile_align import align_profile_to_grid, get_profile_template

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "netanyahu's image.jpg"
    with open(path, "rb") as f:
        data = f.read()
    out = align_profile_to_grid(data)
    out.save("profile_aligned_test.jpg", quality=92)
    get_profile_template().save("profile_template_test.jpg", quality=92)
    print(f"Saved profile_aligned_test.jpg from {path}")

if __name__ == "__main__":
    main()
