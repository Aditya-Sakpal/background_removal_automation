"""Measure face Y positions on a 765x480 image vs guide lines."""
import sys

from PIL import Image, ImageDraw

from profile_align import (
    GUIDE_BOTTOM_Y,
    GUIDE_CENTER_X,
    GUIDE_TOP_Y,
    PROFILE_HEIGHT,
    PROFILE_WIDTH,
    detect_face_metrics,
)

def main(path: str) -> None:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    hairline_y, chin_y, center_x = detect_face_metrics(img)
    face_h = chin_y - hairline_y

    print(f"Image size: {w} x {h} (expected {PROFILE_WIDTH} x {PROFILE_HEIGHT})")
    print(f"Guide top (hairline target):    Y = {GUIDE_TOP_Y}")
    print(f"Guide bottom (chin target):     Y = {GUIDE_BOTTOM_Y}")
    print(f"Guide band height:              {GUIDE_BOTTOM_Y - GUIDE_TOP_Y} px")
    print()
    print(f"Detected hairline (est.):       Y = {hairline_y:.1f}")
    print(f"Detected chin:                  Y = {chin_y:.1f}")
    print(f"Detected face height:           {face_h:.1f} px")
    print(f"Detected face center X:         {center_x:.1f} (target {GUIDE_CENTER_X})")
    print()
    print(f"Hairline error:                 {hairline_y - GUIDE_TOP_Y:+.1f} px")
    print(f"Chin error:                     {chin_y - GUIDE_BOTTOM_Y:+.1f} px")
    print(f"Face height vs guide band:      {face_h:.1f} vs {GUIDE_BOTTOM_Y - GUIDE_TOP_Y}")

    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.line([(0, GUIDE_TOP_Y), (w, GUIDE_TOP_Y)], fill=(0, 255, 255), width=2)
    draw.line([(0, GUIDE_BOTTOM_Y), (w, GUIDE_BOTTOM_Y)], fill=(0, 255, 255), width=2)
    draw.line([(GUIDE_CENTER_X, 0), (GUIDE_CENTER_X, h)], fill=(0, 255, 255), width=1)
    draw.line([(0, int(hairline_y)), (w, int(hairline_y))], fill=(255, 0, 0), width=2)
    draw.line([(0, int(chin_y)), (w, int(chin_y))], fill=(255, 128, 0), width=2)
    out.save("verify_guides_overlay.jpg", quality=92)
    print("\nSaved verify_guides_overlay.jpg (cyan=guides, red=hairline, orange=chin)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "profile_aligned_test.jpg")
