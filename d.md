# Goal

Automatically create and format professional, standardized images for the profiles of Motivational Speakers, Comedians, and other artists featured on `engage4more.com`.

---

# AI Agent Objective

Generate two sets of images per artist:

## 1) Profile Image (Main)

- **Dimension:** 765 x 480 px
- **Format:** `.webp`
- **Max Size:** 50 KB
- **Style:** Highlight face area (soft shading as per reference), centered composition, no head cropping, horizontal orientation

## 2) Gallery Images (6 per artist)

- **Dimension:** 1176 x 516 px
- **Format:** `.webp`
- **Max Size:** 50 KB each
- **Style:** Natural, event/action shots - speaking on stage, interacting with audience, candid or expressive moments

---

# Technical Workflow Proposal

## Step 1: Upload Input

- Upload a few reference images per artist
- Optionally include tags/prompts (e.g., "motivational speaker on stage," "corporate keynote," "stand-up comedy show")

## Step 2: AI Processing

- Detect and crop face-centered framing
- Apply soft gradient shading to emphasize the face (as per reference style)
- Generate:
  - One main profile image (765 x 480)
  - Six gallery variants (1176 x 516)

## Step 3: Optimization

- Compress images under 50 KB using:
  - Resolution balancing
  - WebP optimization
  - Background simplification for clarity and consistency

## Step 4: Output Structure

```text
/artist_name/
  profile.webp
  gallery1.webp
  gallery2.webp
  gallery3.webp
  gallery4.webp
  gallery5.webp
  gallery6.webp
```

---

# Optional Enhancements

- **Auto background cleaner:** Replace cluttered backdrops with a subtle gradient or blurred event background.
- **Consistent color tone:** Ensure all artist images across the site follow a cohesive visual style.


