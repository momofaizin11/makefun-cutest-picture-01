# Plan — cutest-picture-01

## What the entry is
A repository holding `entry.png` (1024x1024 PNG, < 1 MiB) and `main.py`, a Python 3.13
standard-library program that copies those exact bytes to `$MAKEFUN_OUTPUT_DIR/entry.png`
(falling back to `/output`). `tools/render.py` is the studio that drew the image offline
(Pillow + numpy, never imported by the shipped program).

## The two concepts

**A — "Kitten in a teacup"**
A very small, very round kitten curled asleep inside a large ceramic teacup: oversized head,
closed happy eyes, tiny nose, blush cheeks, one ear up and one flopped, tail curled around
the rim like a blanket edge, a paw tucked under the chin. Warm cream cup with a hand-painted
terracotta rim on a saucer with biscuit crumbs and a mint sprig; a thin curl of steam bends
into a small heart. Palette cream / butter / soft pink / pistachio / one terracotta accent.
Warm soft light from the upper left, plush shading, no hard shadows.

**B — "Duckling under a mushroom"**
A fat duckling sheltering under a red-and-white dotted mushroom in the rain, holding a leaf
over its head, one tiny rain boot. Palette grass green / cherry red / duck yellow / rain blue.

Both are rendered at 1024x1024 and compared on criterion 1 (stop-and-melt at one glance) and
2 (an idea of its own). The winner is carried forward through three or more iterations.

## Why the winner meets the five judge criteria
1. Stops you: one huge soft subject — a kitten's face filling the middle of the frame — with
   a single warm glow behind it. Nothing competes with the face, so the eye lands there.
2. The idea: the cup is the bed. The steam that should be a wisp instead curls into a small
   heart, and the tail lies along the rim like the edge of a blanket: a point of view, not a
   catalog of objects.
3. Style: one mood (safe, warm, sleepy). Cream/butter/pistachio/pink with a single darker
   terracotta line for the cup's rim so the silhouette reads; soft light from the upper left.
4. Craft: every shape is analytic (ellipse, tapered stroke, heart curve) at 2x supersampling
   and downsampled once, so edges are clean, shading is smooth and there are no smears or
   stray marks. No text anywhere.
5. Reads small: the head circle is about a third of the frame, the eyes/blush/nose sit at
   large scale, and the silhouette (round head + two ears + cup + saucer) survives at 300 px.

## How it is rendered
`tools/render.py`, numpy for masks and per-shape shading (spherical normal for round parts,
ring-normal shading for the cup's rim, cylinder shading for the cup's body, warm bounce light for
the velvety look), Pillow for Gaussian blur and the final LANCZOS downsample. Deterministic: fixed
seed, no randomness in the shipped file.

## Which concept won, and why
Concept A (kitten in a teacup). Both were rendered and judged. B came back incoherent: the leaf
never read as an umbrella, the mushroom cap looked detached, the rain read as scratches across the
whole frame, and its rain-grey mood fights cuteness. A read correctly at 300 px from the first
render. A also has the stronger idea — the cup is the bed, and the steam that should be a wisp
instead curls into a heart — where B is a scene of the kind seen often.

## Iterations (each judged by looking at the render and its 300 px copy)
1. v1: full scene. Judged faults: kitten read as a cat/rabbit hybrid, the tail draped on the rim
   read as a striped sock, the cup's base was a straight cut, crumbs read as stones, the mint as a
   bow, a halo behind the left ear, a smear for a nose.
2. v2: pointed cat ears, cheek fur, whiskers, a bigger nose and mouth, two paws built from a palm
   plus toe lobes, a biscuit instead of crumbs, cleaner saucer ring, no halo. Still flagged: the
   tail (sock / bone), a cloud for the paw, the rim kinked where the tail crossed it.
3. v3: dropped the drape, moved the steam clear of the ears, flattened the folded ear, one paw with
   cut notches, one mint leaf, no crumbs. Flagged: the paw still a mitten; the heart unreadable.
4. v4: tail removed for good (it never stopped reading as an arm), three separate toe lobes, an
   explicit two-lobe heart, deeper hollow inside the cup, a bolder background, and the seam between
   the rim's two halves closed (it had left a light line). Flagged as the strongest of the series:
   the paw reads, the heart reads, the interior is clean.

The shipped image is that last version.
