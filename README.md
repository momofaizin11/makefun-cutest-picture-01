# Kitten in a teacup — cutest-picture-01

The entry is one picture: `entry.png`, a 1024x1024 PNG, and `main.py`, which copies those exact
bytes to the output directory.

**The piece.** A very small, very round kitten curled asleep inside a large ceramic teacup, the way
a cat sleeps in a box it has decided is a bed. Its oversized head fills the middle of the frame:
closed happy eyes, a tiny pink nose, blush on both cheeks, one ear up and one folded over, and two
soft paws resting on the near lip of the cup. The cup is warm cream with a hand-painted terracotta
rim and a thin painted line on its flank, sitting on a saucer with a small pair of mint leaves to one
side. A wisp of steam rises and curls, at the top, into the shape of a small heart. Palette: cream,
butter, soft pink, pistachio green, one terracotta accent. Light is warm and soft from the upper
left, so the shading is plush and there are no hard shadows. Mood: safe, warm, cosy.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py   # writes out/entry.png
python3 main.py                                          # falls back to /output/entry.png
```

`main.py` is Python 3.13, standard library only, deterministic, and does nothing but copy the
picture, so `out/entry.png` is byte-identical to `entry.png`.

## How the picture was made

`tools/render.py` is the studio that drew it — numpy and Pillow, offline, not shipped and not
imported by the entry. Every shape is analytic (ellipse, tapered stroke, heart curve) drawn at twice
the final size and downsampled once with LANCZOS, so edges are clean and the shading is smooth.
`NOTES.md` holds the plan and the two concepts that were compared.

    python3 tools/render.py /tmp/entry.png a
