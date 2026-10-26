# PROMPT RULES — Locked, paste into every image prompt

## ANCHOR ORDER (MANDATORY)
Every person-containing frame uses these TWO anchors (in this order):
- anchor[0] = `style_tests/ref_mc_fullbody.png` → defines stick-figure body, legs, head, hands. ALL people (MC + zombies + flashback figures) use THIS body type unless explicitly stated otherwise.
- anchor[1] = `style_tests/styleA_highcontrast.png` → defines B&W inkwash style, contrast, environment rendering.

For EXTERIOR store shots, add `style_tests/ref_store_pristine.png` as anchor[2] to lock the building baseline.

## MANDATORY OPENING PARAGRAPH (paste verbatim at START of every prompt)
```
HARD RULES — MUST BE FOLLOWED OR THE IMAGE IS WRONG:
1. ALL people are MINIMALIST STICK FIGURES matching reference 1: round white circle head, thin black stick arms, thin black stick legs (NO pants, NO jeans, NO shorts, NO shoes, NO boots — legs are plain black line same thickness as arms), simple mitten stick hands, flat-color clothing fills only. No photoreal bodies, no musculature, no wrinkles on limbs.
2. MC always wears a FLAT RED SHORT-SLEEVED T-SHIRT (red is his only color).
3. Zombies use the SAME stick-figure body type as MC. Only differences: (a) MILKY WHITE DOT eyes (empty white circles, no black pupil), (b) ragged jagged-edge light-grey shirts (not solid fill), (c) small flat red blood blotches on some (simple splotches, no drips, no wounds drawn, no gore detail). NO grey monster skin, NO skull faces, NO exposed teeth/tongue/muscle/bone, NO claws, NO blade arms, NO detailed fingers.
4. NEVER draw zombies INSIDE the store during Day 1 (except delivery driver inside walk-in cooler). Zombies stay on the outside of glass.
5. Every shot that includes glass MUST state explicitly: CAMERA SIDE (interior or exterior) and include a FOREGROUND ANCHOR on the camera side:
   - If camera is OUTSIDE (exterior POV): foreground = concrete sidewalk + gas pump canopy edge. Zombies' stick feet stand on the OUTSIDE sidewalk. Store interior (shelves, ceiling lights, possibly tiny MC) is seen THROUGH glass behind them.
   - If camera is INSIDE (interior POV): foreground = MC's shoulder/back OR a shelf OR the counter. Zombies are pressed to the FAR SIDE of the glass, seen through it from inside. Their bodies must be BEHIND the glass plane (reflections, slight blur on glass help).
6. Store state tags for the current beat — read from visuals.csv `state` column (GLASS_STATE, POWER_STATE, DOOR_STATE, LOT_STATE) and include them verbatim.
7. NEGATIVE LOCK: no extra random people inside the store, no crashed cars before beat 18, no smoke before beat 18, no broken glass before beat 4.
```

## PER-BEAT PROMPT STRUCTURE
```
[Mandatory opening paragraph above]
BEAT: <slide.moment>
CAMERA: <interior/exterior + shot size + angle>
FOREGROUND ANCHOR: <what's on the camera side to prove POV>
GLASS STATE: <pristine / single-spider-crack / spiderwebbed-all-panes / covered-but-holding / gone>
POWER: <on / flickering / out>
DOOR: <unlocked / deadbolted / deadbolted+barred>
LOT: <empty / one-car / motorcycle-parked / crashed-car-at-pump / horde / dark>
ACTION: <what is happening in one clear sentence>
COMPOSITION: <where is MC? where are zombies? where is focus?>
COLOR: <only red = [exactly which things are red]; everything else B&W inkwash>
NO TEXT, NO LETTERS, NO WORDS IN FRAME.
```

## ZOMBIE ANATOMY RULES (locked)
- Missing limbs: conveyed by EMPTY FLAPPING SHIRT SLEEVE ONLY (no stump, no wound, no blood at shoulder). Just a shirt sleeve with nothing inside it, flapping behind as the zombie runs.
- Jaw injury: conveyed by a WAVY JAGGED BLACK LINE for a mouth (open uneven edge), never exposed teeth/tissue/bones.
- Blood: simple flat RED SPLOTCHES/blotches on shirts, never drippy gore, never wounds painted.
- Eyes: solid WHITE CIRCLE dots (milky), no pupils, no glowing, no demon eyes.
- Legs: ALWAYS thin black stick lines, ALWAYS bare (no pants/shoes/boots of any color, including black).
- Hands: simple mitten/stick hands, 3-4 lines for fingers, never detailed hands, never claws.

## COMMON BUGS TO BLOCK
- "Black pants/jeans/shoes" on any figure → FAIL. Regenerate.
- Zombie hand reaching toward camera on camera side of glass → FAIL (wrong side).
- More than 1 figure inside main store (MC only, until delivery driver reveal in back) → FAIL.
- Crashed/smoking car in lot before beat 18 → FAIL.
- Dark store / lights off before beat 17 → FAIL.
- Photoreal zombie faces (skulls, teeth, eye sockets, wrinkles) → FAIL.
- Blade/claw/tentacle arms → FAIL.
- Door open when it should be locked/barred → FAIL.
- Text, words, signage that isn't a generic OPEN neon → FAIL (no "B&W" on gas pumps, no brand names).
