# Case studies

Three events told in full: one the map got right, one it missed, and one where two
receiver networks disagreed. Every number is from the repository (incident pages in
`results/incidents/`, evaluation protocol); each is reproducible with the commands on its
incident page.

## 1. Delhi spoofing, November 2025 and January-February 2026: found, and different from the rules

**Reported.** In the week before 6 Nov 2025, aircraft approaching Delhi (IGI) reported
false GPS positions starting about 60 NM out, while the ILS on runway 10/28 was out of
service; seven flights diverted to Jaipur on the night of 4 Nov, and the government later
confirmed spoofing in Parliament. In January-February 2026 airlines reported 623
spoofing incidents in Delhi airspace, about 10 a day (events EV13, EV16).

**What the map saw.** On 4 Nov 2025 the model flags spoofing in 16 of 251 judged
cell-windows in a 60 NM box around IGI; on none of 7 ordinary days was the box flagged
as much (highest 1%). On the three checked days of 2026 it flags 135 of 1,311, mostly on
15 Jan (127 of 501) and almost none on 1 Feb (1 of 339).

**The evidence is spoofing, not jamming.** In flagged cell-windows, aircraft keep normal
integrity (low NIC 0.05 per cell against 0.06 in normal ones) but show position jumps
(0.97 aircraft per cell against 0.03) and positions that do not match their reported
speed (2.0 against 0.07); in November 2025 also GPS altitudes far from barometric
altitude (4.2 against 0.05). An integrity-only method sees nothing here.

**Model and rules differ.** On the 2026 days the model flags 135 cell-windows, the rules
92: the largest real-data difference between them. Which is right is not yet known; it is
the subject of disagreement set 2 (evaluation protocol), labeled blind.

**Limits.** Delhi is a candidate region, below 3,000 ft the pipeline ignores aircraft
(approach spoofing is not covered), and the reports give an area and a period, not
cell-level truth.

## 2. Kaliningrad, 9 November 2024: missed

**Reported.** The Stanford GPS Lab (Lo et al., ION ITM 2026) first saw spoofing near
Kaliningrad on this day, alongside the regular jamming (event EV05; found in ADS-B data,
so not independent of this project's input).

**What the map saw.** 1 of 69 judged cell-windows in the box flagged as spoofing, under
the fixed threshold of 3: **not found**, by model and rules alike. It is the only miss
among 15 checkable events.

**Why, as far as the data shows.** Coverage was thin in 2024: 3.7 usable aircraft per
judged cell-window in the box, and only 69 judged cell-windows that day. The one flagged
cell-window shows the spoofing signature clearly (GPS altitude far from barometric
altitude in 4 aircraft, position jumps), so the evidence was there in places, but too
sparse to reach 3 cell-windows. Jamming is routine in the same box (36 of 56 ordinary days
flag it at least as much), so aircraft at NIC 0 from jamming may also hide a spoofed
position. Not established: whether a lower threshold, or spoofing evidence counted from
1-2 aircraft, would have found it without raising false alarms elsewhere.

**What changed.** Nothing was tuned on this miss: the threshold was fixed before any
event was scored, and changing it after seeing a miss would make every later event
check meaningless. The miss stays in the published count (14 of 15).

## 3. The Baltic in two receiver networks: a declared check that failed

**Question.** Does OpenSky, a second network with its own receivers and decoding, see
degraded integrity in the cells the map flags? Declared before any OpenSky data was
queried, with a pass rule: OpenSky must confirm at least 70% of the model's jamming
cells (lower 95% bound), and flag at most 10% of its normal ones.

**Result.** Normal cells agree (about 5% flagged). Jamming cells: 82-84% confirmed in the
Black Sea and Punjab, 70% in the Eastern Mediterranean, **45% in the Baltic**; 60%
pooled. The check failed and is published as failed.

**Why.** Near Kaliningrad most jammed aircraft lose GPS entirely and are positioned only
by ground multilateration (MLAT): 93% of the degraded aircraft in the Baltic jamming
cells, against 0-8% elsewhere. OpenSky's position table holds only aircraft that still
broadcast a position, so it cannot count them. A prediction written down before testing
it: counting only aircraft with their own position in both networks, the Baltic should
reach 70%. It reached 73%.

**What remains unresolved.** Both networks hear the same aircraft, so agreement checks
the data handling, not the physics. Baltic jamming rests largely on aircraft another
network's position data cannot confirm or refute. And in calm Central Europe, more than
half of the few jamming flags rest on MLAT-only aircraft too, where losing a GNSS
position may have other causes; the next review set samples those separately.
