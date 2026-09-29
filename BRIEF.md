# ir-harness — kickoff brief

- **Problem:** circuit-decimator has only ever been validated sim-against-sim
  (DK solver vs ngspice). Nothing in it has been compared to a measured signal,
  so there is no way to know whether the modeling is right about reality, or
  where it's wrong.
- **Done looks like:** one CLI that generates a stimulus set (exponential sine
  sweep, stepped sines at rising levels, a guitar DI clip), runs it through a
  "DUT adapter", time-aligns and level-matches the response, and emits a
  comparison report between any two DUTs: linear IR and magnitude/phase,
  harmonic distortion versus drive level, and a time-domain residual. Two
  adapters prove it: circuit-decimator's headless renderer, and the 192 kHz
  interface in bare loopback. When the JMP arrives it's a third adapter, not a
  rewrite.
- **Not now:** Rigol / SCPI instrument control (the gear isn't USB; design a
  slot, don't fill it). Motorized knobs. NAM training or comparison. A JMP
  circuit model. Psychoacoustics and mix analysis. Mic'd cab captures. All of
  these plug into the same adapter and report once the loop closes on one box.
- **First slice:** the DUT adapter interface, the sim adapter wrapping
  circuit-decimator's renderer, the exponential sweep with Farina deconvolution
  into linear plus harmonic IRs, and a first report comparing a sim render
  against itself with one knob moved.
- **Open question:** sweeps characterize the linear part and the low-order
  harmonics at one level. What makes a JMP a JMP is level-dependent and
  stateful (sag, bias shift, recovery), and it's not obvious that sweeps plus
  stepped sines plus a DI clip yield an error metric that tracks what you hear.
  If the harness measures the wrong thing, sim-vs-reality "agreement" is
  meaningless. Practical: a load box and a reamp box are needed before the JMP
  can be a DUT at line level.

Later, not now: circuit-decimator already has a tube stage and a single-ended
output transformer stage in its netlist catalog, so a JMP model would be
assembled from parts that exist rather than started from scratch.
