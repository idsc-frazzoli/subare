// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.nrm.Vector1Norm;
import ch.ethz.idsc.tensor.num.Boole;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Clips;
import ch.ethz.idsc.tensor.sca.Sign;

/** Example 6.6 p. 132, cliff walking */
public class Cliffwalk extends DeterministicStandardModel implements MonteCarloInterface {
  static final Scalar PRICE_CLIFF = RealScalar.of(-20);
  static final Scalar PRICE_MOVE = RealScalar.ONE.negate();
  // ---
  final int NX;
  final int NY;
  final int MX;
  final int MY;
  final Tensor START;
  final Tensor GOAL;
  final Clip CLIP_X;
  final Clip CLIP_Y;
  // ---
  private final Tensor states;
  static final Tensor ACTIONS = Tensors.matrix(new Number[][] { //
      { +1, 0 }, //
      { -1, 0 }, //
      { 0, +1 }, //
      { 0, -1 } //
  }).unmodifiable();

  /** @param NX
   * @param NY */
  public Cliffwalk(int NX, int NY) {
    this.NX = NX;
    this.NY = NY;
    MX = NX - 1;
    MY = NY - 1;
    START = Tensors.vector(0, MY).unmodifiable();
    GOAL = Tensors.vector(MX, MY).unmodifiable();
    CLIP_X = Clips.positive(MX);
    CLIP_Y = Clips.positive(MY);
    Tensor pre = Tensors.empty();
    for (Tensor coord : Flatten.of(Array.of(Tensors::vector, NX, NY), 1))
      if (!isCliff(coord))
        pre.append(coord);
    states = pre.unmodifiable();
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return ACTIONS;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(next))
      return Boole.of(!isTerminal(state));
    if (next.equals(START) && Scalars.lessThan( //
        RealScalar.ONE, Vector1Norm.between(state, next)))
      return PRICE_CLIFF; // walked off cliff
    return PRICE_MOVE; // -1 until goal is reached
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return GOAL;
    Tensor next = state.add(action);
    next.set(CLIP_X, 0);
    next.set(CLIP_Y, 1);
    if (isCliff(next))
      return START;
    return next;
  }

  boolean isCliff(Tensor coord) {
    Scalar x = coord.Get(0);
    return coord.Get(1).equals(RealScalar.of(MY)) && //
        Sign.isPositive(x) && Scalars.lessThan(x, RealScalar.of(MX));
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(START);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(GOAL);
  }
}
