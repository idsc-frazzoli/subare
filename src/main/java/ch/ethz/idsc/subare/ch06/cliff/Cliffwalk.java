// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.red.Norm;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 6.6 cliff walking */
class Cliffwalk extends DeterministicStandardModel implements MonteCarloInterface, EpisodeSupplier {
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
  final Tensor actions = Tensors.matrix(new Number[][] { //
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
    CLIP_X = Clip.function(0, MX);
    CLIP_Y = Clip.function(0, MY);
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
    return actions;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (isTerminal(stateS))
      return isTerminal(state) ? ZeroScalar.get() : RealScalar.ONE;
    if (stateS.equals(START) && Scalars.lessThan( //
        RealScalar.ONE, Norm._1.of(state.subtract(stateS))))
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
    return coord.get(1).equals(RealScalar.of(MY)) && //
        Scalars.lessThan(ZeroScalar.get(), x) && Scalars.lessThan(x, RealScalar.of(MX));
  }

  /**************************************************/
  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    return new MonteCarloEpisode(this, policyInterface, START);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(GOAL);
  }
}
