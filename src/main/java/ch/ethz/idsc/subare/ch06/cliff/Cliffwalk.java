// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.subare.core.util.StateActionMap;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 6.6 cliff walking */
class Cliffwalk extends DeterministicStandardModel implements MonteCarloInterface, EpisodeSupplier {
  static final int NX = 20;
  static final int NY = 6;
  static final int MX = NX - 1;
  static final int MY = NY - 1;
  static final Tensor START = Tensors.vector(0, MY).unmodifiable();
  static final Tensor GOAL = Tensors.vector(MX, MY).unmodifiable();
  private static final Clip CLIP_X = Clip.function(0, MX);
  private static final Clip CLIP_Y = Clip.function(0, MY);
  // ---
  // TODO remove cliff states!
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, NX, NY), 1).unmodifiable();
  private final StateActionMap stateActionMap;
  final Tensor actions = Tensors.matrix(new Number[][] { //
      { +1, 0 }, //
      { -1, 0 }, //
      { 0, +1 }, //
      { 0, -1 } //
  }).unmodifiable();

  public Cliffwalk() {
    stateActionMap = StateActionMap.build(this, actions, this);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return stateActionMap.actions(state);
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (isTerminal(stateS))
      return ZeroScalar.get();
    if (stateS.equals(START))
      return RealScalar.of(-10); // walked off cliff
    return RealScalar.ONE.negate(); // -1 until goal is reached
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    Tensor next = state.add(action);
    next.set(CLIP_X, 0);
    next.set(CLIP_Y, 1);
    if (isCliff(next))
      return START;
    return next;
  }

  boolean isCliff(Tensor state) {
    Scalar x = state.Get(0);
    return state.get(1).equals(RealScalar.of(MY)) && //
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
