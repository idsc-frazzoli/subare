// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.StateActionMap;
import ch.ethz.idsc.subare.util.Index;
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
class CliffWalk implements StandardModel, MonteCarloInterface, EpisodeSupplier {
  static final Tensor START = Tensors.vector(3, 0).unmodifiable();
  static final Tensor GOAL = Tensors.vector(3, 11).unmodifiable();
  private static final Clip CLIP_X = Clip.function(0, 3);
  private static final Clip CLIP_Y = Clip.function(0, 11);
  Random random = new Random();
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, 4, 12), 1).unmodifiable();
  private final Index statesIndex;
  private final StateActionMap stateActionMap;
  private final Tensor actions = Tensors.matrix(new Number[][] { //
      { 0, -1 }, //
      { 0, +1 }, //
      { -1, 0 }, //
      { +1, 0 } //
  }).unmodifiable();

  public CliffWalk() {
    statesIndex = Index.build(states);
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

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    Tensor next = move(state, action);
    int nextI = statesIndex.of(next);
    return reward(state, action, next).add(gvalues.get(nextI));
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (isTerminal(stateS)) // -1 until goal is reached
      return ZeroScalar.get();
    if (stateS.equals(START))
      return RealScalar.of(-100);
    return RealScalar.ONE.negate();
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
    Scalar y = state.Get(1);
    return state.get(0).equals(RealScalar.of(3)) && //
        Scalars.lessThan(ZeroScalar.get(), y) && Scalars.lessThan(y, RealScalar.of(11));
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
