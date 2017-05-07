// code by jph
package ch.ethz.idsc.subare.ch06.windy;

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
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

/** produces results on p.83: */
class WindyGrid implements StandardModel, MonteCarloInterface, EpisodeSupplier {
  static final Tensor START = Tensors.vector(3, 0);
  static final Tensor GOAL = Tensors.vector(3, 7).unmodifiable();
  private static final Tensor WIND = Tensors.vector(0, 0, 0, 1, 1, 1, 2, 2, 1, 0).negate();
  private static final Clip CLIP_X = Clip.function(0, 6);
  private static final Clip CLIP_Y = Clip.function(0, 9);
  Random random = new Random();
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, 7, 10), 1).unmodifiable();
  private final Index statesIndex;
  private final StateActionMap stateActionMap;

  public static WindyGrid createFour() {
    Tensor actions = Tensors.matrix(new Number[][] { //
        { 0, -1 }, //
        { 0, +1 }, //
        { -1, 0 }, //
        { +1, 0 } //
    }).unmodifiable();
    return new WindyGrid(actions);
  }

  public static WindyGrid createKing() {
    Tensor actions = Tensors.matrix(new Number[][] { //
        { 0, -1 }, //
        { 0, +1 }, //
        { -1, 0 }, //
        { +1, 0 }, //
        // ---
        { +1, -1 }, //
        { +1, +1 }, //
        { -1, -1 }, //
        { -1, +1 } //
    }).unmodifiable();
    return new WindyGrid(actions);
  }

  public WindyGrid(Tensor actions) {
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
    return RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    // wind is added first
    Tensor next = state.copy();
    int y = next.Get(1).number().intValue();
    next.set(scalar -> scalar.add(WIND.Get(y)), 0); // shift in x coordinate
    next.set(CLIP_X, 0);
    next = next.add(action);
    next.set(CLIP_X, 0);
    next.set(CLIP_Y, 1);
    return next;
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
