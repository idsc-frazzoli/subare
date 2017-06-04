// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.Random;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 4.1, p.82 */
class Gridworld extends DeterministicStandardModel implements MonteCarloInterface {
  final int NX = 4;
  final int NY = 4;
  private static final Tensor TERMINATE1 = Tensors.vector(0, 0); // A
  private static final Tensor TERMINATE2 = Tensors.vector(3, 3); // A'
  private static final Clip CLIP = Clip.function(0, 3);
  Random random = new Random();
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, NX, NY), 1).unmodifiable();
  final Tensor actions = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  final Index statesIndex;

  public Gridworld() {
    statesIndex = Index.build(states);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    return RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    return state.add(action).map(CLIP);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return Tensor.of(states.flatten(0).filter(state -> !isTerminal(state)));
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) || state.equals(TERMINATE2);
  }
}
