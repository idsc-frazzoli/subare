// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

/** Example 3.8 p.64: Gridworld
 * 
 * continuous task */
class Gridworld extends DeterministicStandardModel implements MonteCarloInterface {
  private static final Tensor WARP1_ANTE = Tensors.vector(0, 1); // A
  private static final Tensor WARP1_POST = Tensors.vector(4, 1); // A'
  private static final Tensor WARP2_ANTE = Tensors.vector(0, 3); // B
  private static final Tensor WARP2_POST = Tensors.vector(2, 3); // B'
  private static final Clip CLIP = Clip.function(0, 4);
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, 5, 5), 1).unmodifiable();
  final Tensor actions = Tensors.matrix(new Number[][] { //
      { 0, -1 }, //
      { 0, +1 }, //
      { -1, 0 }, //
      { +1, 0 } //
  }).unmodifiable();

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
    return DoubleScalar.of(.9);
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (state.equals(WARP1_ANTE))
      return RealScalar.of(10);
    if (state.equals(WARP2_ANTE))
      return RealScalar.of(5);
    // check if action would take agent off the board
    Tensor effective = state.add(action);
    return effective.map(CLIP).equals(effective) ? //
        RealScalar.ZERO : RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(WARP1_ANTE))
      return WARP1_POST;
    if (state.equals(WARP2_ANTE))
      return WARP2_POST;
    return state.add(action).map(CLIP);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return states;
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return false;
  }
}
