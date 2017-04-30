// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

class GridWorld implements StandardModel {
  private static final Tensor TERMINATE1 = Tensors.vector(0, 0); // A
  private static final Tensor TERMINATE2 = Tensors.vector(3, 3); // A'
  private static final Clip CLIP = Clip.function(0, 3);
  // ---
  final Tensor states = Flatten.of(Array.of(Tensors::vector, 4, 4), 1).unmodifiable();
  final Tensor actions = Tensors.matrix(new Number[][] { //
      { 0, -1 }, //
      { 0, +1 }, //
      { -1, 0 }, //
      { +1, 0 } //
  }).unmodifiable();
  final Index statesIndex;
  final Index actionsIndex;

  public GridWorld() {
    statesIndex = Index.build(states);
    actionsIndex = Index.build(actions);
  }

  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  public Scalar reward(Tensor state, Tensor action) {
    if (state.equals(TERMINATE1))
      return ZeroScalar.get();
    if (state.equals(TERMINATE2))
      return ZeroScalar.get();
    return RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(TERMINATE1))
      return TERMINATE1;
    if (state.equals(TERMINATE2))
      return TERMINATE2;
    return state.add(action).map(CLIP);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    Tensor next = move(state, action);
    int nextI = statesIndex.of(next);
    return reward(state, action).add(gvalues.get(nextI));
  }
}
