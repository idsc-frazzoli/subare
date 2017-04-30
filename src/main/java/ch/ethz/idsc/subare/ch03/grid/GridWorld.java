// code by jph
package ch.ethz.idsc.subare.ch03.grid;

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
  private static final Tensor WARP1_ANTE = Tensors.vector(0, 1); // A
  private static final Tensor WARP1_POST = Tensors.vector(4, 1); // A'
  private static final Tensor WARP2_ANTE = Tensors.vector(0, 3); // B
  private static final Tensor WARP2_POST = Tensors.vector(2, 3); // B'
  private static final Clip CLIP = Clip.function(0, 4);
  // ---
  final Tensor states = Flatten.of(Array.of(Tensors::vector, 5, 5), 1).unmodifiable();
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

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  public Scalar reward(Tensor state, Tensor action) {
    if (state.equals(WARP1_ANTE))
      return RealScalar.of(10);
    if (state.equals(WARP2_ANTE))
      return RealScalar.of(5);
    // check if action would take agent off the board
    Tensor effective = state.add(action);
    return effective.map(CLIP).equals(effective) ? //
        ZeroScalar.get() : RealScalar.ONE.negate();
  }

  Tensor move(Tensor state, Tensor action) {
    if (state.equals(WARP1_ANTE))
      return WARP1_POST;
    if (state.equals(WARP2_ANTE))
      return WARP2_POST;
    return state.add(action).map(CLIP);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    // general term in bellman equation:
    // Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
    // simplifies here to
    // 1 * (r + gamma * v_pi(s'))
    Tensor next = move(state, action);
    int nextI = statesIndex.of(next);
    return reward(state, action).add(gvalues.get(nextI));
  }
}
