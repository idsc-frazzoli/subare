// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.sca.Clip;

class GridWorld implements StandardModel {
  static final Tensor WARP1_ANTE = Tensors.vector(0, 1);
  static final Tensor WARP1_POST = Tensors.vector(4, 1);
  static final Tensor WARP2_ANTE = Tensors.vector(0, 3);
  static final Tensor WARP2_POST = Tensors.vector(2, 3);
  static final Clip CLIP = Clip.function(0, 4);
  // ---
  Tensor actions = Tensors.matrix(new Number[][] { //
      { 0, -1 }, //
      { 0, +1 }, //
      { -1, 0 }, //
      { +1, 0 } //
  }).unmodifiable();
  Tensor states = Flatten.of(Array.of(Tensors::vector, 5, 5), 1).unmodifiable();

  @Override
  public Scalar reward(Tensor state, Tensor action) {
    if (state.equals(WARP1_ANTE))
      return RealScalar.of(10);
    if (state.equals(WARP2_ANTE))
      return RealScalar.of(5);
    Tensor effective = state.add(action);
    return effective.map(CLIP).equals(effective) ? //
        ZeroScalar.get() : RealScalar.ONE.negate();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(WARP1_ANTE))
      return WARP1_POST;
    if (state.equals(WARP2_ANTE))
      return WARP2_POST;
    return state.add(action).map(CLIP);
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    return RationalScalar.of(1, 4);
  }
}
