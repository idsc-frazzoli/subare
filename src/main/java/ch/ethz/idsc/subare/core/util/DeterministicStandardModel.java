// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

/** applicable for models that have deterministic move and reward */
public abstract class DeterministicStandardModel implements StandardModel, SampleModel {
  @Override
  public final Scalar expectedReward(Tensor state, Tensor action) {
    // reward(s,a,s') == expectedReward(s,a)
    return reward(state, action, move(state, action)); // deterministic reward
  }

  @Override
  public final Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action)); // deterministic transition
  }

  @Override
  public final Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(move(state, action), next);
  }
}
