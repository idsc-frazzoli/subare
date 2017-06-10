// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** applicable for models that have deterministic move and reward */
public abstract class DeterministicStandardModel implements StandardModel, //
    SampleModel, ActionValueInterface {
  @Override
  public final Scalar qsa(Tensor state, Tensor action, VsInterface vs) {
    // general term in bellman equation:
    // Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
    // simplifies here to
    // 1 * (r + gamma * v_pi(s'))
    Tensor next = move(state, action);
    return reward(state, action, next).add(vs.value(next));
  }

  @Override
  public final Scalar expectedReward(Tensor state, Tensor action) {
    return reward(state, action, move(state, action)); // deterministic reward
  }

  @Override
  public final Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action)); // deterministic transition
  }

  @Override
  public final Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (!move(state, action).equals(next))
      throw new RuntimeException();
    return RealScalar.ONE; // deterministic transition
  }
}
