// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.subare.core.RewardInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public abstract class DeterministicStandardModel implements StandardModel, MoveInterface, RewardInterface {
  @Override
  public final Scalar qsa(Tensor state, Tensor action, VsInterface vs) {
    // general term in bellman equation:
    // Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
    // simplifies here to
    // 1 * (r + gamma * v_pi(s'))
    Tensor next = move(state, action);
    return reward(state, action, next).add(vs.value(next));
  }
}
