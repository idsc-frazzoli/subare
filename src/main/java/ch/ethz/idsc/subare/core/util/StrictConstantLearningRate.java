// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashSet;
import java.util.Set;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
/* package */ class StrictConstantLearningRate implements LearningRate {
  private final Set<Tensor> visited = new HashSet<>();
  private final Scalar alpha;

  protected StrictConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override // from LearningRate
  public final void digest(StepInterface stepInterface) {
    visited.add(StateAction.key(stepInterface));
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface) {
    return alpha;
  }

  @Override // from LearningRate
  public final boolean encountered(Tensor state, Tensor action) {
    return visited.contains(StateAction.key(state, action));
  }
}
