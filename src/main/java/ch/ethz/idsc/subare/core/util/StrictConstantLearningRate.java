// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
@SuppressWarnings("serial")
/* package */ class StrictConstantLearningRate extends LearningRate implements Serializable {
  private final Set<Tensor> visited = new HashSet<>();
  private final Scalar alpha;

  protected StrictConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface) {
    return alpha;
  }

  @Override
  protected Tensor key(Tensor prev, Tensor action) {
    return StateAction.key(prev, action);
  }
}
