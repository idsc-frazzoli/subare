// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.LearningRateWithCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
@SuppressWarnings("serial")
/* package */ class StrictConstantLearningRate extends LearningRateWithCounter implements Serializable {
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
