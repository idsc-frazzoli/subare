// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
public class ConstantExplorationRate implements ExplorationRate {
  /** @param alpha
   * @return constant learning rate with factor alpha */
  public static ConstantExplorationRate of(double epsilon) {
    return new ConstantExplorationRate(RealScalar.of(epsilon));
  }

  // ---
  private Scalar epsilon;

  private ConstantExplorationRate(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  @Override
  public Scalar epsilon(Tensor state, StateActionCounter stateActionCounter) {
    return epsilon;
  }

  public void setEpsilon(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  public Scalar getEpsilon() {
    return epsilon;
  }
}
