// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
public class ConstantLearningRate extends UnbiasedLearningRate implements Serializable {
  /** @param alpha
   * @return constant learning rate with factor alpha */
  public static LearningRate of(Scalar alpha) {
    return new ConstantLearningRate(alpha);
  }

  /** @return constant learning rate with factor 1.0,
   * that means the updates have numeric precision */
  public static LearningRate one() {
    return of(RealScalar.of(1.0));
  }

  /** @return constant learning rate with exact factor 1,
   * that means the precision in the updates is preserved */
  public static LearningRate one_exact() {
    return of(RealScalar.ONE);
  }

  // ---
  private final Scalar alpha;

  private ConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override
  protected Tensor key(Tensor prev, Tensor action) {
    return Tensors.of(prev, action);
  }

  @Override
  public Scalar alpha(StepInterface stepInterface) {
    return isEncountered(stepInterface.prevState(), stepInterface.action()) //
        ? alpha
        : RealScalar.ONE;
  }
}
