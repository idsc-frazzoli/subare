// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
public class ConstantLearningRate extends StrictConstantLearningRate {
  /** @param alpha
   * @return constant learning rate with factor alpha */
  public static LearningRate of(Scalar alpha) {
    return new ConstantLearningRate(alpha);
  }

  /** @param alpha
   * @param warmStart whether to warmStart (alpha=1 if state-action pair not yet seen) or not
   * @return constant learning rate with factor alpha */
  public static LearningRate of(Scalar alpha, boolean warmStart) {
    return warmStart //
        ? new ConstantLearningRate(alpha)
        : new StrictConstantLearningRate(alpha);
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
  private ConstantLearningRate(Scalar alpha) {
    super(alpha);
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface) {
    return encountered(stepInterface.prevState(), stepInterface.action()) //
        ? super.alpha(stepInterface)
        : RealScalar.ONE; // overcome initialization bias
  }
}
