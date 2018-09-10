// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/** adaptive learning rate for state-action pairs
 * 
 * see documentation of {@link DecayedLearningRate} */
public class DefaultLearningRate extends DecayedLearningRate {
  /** @param factor positive, larger values result in larger alpha's
   * @param exponent greater than 1/2, larger values result in smaller alpha's
   * @return */
  public static LearningRate of(Scalar factor, Scalar exponent) {
    return new DefaultLearningRate(factor, exponent);
  }

  /** @param factor positive, larger values result in larger alpha's
   * @param exponent greater than 1/2, larger values result in smaller alpha's
   * @return */
  public static LearningRate of(double factor, double exponent) {
    return of(RealScalar.of(factor), RealScalar.of(exponent));
  }

  // ---
  private DefaultLearningRate(Scalar factor, Scalar exponent) {
    super(factor, exponent);
  }
}
