// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/** adaptive learning rate for states but independent of the action taken
 * 
 * see documentation of {@link DecayedLearningRate}
 * 
 * conditions required for convergence with probability 1:
 * sum_n alpha_n(s)^1 == infinity
 * sum_n alpha_n(s)^2 < infinity */
public class StateLearningRate extends DecayedLearningRate {
  /** @param factor positive, larger values result in larger alpha's
   * @param exponent greater than 1/2, larger values result in smaller alpha's
   * @return */
  public static LearningRate of(Scalar factor, Scalar exponent) {
    return new StateLearningRate(factor, exponent);
  }

  /** @param factor
   * @param exponent
   * @return */
  public static LearningRate of(Number factor, Number exponent) {
    return of(RealScalar.of(factor), RealScalar.of(exponent));
  }

  /***************************************************/
  private StateLearningRate(Scalar factor, Scalar exponent) {
    super(factor, exponent);
  }
}
