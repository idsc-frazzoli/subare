// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** stochastic approximation theory
 * p.35 equation (2.7)
 * 
 * conditions required for convergence with probability 1:
 * sum_n alpha_n(s,a)^1 == infinity
 * sum_n alpha_n(s,a)^2 < infinity
 * 
 * Example:
 * in the Gambler problem the following values seem to work well
 * {@link OriginalSarsa} factor == 1.3, and exponent == 0.51
 * {@link QLearning} factor == 0.2, and exponent == 0.55 */
public class DefaultLearningRate extends DecayedLearningRate {
  /** @param factor positive, larger values result in larger alpha's
   * @param exponent greater than 1/2, larger values result in smaller alpha's
   * @return */
  public static LearningRate of(Scalar factor, Scalar exponent) {
    return new DefaultLearningRate(factor, exponent);
  }

  /** @param factor
   * @param exponent
   * @return */
  public static LearningRate of(double factor, double exponent) {
    return of(RealScalar.of(factor), RealScalar.of(exponent));
  }

  private DefaultLearningRate(Scalar factor, Scalar exponent) {
    super(factor, exponent);
  }

  @Override
  Tensor key(StepInterface stepInterface) {
    return StateAction.key(stepInterface);
  }
}
