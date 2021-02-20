// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.Power;
import ch.ethz.idsc.tensor.sca.Sign;

/** stochastic approximation theory
 * p.35 equation (2.7)
 * 
 * conditions required for convergence with probability 1:
 * sum_n alpha_n(s, a)^1 == infinity
 * sum_n alpha_n(s, a)^2 < infinity
 * 
 * Example:
 * in the Gambler problem the following values seem to work well
 * OriginalSarsa factor == 1.3, and exponent == 0.51
 * QLearning factor == 0.2, and exponent == 0.55 */
abstract class DecayedLearningRate implements LearningRate {
  private final Scalar factor;
  private final Scalar exponent;
  /** lookup table to speed up computation */
  private final Tensor memo = Tensors.vector(1.0); // index == 0 => learning rate == 1

  /* package */ DecayedLearningRate(Scalar factor, Scalar exponent) {
    if (Scalars.lessEquals(exponent, RationalScalar.HALF))
      throw TensorRuntimeException.of(factor, exponent);
    this.factor = Sign.requirePositive(factor);
    this.exponent = exponent;
  }

  @Override // from LearningRate
  public synchronized final Scalar alpha(StepInterface stepInterface, StateActionCounter stateActionCounter) {
    Tensor key = StateAction.key(stepInterface);
    int index = Scalars.intValueExact(stateActionCounter.stateActionCount(key));
    while (memo.length() <= index)
      memo.append(Min.of( // TODO the "+1" in the denominator may not be ideal... perhaps +0.5, or +0 ?
          factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return memo.Get(index);
  }

  /** @return */
  final int maxCount() { // function is not used yet...
    return memo.length();
  }
}
