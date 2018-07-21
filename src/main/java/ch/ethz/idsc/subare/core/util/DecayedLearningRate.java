// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.LearningRate;
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
 * sum_n alpha_n(s,a)^1 == infinity
 * sum_n alpha_n(s,a)^2 < infinity
 * 
 * Example:
 * in the Gambler problem the following values seem to work well
 * OriginalSarsa factor == 1.3, and exponent == 0.51
 * QLearning factor == 0.2, and exponent == 0.55 */
abstract class DecayedLearningRate implements LearningRate, Serializable {
  private final Scalar factor;
  private final Scalar exponent;
  /** the map counts the frequency of the state-action pair */
  private final Map<Tensor, Integer> map = new HashMap<>();
  /** lookup table to speed up computation */
  private final Tensor MEMO = Tensors.vector(1.0); // index == 0 => learning rate == 1

  DecayedLearningRate(Scalar factor, Scalar exponent) {
    if (Scalars.lessEquals(exponent, RationalScalar.HALF))
      throw TensorRuntimeException.of(factor, exponent);
    this.factor = Sign.requirePositive(factor);
    this.exponent = exponent;
  }

  @Override // from LearningRate
  public synchronized final Scalar alpha(StepInterface stepInterface) {
    Tensor key = key(stepInterface.prevState(), stepInterface.action());
    int index = map.containsKey(key) ? map.get(key) : 0;
    while (MEMO.length() <= index)
      MEMO.append(Min.of( // TODO the "+1" in the denominator may not be ideal... perhaps +0.5, or +0 ?
          factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return MEMO.Get(index);
  }

  @Override // from StepDigest
  public synchronized final void digest(StepInterface stepInterface) {
    Tensor key = key(stepInterface.prevState(), stepInterface.action());
    map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
  }

  /** @return */
  public final int maxCount() { // function is not used yet...
    return MEMO.length();
  }

  @Override
  public final boolean encountered(Tensor state, Tensor action) {
    Tensor key = key(state, action);
    return map.containsKey(key);
  }

  /** @param stepInterface
   * @return key for identifying steps that are considered identical for counting */
  protected abstract Tensor key(Tensor prev, Tensor action);
}
