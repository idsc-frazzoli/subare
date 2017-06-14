// code by jz and jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.QLearning;
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
public class DefaultLearningRate implements LearningRate {
  /** @param factor positive, larger values result in larger alpha's
   * @param exponent greater than 1/2, larger values result in smaller alpha's
   * @return */
  public static LearningRate of(Scalar factor, Scalar exponent) {
    if (Scalars.lessEquals(factor, RealScalar.ZERO))
      throw TensorRuntimeException.of(factor, exponent);
    if (Scalars.lessEquals(exponent, RationalScalar.of(1, 2)))
      throw TensorRuntimeException.of(factor, exponent);
    return new DefaultLearningRate(factor, exponent);
  }

  /** @param factor
   * @param exponent
   * @return */
  public static LearningRate of(double factor, double exponent) {
    return of(RealScalar.of(factor), RealScalar.of(exponent));
  }

  // ---
  private final Scalar factor;
  private final Scalar exponent;
  /** the map counts the frequency of the state-action pair */
  private final Map<Tensor, Integer> map = new HashMap<>();
  /** lookup table to speed up computation */
  private final Tensor MEMO = Tensors.vector(1.0); // index == 0 => learning rate == 1

  private DefaultLearningRate(Scalar factor, Scalar exponent) {
    this.factor = factor;
    this.exponent = exponent;
  }

  @Override // from LearningRate
  public Scalar alpha(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    int index = map.containsKey(key) ? map.get(key) : 0;
    while (MEMO.length() <= index)
      MEMO.append(Min.of( //
          factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return MEMO.Get(index);
  }

  @Override // from StepDigest
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Tensor key = DiscreteQsa.createKey(state0, action);
    int index = map.containsKey(key) ? map.get(key) : 0;
    map.put(key, index + 1);
  }

  /** @return */
  public int maxCount() {
    return MEMO.length();
  }
}
