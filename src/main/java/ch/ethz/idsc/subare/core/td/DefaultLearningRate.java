// code by jz and jph
package ch.ethz.idsc.subare.core.td;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.Power;

public class DefaultLearningRate implements LearningRate {
  /** @param factor positive, large for high learning rate
   * @param exponent greater than 1/2 */
  public static LearningRate of(Scalar factor, Scalar exponent) {
    if (Scalars.lessEquals(factor, RealScalar.ZERO))
      throw TensorRuntimeException.of(factor, exponent);
    if (Scalars.lessEquals(exponent, RationalScalar.of(1, 2)))
      throw TensorRuntimeException.of(factor, exponent);
    return new DefaultLearningRate(factor, exponent);
  }

  public static LearningRate of(double factor, double exponent) {
    return of(RealScalar.of(factor), RealScalar.of(exponent));
  }

  // ---
  private final Scalar factor;
  private final Scalar exponent;
  private final Map<Tensor, Integer> map = new HashMap<>();

  private DefaultLearningRate(Scalar factor, Scalar exponent) {
    this.factor = factor;
    this.exponent = exponent;
  }

  @Override
  public Scalar learningRate(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    int count = map.containsKey(key) ? map.get(key) : 0;
    map.put(key, count + 1);
    // TODO store results
    Scalar alpha = Min.of( //
        factor.multiply(Power.of(DoubleScalar.of(1.0 / (count + 1.0)), exponent)), //
        RealScalar.ONE);
    // System.out.println(count + " " + alpha);
    return alpha;
  }
}
