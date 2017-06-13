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
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.Power;

/** for {@link OriginalSarsa} on the Gamber problem the values
 * factor == 1.3, and exponent == 0.51 seem to work well
 * 
 * for {@link QLearning} on the Gamber problem the values
 * factor == 0.2, and exponent == 0.55 seem to work well
 * 
 * TODO comment */
public class DefaultLearningRate implements LearningRate {
  /** @param factor positive, large for high learning rate
   * @param exponent greater than 1/2
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
  private final Map<Tensor, Integer> map = new HashMap<>();
  /** for index == 0 the learning rate == 1 to prevent initialization bias */
  private final Tensor values = Tensors.vector(1.0);

  private DefaultLearningRate(Scalar factor, Scalar exponent) {
    this.factor = factor;
    this.exponent = exponent;
  }

  @Override
  public Scalar learningRate(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    final int index = map.containsKey(key) ? map.get(key) : 0;
    map.put(key, index + 1);
    while (values.length() <= index)
      values.append(Min.of( //
          factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return values.Get(index);
  }

  @Override
  public Scalar exploration(Tensor state, Tensor action) {
    // TODO Auto-generated method stub
    return null;
  }
}
