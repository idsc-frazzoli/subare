// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public class ConstantLearningRate implements LearningRate {
  private final Scalar alpha;

  public ConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override
  public Scalar learningRate(Tensor state, Tensor action) {
    return alpha;
  }
}
