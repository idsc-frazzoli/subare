// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface LearningRate {
  /** @param count
   * @return */
  Scalar learningRate(Tensor state, Tensor action);
}
