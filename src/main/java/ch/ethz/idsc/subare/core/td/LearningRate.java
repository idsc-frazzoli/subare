// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

// TODO rename or split into two interfaces
public interface LearningRate {
  /** @param count
   * @return */
  Scalar learningRate(Tensor state, Tensor action);

  // TODO to bias policy toward exploration if state-action pair has not been visited frequently
  Scalar exploration(Tensor state, Tensor action); // UCB
}
