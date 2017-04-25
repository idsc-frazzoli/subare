// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface RewardInterface {
  Scalar reward(Tensor state, Tensor action);
}
