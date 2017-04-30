// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

// not yet used
public interface RewardInterface {
  // TODO possibly rename state in next
  Scalar reward(Tensor state, Tensor action);
}
