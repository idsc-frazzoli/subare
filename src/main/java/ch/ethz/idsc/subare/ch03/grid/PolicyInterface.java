// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface PolicyInterface {
  Scalar policy(Tensor state, Tensor action);
}
