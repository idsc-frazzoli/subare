// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.ScalarQ;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;

public enum StateActionMapper {
  ;
  public static Tensor getMap(Tensor state, Tensor action) {
    if (ScalarQ.of(action))
      return Tensors.of(state, action);
    return Join.of(state, action);
  }
}
