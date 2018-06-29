package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.ScalarQ;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;

public class StateActionMapper {
  public static Tensor getMap(Tensor state, Tensor action) {
    if (ScalarQ.of(action))
      return Tensors.of(state, action);
    else
      return Join.of(state, action);
  }
}
