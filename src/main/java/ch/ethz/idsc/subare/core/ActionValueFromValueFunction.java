// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface ActionValueFromValueFunction {
  Scalar actionValue(Tensor state, Tensor action, Tensor values);
}
