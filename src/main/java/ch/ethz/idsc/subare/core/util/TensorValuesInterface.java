// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.stream.Stream;

import ch.ethz.idsc.tensor.Tensor;

public interface TensorValuesInterface {
  Tensor keys();

  /** @return unmodifiable tensor of (state)-, or (state, action)-values */
  Tensor values();

  TensorValuesInterface create(Stream<? extends Tensor> stream);
}
