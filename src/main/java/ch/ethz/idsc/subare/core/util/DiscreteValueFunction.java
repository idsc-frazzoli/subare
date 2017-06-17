// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.stream.Stream;

import ch.ethz.idsc.tensor.Tensor;

public interface DiscreteValueFunction {
  Tensor keys();

  /** @return unmodifiable tensor of (state)-, or (state, action)-values */
  Tensor values();

  DiscreteValueFunction create(Stream<? extends Tensor> stream);
}
