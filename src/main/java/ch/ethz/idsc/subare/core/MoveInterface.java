// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface MoveInterface {
  Tensor move(Tensor state, Tensor action);
}
