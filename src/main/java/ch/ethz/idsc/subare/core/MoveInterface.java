// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

/** optional interface for models where moves/transitions are deterministic */
public interface MoveInterface {
  /** @param state
   * @param action
   * @return new state as consequence of given state and action */
  Tensor move(Tensor state, Tensor action);
}
