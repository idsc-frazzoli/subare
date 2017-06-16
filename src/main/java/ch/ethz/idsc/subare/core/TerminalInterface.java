// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface TerminalInterface {
  /** Remark:
   * the list of possible actions from a terminal state should have length == 1
   * 
   * @param state
   * @return true if state is terminal state, in which case the episode ends */
  boolean isTerminal(Tensor state);
}
