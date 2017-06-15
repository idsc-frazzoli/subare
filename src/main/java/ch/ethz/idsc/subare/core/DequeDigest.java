// code by jph
package ch.ethz.idsc.subare.core;

import java.util.Deque;

/** interface is implemented by n-step temporal difference learning algorithms */
public interface DequeDigest {
  /** typically the implementation uses all {@link StepInterface}s in the deque
   * to update the first state, or state-action of only the first step in the deque
   * 
   * subsequent calls to digest will have the deque with the formally first step removed
   * 
   * @param deque unmodifiable contains a contiguous part of an episode */
  void digest(Deque<StepInterface> deque);
}
