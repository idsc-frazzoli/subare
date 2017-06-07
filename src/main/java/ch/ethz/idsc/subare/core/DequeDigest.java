// code by jph
package ch.ethz.idsc.subare.core;

import java.util.Deque;

/** interface is implemented by n-step temporal difference learning algorithms */
public interface DequeDigest {
  /** @param deque unmodifiable contains a contiguous part of an episode */
  void digest(Deque<StepInterface> deque);
}
