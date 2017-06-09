// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Deque;
import java.util.LinkedList;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;

public abstract class DequeDigestAdapter implements StepDigest, DequeDigest {
  @Override
  public final void digest(StepInterface stepInterface) {
    Deque<StepInterface> deque = new LinkedList<>();
    deque.add(stepInterface); // deque holds a single step
    digest(deque);
  }
}
