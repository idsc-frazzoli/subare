// code by jph
package ch.ethz.idsc.subare.core.adapter;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.StepInterface;

public abstract class DequeDigestAdapter implements DequeDigest {
  @Override
  public final void digest(StepInterface stepInterface) {
    Deque<StepInterface> deque = new ArrayDeque<>();
    deque.add(stepInterface); // deque holds a single step
    digest(deque);
  }
}
