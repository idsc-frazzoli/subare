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
    deque.add(stepInterface); // deque holds single step
    digest(deque);
  }
  // /** digest(step) is equivalent to digest of deque with single step */
  // @Override
  // public final void digest(StepInterface stepInterface) {
  // Tensor state0 = stepInterface.prevState();
  // Tensor action0 = stepInterface.action();
  // Scalar reward = stepInterface.reward();
  // Tensor state1 = stepInterface.nextState();
  // Scalar value0 = qsa.value(state0, action0);
  // // ---
  // Scalar value1 = evaluate(state1); // <- call implementation
  // // ---
  // // [reward value1] . [gamma^0 gamma^1]
  // Scalar delta = reward.add(gamma.multiply(value1)).subtract(value0).multiply(alpha);
  // qsa.assign(state0, action0, value0.add(delta));
  // }
}
