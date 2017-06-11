// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** adapter to hold the four entries (s,a,r,s') */
public class StepAdapter implements StepInterface {
  private final Tensor prev;
  private final Tensor action;
  private final Scalar reward;
  private final Tensor next;

  public StepAdapter(Tensor prev, Tensor action, Scalar reward, Tensor next) {
    this.prev = prev.unmodifiable();
    this.action = action.unmodifiable();
    this.reward = reward; // Scalar is immutable
    this.next = next.unmodifiable();
  }

  @Override
  public Tensor prevState() {
    return prev;
  }

  @Override
  public Tensor action() {
    return action;
  }

  @Override
  public Scalar reward() {
    return reward;
  }

  @Override
  public Tensor nextState() {
    return next;
  }

  @Override
  public String toString() {
    return String.format("(%s, %s) -> r=%s next=%s", prev, action, reward, next);
  }
}
