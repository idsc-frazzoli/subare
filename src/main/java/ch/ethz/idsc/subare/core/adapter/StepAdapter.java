// code by jph
package ch.ethz.idsc.subare.core.adapter;

import java.util.Objects;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** adapter to hold the four entries (s, a, r, s') */
public final class StepAdapter implements StepInterface {
  private final Tensor prev;
  private final Tensor action;
  private final Scalar reward;
  private final Tensor next;

  /** none of the input parameters must be null
   * 
   * @param prev state
   * @param action from prev state
   * @param reward obtained subsequent to taking action
   * @param next state */
  public StepAdapter(Tensor prev, Tensor action, Scalar reward, Tensor next) {
    this.prev = prev.unmodifiable();
    this.action = action.unmodifiable();
    this.reward = Objects.requireNonNull(reward); // Scalar is immutable
    this.next = next.unmodifiable();
  }

  @Override // from StepInterface
  public Tensor prevState() {
    return prev;
  }

  @Override // from StepInterface
  public Tensor action() {
    return action;
  }

  @Override // from StepInterface
  public Scalar reward() {
    return reward;
  }

  @Override // from StepInterface
  public Tensor nextState() {
    return next;
  }

  @Override // from Object
  public String toString() {
    return String.format("(%s, %s) -> r=%s next=%s", prev, action, reward, next);
  }
}
