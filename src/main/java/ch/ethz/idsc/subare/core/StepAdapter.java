// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** adapter to hold the four entries (s,a,r,s') */
public class StepAdapter implements StepInterface {
  private final Tensor prev;
  private final Tensor action;
  private final Scalar reward;
  private final Tensor stateS;

  public StepAdapter(Tensor prev, Tensor action, Scalar reward, Tensor stateS) {
    this.prev = prev;
    this.action = action;
    this.reward = reward;
    this.stateS = stateS;
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
    return stateS;
  }
}
