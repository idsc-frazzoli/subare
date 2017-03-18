// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.Scalar;

/** an agent that always produces the same predefined action */
public class ConstantAgent extends Agent {
  final int action;

  public ConstantAgent(int action) {
    this.action = action;
  }

  @Override
  public int takeAction() {
    return action;
  }

  @Override
  void protected_feedReward(int a, Scalar value) {
    // ---
  }

  @Override
  public String getDescription() {
    return "A=" + action;
  }
}
