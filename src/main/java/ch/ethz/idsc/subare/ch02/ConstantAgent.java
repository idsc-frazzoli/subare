// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.RealScalar;

public class ConstantAgent extends Agent {
  final int action;

  public ConstantAgent(int action) {
    this.action = action;
  }

  @Override
  public
  int takeAction() {
    return action;
  }

  @Override
  void protected_feedReward(int a, RealScalar value) {
    // ---
  }

  @Override
  public String getDescription() {
    return "A=" + action;
  }
}
