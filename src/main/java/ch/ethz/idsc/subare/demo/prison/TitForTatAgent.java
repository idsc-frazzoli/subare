// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

class TitForTatAgent extends Agent {
  private int nextAction = 1;

  @Override
  public int protected_takeAction() {
    return nextAction;
  }

  @Override
  protected Tensor protected_QValues() {
    return Tensors.vectorInt(-1, -1);
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    Tensor rew = Training.r2.get(a);
    if (rew.Get(0).equals(value)) {
      nextAction = 0;
      return;
    }
    if (rew.Get(1).equals(value)) {
      nextAction = 1;
      return;
    }
    GlobalAssert.that(false);
  }

  @Override
  public String getDescription() {
    return "";
  }
}
