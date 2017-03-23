// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.UnitVector;

/** an agent that always produces the same predefined action */
public class ConstantAgent extends Agent {
  final int n;
  final int action;

  public ConstantAgent(int n, int action) {
    this.n = n;
    this.action = action;
  }

  @Override
  public int takeAction() {
    return action;
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    // ---
  }

  @Override
  public String getDescription() {
    return "A=" + action;
  }

  @Override
  protected Tensor protected_QValues() {
    return UnitVector.of(action, n);
  }
}
