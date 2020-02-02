// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Section 2.5 "Optimistic Initial Values" */
public final class OptimistAgent extends FairArgAgent {
  final Scalar Q0;
  final Tensor Qt;
  final Scalar alpha;

  /** @param n
   * @param Q0 initial value of all actions
   * @param alpha is weight for difference (r-Qa) */
  public OptimistAgent(int n, Scalar Q0, Scalar alpha) {
    this.Q0 = Q0;
    Qt = Tensors.vector(i -> Q0, n);
    this.alpha = alpha;
  }

  @Override
  protected Tensor getQVector() {
    return Qt.unmodifiable();
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // (2.4) with constant StepSize
    Qt.set(QA -> QA.add(r.subtract(QA).multiply(alpha)), a);
  }

  @Override
  protected Tensor protected_QValues() {
    return Qt;
  }

  @Override
  public String getDescription() {
    return "Q0=" + Q0 + " a=" + alpha;
  }
}
