// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.FairArg;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Section 2.5 "Optimistic Initial Values" */
public final class OptimistAgent extends Agent {
  final RealScalar Q0;
  Tensor Qt;
  final RealScalar alpha;

  /** @param n
   * @param Q0
   * @param alpha is weight for difference (r-Qa) */
  public OptimistAgent(int n, RealScalar Q0, RealScalar alpha) {
    this.Q0 = Q0;
    Qt = Tensors.vector(i -> Q0, n);
    this.alpha = alpha;
  }

  @Override
  public int takeAction() {
    return FairArg.max(Qt); // (2.2)
  }

  @Override
  void protected_feedReward(int a, Scalar r) {
    // (2.4) with constant StepSize
    Qt.set(QA -> QA.add( //
        r.subtract(QA).multiply(alpha) //
    ), a);
  }

  @Override
  public String getDescription() {
    return "Q0=" + Q0 + " a=" + alpha;
  }
}
