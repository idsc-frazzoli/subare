// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.FairArg;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.api.ScalarUnaryOperator;

/** The e-greedy agent is described in
 * Section 2.2 "Action-Value Methods" */
public class EGreedyAgent extends Agent {
  final Tensor Na;
  final Tensor Qt;
  final ScalarUnaryOperator eps;
  final String string;

  public EGreedyAgent(int n, ScalarUnaryOperator eps, String string) {
    Na = Array.zeros(n);
    Qt = Array.zeros(n);
    this.eps = eps;
    this.string = string;
  }

  @Override
  public int protected_takeAction() {
    // as described in the algorithm box on p.33
    if (RANDOM.nextDouble() < eps.apply(getCount()).number().doubleValue()) {
      notifyAboutRandomizedDecision();
      return RANDOM.nextInt(Qt.length());
    }
    FairArg fairArgMax = FairArg.max(Qt);
    if (!fairArgMax.isUnique())
      notifyAboutRandomizedDecision();
    return fairArgMax.nextRandomIndex(); // (2.2)
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(RealScalar.of(1)), a);
    Qt.set(QA -> QA.add( //
        r.subtract(QA).divide(Na.Get(a)) //
    ), a);
  }

  @Override
  protected Tensor protected_QValues() {
    return Qt;
  }

  @Override
  public String getDescription() {
    return "a=" + string;
  }
}
