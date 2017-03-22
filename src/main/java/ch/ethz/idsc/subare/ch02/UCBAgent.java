// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** Section 2.6 "upper-confidence-bound action selection" */
public class UCBAgent extends Agent {
  public static final Scalar ONE = RealScalar.of(1);
  final Scalar c;
  final Tensor Na;
  final Tensor Qt;

  public UCBAgent(int n, Scalar c) {
    this.c = c;
    Na = Array.zeros(n);
    // init of Qt is irrelevant:
    // first update will set value to the actual reward
    Qt = Array.zeros(n);
  }

  @Override
  public int takeAction() {
    return FairArgMax.of(getQBiased());
  }

  private Tensor getQBiased() {
    // rational estimation without bias, except initial bias Q0:
    final Tensor dec = Qt.copy();
    // add bias
    // (2.8)
    for (int a = 0; a < Qt.length(); ++a) {
      Scalar bias;
      Scalar Nta = Na.Get(a);
      if (Nta.equals(ZeroScalar.get())) {
//        bias = RealScalar.of(Double.POSITIVE_INFINITY);
        bias = RealScalar.of(10); // TODO preliminary, only for plotting
      } else {
        GlobalAssert.of(0 < (Integer) getCount().number());
        Scalar logt = Log.function.apply(getCount());
        bias = c.multiply( //
            Sqrt.function.apply(logt.divide(Nta)));
      }
      dec.set(QA -> QA.add(bias), a);
    }
    return dec;
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(ONE), a);
    Qt.set(QA -> QA.add( //
        r.subtract(QA).divide(Na.Get(a)) // <- compensate for init bias
    ), a);
  }

  @Override
  protected Tensor protected_values() {
    return getQBiased();
  }

  @Override
  public String getDescription() {
    // System.out.println("Qt=" + Qt);
    return "c=" + c;
  }
}
