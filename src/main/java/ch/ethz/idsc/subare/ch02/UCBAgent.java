// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** Section 2.6 "upper-confidence-bound action selection" */
public class UCBAgent extends Agent {
  final Scalar c;
  final Tensor Na;
  final Tensor Qt;

  public UCBAgent(int n, Scalar c) {
    this.c = c;
    Na = Array.zeros(n);
    Qt = Array.zeros(n);
  }

  @Override
  public int takeAction() {
    // (2.8)
    final int t = (Integer) getCount().number();
    Scalar logt = t == 0 ? ZeroScalar.get() : Log.function.apply(getCount());
    Tensor bias = Na.map(v -> //
    Sqrt.function.apply(logt.divide( //
        v.equals(ZeroScalar.get()) ? RealScalar.of(1) : v)));
    Tensor dec = Qt.add(bias.multiply(c));
    // System.out.println(dec);
    return FairArgMax.of(dec);
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(RealScalar.of(1)), a);
    // two possibilities
    // Qt.data[a] += (value - Qt.data[a]) / Na.data[a];
    // Qt.data[a] += (value - Qt.data[a]) / getCount();
    Qt.set(QA -> QA.add( //
        r.subtract(QA).divide(Na.Get(a)) //
    ), a);
  }

  @Override
  public String getDescription() {
    // System.out.println("Qt=" + Qt);
    return "c=" + c;
  }
}
