// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Arg;
import ch.ethz.idsc.tensor.alg.Array;

/** Section 2.6 "upper-confidence-bound action selection" */
public class UCBAgent extends Agent {
  final RealScalar c;
  final Tensor Na;
  final Tensor Qt;

  public UCBAgent(int n, RealScalar c) {
    this.c = c;
    Na = Array.zeros(n);
    Qt = Array.zeros(n);
  }

  @Override
  public int takeAction() {
    // (2.8)
    final int t = getCount();
    RealScalar logt = DoubleScalar.of(t == 0 ? 0 : Math.log(t));
    Tensor bias = Na.map(v -> ((RealScalar) logt.divide( //
        v.equals(ZeroScalar.get()) ? RealScalar.of(1) : v//
    )).sqrt());
    Tensor dec = Qt.add(bias.multiply(c));
    // System.out.println(dec);
    return Arg.max(dec);
  }

  @Override
  void protected_feedReward(int a, RealScalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(RealScalar.of(1)), a);
    // two possibilities
    // Qt.data[a] += (value - Qt.data[a]) / Na.data[a];
    // Qt.data[a] += (value - Qt.data[a]) / getCount();
    Qt.set(QA -> QA.add( //
        r.minus((Scalar) QA).divide(Na.Get(a)) //
    ), a);
  }

  @Override
  public String getDescription() {
    // System.out.println("Qt=" + Qt);
    return "c=" + c;
  }
}
