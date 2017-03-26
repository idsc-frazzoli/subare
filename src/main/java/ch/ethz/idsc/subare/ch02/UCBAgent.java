// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** Section 2.6 "upper-confidence-bound action selection" */
public class UCBAgent extends FairMaxAgent {
  public static final Scalar ONE = RealScalar.of(1); // TODO use RealScalar.ONE
  final Scalar c;
  final Tensor Na;
  final Tensor Qt;

  public UCBAgent(int n, Scalar c) {
    this.c = c;
    Na = Array.zeros(n);
    // init of Qt is irrelevant:
    // first update will set value to the actual reward
    Qt = Array.zeros(n); // init bias == 0
  }

  @Override
  protected Tensor getQVector() {
    // rational estimation without bias, except initial bias Q0:
    final Tensor dec = Qt.copy();
    // add bias
    // (2.8)
    for (int a = 0; a < Qt.length(); ++a) {
      final Scalar bias;
      Scalar Nta = Na.Get(a);
      if (Nta.equals(ZeroScalar.get()))
        // if an action hasn't been taken yet, bias towards this action is infinite
        bias = RealScalar.of(Double.POSITIVE_INFINITY); // TODO RealScalar.POSITIVE_INFINITY
      else {
        Scalar count = Total.of(Na).Get();
        GlobalAssert.of(0 < (Integer) count.number());
        Scalar logt = Log.function.apply(count);
        bias = c.multiply( //
            Sqrt.function.apply(logt.divide(Nta)));
      }
      dec.set(QA -> QA.add(bias), a);
    }
    return dec.unmodifiable();
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(ONE), a);
    Qt.set(QA -> QA.add( //
        r.subtract(QA).divide(Na.Get(a)) // <- compensate for init bias == 0
    ), a);
  }

  @Override
  protected Tensor protected_QValues() {
    Tensor dec = getQVector();
    boolean inf = dec.flatten(-1) //
        .filter(t -> Double.isInfinite(((Scalar) t).number().doubleValue())) //
        .findFirst().isPresent();
    return inf ? Qt : dec;
  }

  @Override
  public String getDescription() {
    return "c=" + c;
  }
}
