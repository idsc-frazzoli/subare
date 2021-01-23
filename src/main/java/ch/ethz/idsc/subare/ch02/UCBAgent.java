// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.DeterminateScalarQ;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.IntegerQ;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sign;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** Section 2.6 "upper-confidence-bound action selection" */
public class UCBAgent extends FairArgAgent {
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
      if (Scalars.isZero(Nta))
        // if an action hasn't been taken yet, bias towards this action is infinite
        bias = DoubleScalar.POSITIVE_INFINITY;
      else {
        Scalar count = Sign.requirePositive(IntegerQ.require(Total.ofVector(Na)));
        Scalar logt = Log.of(count);
        bias = c.multiply(Sqrt.of(logt.divide(Nta)));
      }
      dec.set(QA -> QA.add(bias), a);
    }
    return dec.unmodifiable();
  }

  @Override
  protected void protected_feedback(int a, Scalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(RealScalar.ONE), a);
    Qt.set(QA -> QA.add( //
        r.subtract(QA).divide(Na.Get(a)) // <- compensate for init bias == 0
    ), a);
  }

  @Override
  protected Tensor protected_QValues() {
    Tensor dec = getQVector();
    return dec.flatten(-1).map(Scalar.class::cast).allMatch(DeterminateScalarQ::of) //
        ? dec
        : Qt;
  }

  @Override
  public String getDescription() {
    return "c=" + c;
  }
}
