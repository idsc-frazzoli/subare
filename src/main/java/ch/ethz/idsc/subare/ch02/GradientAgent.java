// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.opt.SoftmaxLayer;
import ch.ethz.idsc.tensor.sca.Chop;

public class GradientAgent extends Agent {
  final int n;
  final Scalar alpha;
  private final Tensor Ht;

  public GradientAgent(int n, Scalar alpha) {
    this.n = n;
    this.alpha = alpha;
    Ht = Array.zeros(n); // initially all values equal, p.38
  }

  @Override
  public int protected_takeAction() {
    Tensor pi = SoftmaxLayer.of(Ht);
    final double rnd = random.nextDouble(); // value in [0,1)
    notifyAboutRandomizedDecision();
    double sum = 0;
    Integer a = null;
    for (int k = 0; k < n; ++k) {
      sum += pi.Get(k).number().doubleValue();
      if (rnd < sum && a == null)
        a = k;
    }
    Scalar zer = Chop.function.apply(RealScalar.of(1 - sum));
    if (!zer.equals(RealScalar.ZERO))
      throw new RuntimeException();
    return a;
  }

  @Override
  protected void protected_feedback(final int a, Scalar r) {
    Tensor pi = SoftmaxLayer.of(Ht);
    for (int k = 0; k < n; ++k) {
      Scalar delta = r.subtract(getRewardAverage());
      // (2.10)
      Scalar pa = pi.Get(a);
      Scalar prob = k == a ? //
          RealScalar.of(1).subtract(pa) : // 1 - pi(At)
          pa.negate(); // - pi(At)
      Ht.set(HA -> HA.add(alpha.multiply(delta).multiply(prob)), k);
    }
  }

  @Override
  protected Tensor protected_QValues() {
    return Ht;
  }

  @Override
  public String getDescription() {
    return "a=" + alpha;
  }
}
