// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Total;

public class GradientAgent extends Agent {
  final int n;
  final RealScalar alpha;
  final Tensor Ht;

  public GradientAgent(int n, RealScalar alpha) {
    this.n = n;
    this.alpha = alpha;
    Ht = Array.zeros(n); // initially all values equal, p.38
  }

  private Tensor getPi() {
    // (2.9)
    Tensor exp = Tensor.of(Ht.flatten(0) //
        .map(Scalar.class::cast) //
        .map(x -> DoubleScalar.of(Math.exp(x.number().doubleValue()))));
    return exp.multiply(((Scalar) Total.of(exp)).invert()).unmodifiable();
  }

  @Override
  public int takeAction() {
    Tensor pi = getPi();
    double sum = 0;
    double rnd = random.nextDouble();
    for (int k = 0; k < n; ++k) {
      sum += pi.Get(k).abs().number().doubleValue(); // TODO why abs?
      if (rnd < sum)
        return k;
    }
    throw new RuntimeException();
  }

  @Override
  void protected_feedReward(final int a, Scalar r) {
    Tensor pi = getPi();
    for (int k = 0; k < n; ++k) {
      final int fk = k;
      RealScalar delta = (RealScalar) r.subtract(getR_mean()).multiply(alpha);
      // (2.10)
      if (k == a) {
        Ht.set(x -> x.add( //
            RealScalar.of(1).subtract(pi.Get(fk)).multiply(delta) //
        ), k);
      } else {
        Ht.set(x -> x.subtract( //
            pi.Get(fk).multiply(delta) //
        ), k);
      }
    }
  }

  /** @return average of all the rewards up through and including time t */
  private Scalar getR_mean() {
    return getTotal().divide(getCount());
  }

  @Override
  public String getDescription() {
    return "a=" + alpha;
  }
}
