// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Exp;

public class GradientAgent extends Agent {
  final int n;
  final Scalar alpha;
  final Tensor Ht;

  public GradientAgent(int n, Scalar alpha) {
    this.n = n;
    this.alpha = alpha;
    Ht = Array.zeros(n); // initially all values equal, p.38
  }

  private Tensor getPi() {
    // (2.9)
    Tensor exp = Exp.of(Ht);
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
  protected void protected_feedback(final int a, Scalar r) {
    Tensor pi = getPi();
    for (int k = 0; k < n; ++k) {
      final int fk = k;
      Scalar delta = r.subtract(getR_mean()).multiply(alpha);
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
  
  @Override
  protected Tensor protected_QValues() {
    return Ht;
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
