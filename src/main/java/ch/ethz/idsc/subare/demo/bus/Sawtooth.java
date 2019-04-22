// code by jph
package ch.ethz.idsc.subare.demo.bus;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Abs;
import ch.ethz.idsc.tensor.sca.Mod;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;

/* package */ class Sawtooth implements ScalarUnaryOperator {
  private final Mod mod;

  public Sawtooth(int half_period) {
    mod = Mod.function(RealScalar.of(2 * half_period), RealScalar.of(-half_period));
  }

  @Override
  public Scalar apply(Scalar t) {
    return Abs.of(mod.apply(t));
  }
}
