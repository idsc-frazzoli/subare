// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.N;
import ch.ethz.idsc.tensor.sca.Power;

class LearningRateScheduler {
  private final double exponent;
  private final double factor;
  private final double epsilon;

  public LearningRateScheduler(double exponent, double factor, double alpha, double epsilon) {
    this.exponent = exponent;
    this.factor = factor;
    this.epsilon = epsilon;
  }

  Scalar getRate(int index) {
    return Power.of(N.of(DoubleScalar.of(1 / (factor * (index + 1)))), exponent);
  }

  public Scalar getEpsilon() {
    return RealScalar.of(epsilon);
  }
}
