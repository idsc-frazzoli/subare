// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.pdf.Distribution;

class ConstantPolicy implements Policy {
  final Scalar backProb;

  public ConstantPolicy(Scalar backProb) {
    this.backProb = backProb;
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    if (state.equals(RealScalar.ZERO))
      return action.equals(RealScalar.ZERO) //
          ? backProb
          : RealScalar.ONE.subtract(backProb);
    return RealScalar.ONE;
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    throw new UnsupportedOperationException();
  }
}
