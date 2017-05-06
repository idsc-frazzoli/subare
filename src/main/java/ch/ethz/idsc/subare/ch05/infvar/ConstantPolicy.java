// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;

class ConstantPolicy implements PolicyInterface {
  final Scalar backProb;

  public ConstantPolicy(Scalar backProb) {
    this.backProb = backProb;
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    if (state.equals(ZeroScalar.get()))
      return action.equals(ZeroScalar.get()) ? //
          backProb : RealScalar.ONE.subtract(backProb);
    return RealScalar.ONE;
  }
}
