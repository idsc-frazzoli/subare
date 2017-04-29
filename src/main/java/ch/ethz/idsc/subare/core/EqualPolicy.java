// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public class EqualPolicy implements PolicyInterface {
  final Scalar scalar;

  public EqualPolicy(int actionsLength) {
    scalar = RationalScalar.of(1, actionsLength);
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    return scalar;
  }
}
