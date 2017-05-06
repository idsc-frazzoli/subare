// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** the term "equiprobable" appears in Exercise 4.1 */
public class EquiprobablePolicy implements PolicyInterface {
  final Scalar scalar;

  public EquiprobablePolicy(int actionsLength) {
    scalar = RationalScalar.of(1, actionsLength);
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    return scalar;
  }
}
