// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.api.TensorScalarFunction;
import ch.ethz.idsc.tensor.num.Series;
import ch.ethz.idsc.tensor.sca.Clips;

/** provides different implementation for adding the discounted rewards:
 * in case gamma == 1, the rewards are simply added, else the horner scheme is used */
@FunctionalInterface
public interface DiscountFunction extends TensorScalarFunction {
  /** @param gamma in the interval [0, 1]
   * @return */
  static DiscountFunction of(Scalar gamma) {
    if (gamma.equals(RealScalar.ONE))
      return StaticHelper.TOTAL;
    Clips.unit().requireInside(gamma);
    return rewards -> Series.of(rewards).apply(gamma);
  }
}
