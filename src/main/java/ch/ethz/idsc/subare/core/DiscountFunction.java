// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.alg.Multinomial;
import ch.ethz.idsc.tensor.opt.TensorScalarFunction;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Clip;

/** provides different implementation for adding the discounted rewards:
 * in case gamma == 1, the rewards are simply added, else the horner scheme is used */
public interface DiscountFunction extends TensorScalarFunction {
  static final DiscountFunction TOTAL = rewards -> Total.of(rewards).Get();

  /** @param gamma in the interval [0, 1]
   * @return */
  static DiscountFunction of(Scalar gamma) {
    if (gamma.equals(RealScalar.ONE))
      return TOTAL;
    Clip.unit().requireInside(gamma);
    return rewards -> Multinomial.horner(rewards, gamma);
  }
}
