// code by jph
package ch.ethz.idsc.subare.core;

import java.io.Serializable;
import java.util.function.Function;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;
import ch.ethz.idsc.tensor.red.Total;

/** provides different implementation for adding the discounted rewards:
 * in case gamma == 1, the rewards are simply added, else the horner scheme is used */
public interface DiscountFunction extends Function<Tensor, Scalar>, Serializable {
  static final DiscountFunction TOTAL = rewards -> Total.of(rewards).Get();

  static DiscountFunction of(Scalar gamma) {
    if (gamma.equals(RealScalar.ONE))
      return TOTAL;
    return rewards -> Multinomial.horner(rewards, gamma);
  }
}
