// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;

public interface DiscountInterface {
  /** @return discount factor */
  Scalar gamma();
}
