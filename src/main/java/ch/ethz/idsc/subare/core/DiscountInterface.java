// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;

/* package */ interface DiscountInterface {
  /** @return discount factor in the interval [0, 1] */
  Scalar gamma();
}
