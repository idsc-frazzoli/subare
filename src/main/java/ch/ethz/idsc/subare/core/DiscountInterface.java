// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;

// TODO use in most models!
public interface DiscountInterface {
  /** @return discount factor */
  Scalar gamma();
}
