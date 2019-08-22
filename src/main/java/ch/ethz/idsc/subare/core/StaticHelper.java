// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.red.Total;

enum StaticHelper {
  ;
  public static final DiscountFunction TOTAL = rewards -> (Scalar) Total.of(rewards);
}
