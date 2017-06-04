// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Norm;

public enum DiscreteVss {
  ;
  // ---
  public static Scalar distance(DiscreteVs qsa1, DiscreteVs qsa2) {
    return Norm._1.of(_difference(qsa1, qsa2));
  }

  private static boolean _isCompatible(DiscreteVs qsa1, DiscreteVs qsa2) {
    return qsa1.index.keys().equals(qsa2.index.keys());
  }

  private static Tensor _difference(DiscreteVs qsa1, DiscreteVs qsa2) {
    if (!_isCompatible(qsa1, qsa2))
      throw new RuntimeException();
    return qsa1.values.subtract(qsa2.values);
  }
}
