// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Norm;

public enum DiscreteVss {
  ;
  // ---
  public static Scalar distance(DiscreteVs vs1, DiscreteVs vs2) {
    return Norm._1.of(_difference(vs1, vs2));
  }

  // helper function
  private static boolean _isCompatible(DiscreteVs vs1, DiscreteVs vs2) {
    return vs1.index.keys().equals(vs2.index.keys());
  }

  // helper function
  private static Tensor _difference(DiscreteVs vs1, DiscreteVs vs2) {
    if (!_isCompatible(vs1, vs2))
      throw new RuntimeException();
    return vs1.values.subtract(vs2.values);
  }
}
