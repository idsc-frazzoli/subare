// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.red.Norm;
import ch.ethz.idsc.tensor.sca.LogisticSigmoid;

public enum DiscreteQsas {
  ;
  // ---
  public static DiscreteQsa rescaled(DiscreteQsa qsa) {
    return qsa.create(Rescale.of(qsa.values).flatten(0));
  }

  public static Scalar distance(DiscreteQsa qsa1, DiscreteQsa qsa2) {
    return Norm._1.of(_difference(qsa1, qsa2));
  }

  public static DiscreteQsa logisticDifference(DiscreteQsa qsa1, DiscreteQsa qsa2, Scalar factor) {
    return qsa1.create(LogisticSigmoid.of(_difference(qsa1, qsa2).multiply(factor)).flatten(0));
  }

  // helper function
  private static boolean _isCompatible(DiscreteQsa qsa1, DiscreteQsa qsa2) {
    return qsa1.index.keys().equals(qsa2.index.keys());
  }

  // helper function
  private static Tensor _difference(DiscreteQsa qsa1, DiscreteQsa qsa2) {
    if (!_isCompatible(qsa1, qsa2))
      throw new RuntimeException();
    return qsa1.values.subtract(qsa2.values);
  }
}
