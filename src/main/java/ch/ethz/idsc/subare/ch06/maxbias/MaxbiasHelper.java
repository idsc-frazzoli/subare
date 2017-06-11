// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;

public enum MaxbiasHelper {
  ;
  // ---
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIterations.getOptimal(maxbias, DecimalScalar.of(.0001));
  }
}
