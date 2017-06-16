// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;

enum BanditsHelper {
  ;
  static DiscreteQsa getOptimalQsa(Bandits bandits) {
    return ActionValueIterations.solve(bandits, DecimalScalar.of(.0001));
  }
}
