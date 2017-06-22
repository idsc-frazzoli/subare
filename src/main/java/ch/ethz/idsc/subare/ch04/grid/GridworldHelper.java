// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIterations.solve(gridworld, DecimalScalar.of(.0001));
  }
}
