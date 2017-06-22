// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;

enum GamblerHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gambler gambler) {
    return ActionValueIterations.solve(gambler, DecimalScalar.of(.0001));
  }

  public static DiscreteVs getOptimalVs(Gambler gambler) {
    return ValueIterations.solve(gambler, RealScalar.of(1e-10));
  }

  public static Policy getOptimalPolicy(Gambler gambler) {
    // TODO test for equality of policies from qsa and vs
    ValueIteration vi = new ValueIteration(gambler, gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(gambler, vi.vs());
  }
}
