package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.RealScalar;

enum GamblerHelper {
  ;
  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler, RealScalar.ONE);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobableGreedy(gambler, vi.vs());
  }
}
