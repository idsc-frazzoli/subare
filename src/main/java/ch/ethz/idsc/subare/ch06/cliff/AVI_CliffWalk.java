// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;

/** */
class AVI_CliffWalk {
  public static void main(String[] args) {
    CliffWalk cliffWalk = new CliffWalk();
    ActionValueIteration vi = new ActionValueIteration(cliffWalk, cliffWalk, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(.0001));
    DiscreteVs dvs = DiscreteUtils.createVs(cliffWalk, vi.qsa());
    dvs.print();
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(cliffWalk, vi.qsa());
    greedyPolicy.print(cliffWalk.states());
  }
}
