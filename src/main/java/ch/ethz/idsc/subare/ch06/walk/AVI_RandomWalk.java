// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;

class AVI_RandomWalk {
  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    ActionValueIteration vi = new ActionValueIteration(randomWalk, randomWalk, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(.0001));
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, vi.qsa());
    greedyPolicy.print(randomWalk.states());
    System.out.println("q*(s,a)");
    vi.qsa().print();
  }
}
