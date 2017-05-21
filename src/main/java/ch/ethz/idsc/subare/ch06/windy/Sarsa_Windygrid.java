// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy
 * 
 * {0, 0} 0
 * {1, -1} 0
 * {1, 1} 0.42
 * {2, -1} 0.25
 * {2, 1} 0.53
 * {3, -1} 0.35
 * {3, 1} 0.70
 * {4, -1} 0.55
 * {4, 1} 0.82
 * {5, -1} 0.60
 * {5, 1} 1.00
 * {6, 0} 0 */
class Sarsa_Windygrid {
  public static void main(String[] args) {
    Windygrid windyGrid = Windygrid.createFour();
    PolicyInterface policy = new EquiprobablePolicy(windyGrid);
    DiscreteQsa qsa = DiscreteQsa.build(windyGrid);
    System.out.println(qsa.size());
    for (int c = 0; c < 10; ++c) {
      System.out.println(c);
      Sarsa sarsa = new Sarsa( //
          windyGrid, policy, //
          windyGrid, //
          qsa, RealScalar.of(.8), RealScalar.of(.1));
      sarsa.simulate(5);
      policy = EGreedyPolicy.bestEquiprobable(windyGrid, qsa, RealScalar.of(.1));
      // policy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, qsa); //
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
