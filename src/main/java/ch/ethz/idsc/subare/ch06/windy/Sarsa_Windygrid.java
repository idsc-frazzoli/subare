// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy */
class Sarsa_Windygrid {
  public static void main(String[] args) {
    Windygrid windyGrid = Windygrid.createFour();
    PolicyInterface policy = new EquiprobablePolicy(windyGrid);
    DiscreteQsa qsa = DiscreteQsa.build(windyGrid);
    System.out.println(qsa.size());
    for (int c = 0; c < 10; ++c) {
      System.out.println(c);
      Sarsa sarsa = new OriginalSarsa( //
          windyGrid, policy, qsa, RealScalar.of(.1));
      // sarsa.simulate(5); // FIXME
      policy = EGreedyPolicy.bestEquiprobable(windyGrid, qsa, RealScalar.of(.1));
      // policy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, qsa); //
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
