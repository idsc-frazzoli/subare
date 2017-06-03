// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy */
class Sarsa_Windygrid {
  public static void main(String[] args) {
    Windygrid windygrid = Windygrid.createFour();
    PolicyInterface policyInterface = new EquiprobablePolicy(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    System.out.println(qsa.size());
    for (int c = 0; c < 10; ++c) {
      System.out.println(c);
      Sarsa sarsa = new OriginalSarsa( //
          windygrid, qsa, RealScalar.of(.1), //
          policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStartsBatch.apply(windygrid, sarsa, policyInterface);
      policyInterface = EGreedyPolicy.bestEquiprobable(windygrid, qsa, RealScalar.of(.1));
      // policy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, qsa); //
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
