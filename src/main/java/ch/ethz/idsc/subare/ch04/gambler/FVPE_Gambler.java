// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.N;

/** FirstVisitPolicyEvaluation of optimal greedy policy */
class FVPE_Gambler {
  public static void main(String[] args) {
    final Gambler gambler = Gambler.createDefault();
    final DiscreteVs ref = GamblerHelper.getOptimalVs(gambler);
    final Policy policy = GreedyPolicy.bestEquiprobable(gambler, ref);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gambler, null);
    for (int count = 0; count < 100; ++count) {
      ExploringStarts.batch(gambler, policy, fvpe);
      DiscreteVs vs = fvpe.vs();
      Scalar diff = DiscreteValueFunctions.distance(vs, ref);
      System.out.println(count + " " + N.of(diff));
    }
  }
}
