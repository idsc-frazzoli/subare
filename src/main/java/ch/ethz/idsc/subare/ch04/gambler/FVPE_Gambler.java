// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.N;

/** FirstVisitPolicyEvaluation of optimal greedy policy */
/* package */ enum FVPE_Gambler {
  ;
  public static void main(String[] args) {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    DiscreteVs ref = GamblerHelper.getOptimalVs(gamblerModel);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, ref, null);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gamblerModel, null);
    for (int count = 0; count < 100; ++count) {
      ExploringStarts.batch(gamblerModel, policy, fvpe);
      DiscreteVs vs = fvpe.vs();
      Scalar diff = DiscreteValueFunctions.distance(vs, ref);
      System.out.println(count + " " + N.DOUBLE.of(diff));
    }
  }
}
