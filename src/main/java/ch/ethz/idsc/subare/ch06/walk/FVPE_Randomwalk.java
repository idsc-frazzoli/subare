// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.sca.Round;

/** first visit policy evaluation determines state values v(s)
 * 
 * 0 0
 * 1 0.16
 * 2 0.30
 * 3 0.47
 * 4 0.64
 * 5 0.84
 * 6 0 */
enum FVPE_Randomwalk {
  ;
  public static void main(String[] args) {
    Randomwalk randomwalk = new Randomwalk();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        randomwalk, null);
    Policy policy = new EquiprobablePolicy(randomwalk);
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(randomwalk, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    DiscreteUtils.print(vs, Round._2);
  }
}
