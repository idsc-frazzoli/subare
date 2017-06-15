// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 4.1, p.82
 * Figure 4.1, p.83
 * 
 * {0, 0} 0
 * {0, 1} -18.3
 * {0, 2} -27.5
 * {0, 3} -28.3
 * {1, 0} -17.4
 * {1, 1} -22.0
 * {1, 2} -25.0
 * {1, 3} -22.5
 * {2, 0} -22.7
 * {2, 1} -23.7
 * {2, 2} -21.1
 * {2, 3} -16.2
 * {3, 0} -26.1
 * {3, 1} -21.5
 * {3, 2} -13.1
 * {3, 3} 0 */
class FVPE_Gridworld {
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gridworld, null);
    PolicyInterface policyInterface = new EquiprobablePolicy(gridworld);
    ExploringStarts.batch(gridworld, policyInterface, fvpe);
    DiscreteVs vs = fvpe.vs();
    vs.print(Round._1);
  }
}
