// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;

/** the code produces the correct state value function
 *
 * Q-function
 * 
 * {0, 0} 0
 * {1, 0} -0.1
 * {1, 1} -0.1
 * {1, 2} -0.1
 * {1, 3} -0.1
 * {1, 4} -0.1
 * {2, -1} -0.1
 * {2, 1} 0
 * {3, 0} 0
 * 
 * V-function
 * 
 * 0 0
 * 1 -0.1
 * 2 0
 * 3 0 */
class AVI_Maxbias {
  public static void main(String[] args) {
    Maxbias maxbias = new Maxbias(5);
    DiscreteQsa qsa = MaxbiasHelper.getOptimalQsa(maxbias);
    qsa.print();
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    vs.print();
  }
}
