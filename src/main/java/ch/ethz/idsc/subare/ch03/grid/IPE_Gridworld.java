// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** produces results on p.64-65:
 * 
 * {0, 0} 3.3
 * {0, 1} 8.8
 * {0, 2} 4.4
 * {0, 3} 5.3
 * {0, 4} 1.5
 * {1, 0} 1.5
 * {1, 1} 3.0
 * {1, 2} 2.3
 * {1, 3} 1.9
 * {1, 4} 0.5
 * {2, 0} 0.1
 * {2, 1} 0.7
 * {2, 2} 0.7
 * {2, 3} 0.4
 * {2, 4} -0.4
 * {3, 0} -1.0
 * {3, 1} -0.4
 * {3, 2} -0.4
 * {3, 3} -0.6
 * {3, 4} -1.2
 * {4, 0} -1.9
 * {4, 1} -1.3
 * {4, 2} -1.2
 * {4, 3} -1.4
 * {4, 4} -2.0 */
/* package */ enum IPE_Gridworld {
  ;
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        gridworld, EquiprobablePolicy.create(gridworld));
    ipe.until(RealScalar.of(.0001));
    DiscreteUtils.print(ipe.vs(), Round._1);
  }
}
