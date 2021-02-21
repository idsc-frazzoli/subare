// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world using action value iteration
 * 
 * produces results on p.71:
 * 
 * {0, 0} 22.0
 * {0, 1} 24.4
 * {0, 2} 22.0
 * {0, 3} 19.4
 * {0, 4} 17.5
 * {1, 0} 19.8
 * {1, 1} 22.0
 * {1, 2} 19.8
 * {1, 3} 17.8
 * {1, 4} 16.0
 * {2, 0} 17.8
 * {2, 1} 19.8
 * {2, 2} 17.8
 * {2, 3} 16.0
 * {2, 4} 14.4
 * {3, 0} 16.0
 * {3, 1} 17.8
 * {3, 2} 16.0
 * {3, 3} 14.4
 * {3, 4} 13.0
 * {4, 0} 14.4
 * {4, 1} 16.0
 * {4, 2} 14.4
 * {4, 3} 13.0
 * {4, 4} 11.7 */
/* package */ enum AVI_Gridworld {
  ;
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    avi.untilBelow(RealScalar.of(.0001));
    System.out.println("iterations=" + avi.iterations());
    DiscreteUtils.print(avi.qsa(), Round._1);
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    DiscreteUtils.print(dvs, Round._1);
  }
}
