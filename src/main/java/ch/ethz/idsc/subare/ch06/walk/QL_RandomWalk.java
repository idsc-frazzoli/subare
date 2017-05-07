// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy
 * 
 * {0, 0} 0
 * {1, -1} 0
 * {1, 1} 1.00
 * {2, -1} 1.00
 * {2, 1} 1.00
 * {3, -1} 1.00
 * {3, 1} 1.00
 * {4, -1} 1.00
 * {4, 1} 1.00
 * {5, -1} 1.00
 * {5, 1} 1.00
 * {6, 0} 0 */
class QL_RandomWalk {
  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    DiscreteQsa qsa = DiscreteQsa.build(randomWalk);
    QLearning qLearning = new QLearning( //
        randomWalk, new EquiprobablePolicy(randomWalk), //
        randomWalk, //
        qsa, RealScalar.ONE, RealScalar.of(.1)); // TODO ask jz
    qLearning.simulate(10000);
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
