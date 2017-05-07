// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy
 * 
 * {0, 0} 0
 * {1, -1} 0
 * {1, 1} 0.42
 * {2, -1} 0.25
 * {2, 1} 0.53
 * {3, -1} 0.35
 * {3, 1} 0.70
 * {4, -1} 0.55
 * {4, 1} 0.82
 * {5, -1} 0.60
 * {5, 1} 1.00
 * {6, 0} 0 */
class Sarsa_RandomWalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    Sarsa sarsa = new Sarsa( //
        randomWalk, new EquiprobablePolicy(randomWalk), RealScalar.of(.1), RealScalar.ONE, randomWalk);
    Tensor result = sarsa.simulate(10000);
    Index statesIndex = sarsa.getQsa();
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND));
    }
  }
}
