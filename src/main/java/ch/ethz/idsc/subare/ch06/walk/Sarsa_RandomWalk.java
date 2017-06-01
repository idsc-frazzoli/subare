// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
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
  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    PolicyInterface policy = new EquiprobablePolicy(randomWalk);
    DiscreteQsa qsa = DiscreteQsa.build(randomWalk);
    for (int c = 0; c < 10; ++c) {
      Sarsa sarsa = new OriginalSarsa( //
          randomWalk, policy, randomWalk, //
          qsa, RealScalar.of(.1));
      sarsa.simulate(100);
      policy = EGreedyPolicy.bestEquiprobable(randomWalk, qsa, RealScalar.of(.01));
      // policy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, qsa); //
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
