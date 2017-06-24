// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines state action value function q(s,a).
 * initial policy is irrelevant because each state allows only one action.
 * 
 * {0, 0} 0
 * {1, 0} 0.16
 * {2, 0} 0.35
 * {3, 0} 0.47
 * {4, 0} 0.59
 * {5, 0} 0.79
 * {6, 0} 0 */
class Sarsa_Randomwalk {
  static void handle(SarsaType sarsaType, int nstep) {
    System.out.println(sarsaType);
    Randomwalk randomwalk = new Randomwalk();
    DiscreteQsa qsa = DiscreteQsa.build(randomwalk);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = sarsaType.supply(randomwalk, qsa, learningRate);
    Policy policy = new EquiprobablePolicy(randomwalk);
    sarsa.setExplore(RealScalar.of(.2));
    for (int count = 0; count < 1000; ++count)
      ExploringStarts.batch(randomwalk, policy, nstep, sarsa);
    qsa.print(Round._2);
  }

  public static void main(String[] args) {
    for (SarsaType type : SarsaType.values())
      handle(type, 1);
  }
}
