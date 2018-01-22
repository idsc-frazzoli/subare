// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 7.1
 * 
 * determines state action value function q(s,a).
 * initial policy is irrelevant because each state allows only one action.
 * 
 * <pre>
 * {0, 0} 0.00
 * {1, 0} 0.03
 * {2, 0} 0.10
 * {3, 0} 0.15
 * {4, 0} 0.18
 * {5, 0} 0.22
 * {6, 0} 0.27
 * {7, 0} 0.32
 * {8, 0} 0.37
 * {9, 0} 0.43
 * {10, 0} 0.47
 * {11, 0} 0.53
 * {12, 0} 0.58
 * {13, 0} 0.61
 * {14, 0} 0.67
 * {15, 0} 0.72
 * {16, 0} 0.77
 * {17, 0} 0.84
 * {18, 0} 0.89
 * {19, 0} 0.95
 * {20, 0} 0.00
 * </pre> */
enum SarsaNStep_Randomwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep) {
    System.out.println(sarsaType);
    Randomwalk randomwalk = new Randomwalk(19);
    DiscreteQsa qsa = DiscreteQsa.build(randomwalk);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = sarsaType.supply(randomwalk, qsa, learningRate);
    Policy policy = EquiprobablePolicy.create(randomwalk);
    sarsa.setExplore(RealScalar.of(.2));
    for (int count = 0; count < 1000; ++count)
      ExploringStarts.batch(randomwalk, policy, nstep, sarsa);
    DiscreteUtils.print(qsa, Round._2);
  }

  public static void main(String[] args) {
    for (SarsaType type : SarsaType.values())
      handle(type, 4);
  }
}
