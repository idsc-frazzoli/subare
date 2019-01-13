// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

/** determines q(s,a) function for equiprobable "random" policy */
enum Sarsa_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.3, 0.01));
    LearningRate learningRate = DefaultLearningRate.of(15, 0.51);
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    try (AnimationWriter animationWriter = AnimationWriter.of(HomeDirectory.Pictures(name + "n" + nstep + "_qsa_" + sarsaType + ".gif"), 200)) {
      for (int index = 0; index < batches; ++index) {
        // if (EPISODES - 10 < index)
        Infoline infoline = Infoline.print(dynamaze, index, ref, qsa);
        // sarsa.supplyPolicy(() -> policy);
        // for (int count = 0; count < 5; ++count)
        ExploringStarts.batch(dynamaze, policy, nstep, sarsa);
        animationWriter.append(StateRasters.vs_rescale(dynamazeRaster, qsa));
        if (infoline.isLossfree())
          break;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 3, 50);
    // handle(SarsaType.expected, 2, 50);
    handle(SarsaType.QLEARNING, 1, 50);
  }
}
