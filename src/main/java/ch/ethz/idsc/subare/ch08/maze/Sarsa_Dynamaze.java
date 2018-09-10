// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;

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
    Tensor epsilon = Subdivide.of(.3, .01, batches);
    LearningRate learningRate = DefaultLearningRate.of(15, 0.51);
    Sarsa sarsa = sarsaType.supply(dynamaze, learningRate, qsa);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures(name + "n" + nstep + "_qsa_" + sarsaType + ".gif"), 200)) {
      for (int index = 0; index < batches; ++index) {
        // if (EPISODES - 10 < index)
        Infoline infoline = Infoline.print(dynamaze, index, ref, qsa);
        Policy policy = EGreedyPolicy.bestEquiprobable(dynamaze, qsa, epsilon.Get(index));
        // sarsa.supplyPolicy(() -> policy);
        sarsa.setExplore(epsilon.Get(index));
        // for (int count = 0; count < 5; ++count)
        ExploringStarts.batch(dynamaze, policy, nstep, sarsa);
        gsw.append(StateRasters.vs_rescale(dynamazeRaster, qsa));
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
