// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.PrioritizedSweeping;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.StepExploringStarts;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

/** determines q(s, a) function for equiprobable "random" policy */
enum PS_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze2";
    Dynamaze dynamaze;
    // dynamaze = DynamazeHelper.original(name);
    dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(7, 1.01);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping( //
        sarsa, 10, RealScalar.ZERO);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_ps_" + sarsaType + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      StepExploringStarts stepExploringStarts = //
          new StepExploringStarts(dynamaze, prioritizedSweeping) {
            @Override
            public Policy batchPolicy(int batch) {
              return policy;
            }
          };
      while (stepExploringStarts.batchIndex() < batches) {
        Infoline infoline = Infoline.print(dynamaze, stepExploringStarts.batchIndex(), ref, qsa);
        stepExploringStarts.nextEpisode();
        animationWriter.write(StateRasters.qsaLossRef(dynamazeRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 50);
    handle(SarsaType.QLEARNING, 10);
  }
}
