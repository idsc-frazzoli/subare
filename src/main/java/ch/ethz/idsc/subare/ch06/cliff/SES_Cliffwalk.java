// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DequeExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum SES_Cliffwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    // CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    // Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    try (AnimationWriter animationWriter = new GifAnimationWriter( //
        HomeDirectory.Pictures("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      LearningRate learningRate = DefaultLearningRate.of(7, 0.61);
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(cliffwalk, qsa, sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
      Sarsa sarsa = sarsaType.sarsa(cliffwalk, learningRate, qsa, sac, policy);
      DequeExploringStarts exploringStartsStream = new DequeExploringStarts(cliffwalk, nstep, sarsa) {
        @Override
        public Policy batchPolicy(int batch) {
          System.out.println("batch " + batch);
          return policy;
        }
      };
      int episode = 0;
      while (exploringStartsStream.batchIndex() < batches) {
        exploringStartsStream.nextEpisode();
        // if (episode % 5 == 0)
        {
          Infoline infoline = Infoline.print(cliffwalk, episode, ref, qsa);
          animationWriter.write(StateActionRasters.qsaLossRef(new CliffwalkRaster(cliffwalk), qsa, ref));
          if (infoline.isLossfree())
            break;
        }
        ++episode;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    int nstep = 1;
    // handle(SarsaType.original, nstep, 3);
    // handle(SarsaType.expected, nstep, 3);
    handle(SarsaType.QLEARNING, nstep, 10);
  }
}
