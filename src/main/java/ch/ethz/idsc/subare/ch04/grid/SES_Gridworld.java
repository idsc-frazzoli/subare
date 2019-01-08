// code by jph
package ch.ethz.idsc.subare.ch04.grid;

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
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum SES_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    try (AnimationWriter gsw = AnimationWriter.of( //
        HomeDirectory.Pictures("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 250)) {
      LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
      Sarsa sarsa = sarsaType.supply(gridworld, learningRate, qsa, sac, policy);
      DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
        @Override
        public Policy batchPolicy(int batch) {
          return policy;
        }
      };
      int episode = 0;
      while (exploringStartsStream.batchIndex() < batches) {
        exploringStartsStream.nextEpisode();
        if (episode % 5 == 0) {
          Infoline infoline = Infoline.print(gridworld, episode, ref, qsa);
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          if (infoline.isLossfree())
            break;
        }
        ++episode;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    int nstep = 1;
    handle(SarsaType.ORIGINAL, nstep, 3);
    handle(SarsaType.EXPECTED, nstep, 3);
    handle(SarsaType.QLEARNING, nstep, 3);
  }
}
