// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DequeExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum SES_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    try (AnimationWriter gsw = AnimationWriter.of( //
        UserHome.Pictures("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 250)) {
      LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
      Sarsa sarsa = sarsaType.supply(gridworld, qsa, learningRate);
      DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
        @Override
        public Policy batchPolicy(int batch) {
          Scalar eps = epsilon.Get(batch);
          sarsa.setExplore(eps);
          return EGreedyPolicy.bestEquiprobable(gridworld, qsa, eps);
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
