// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DequeExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
class SES_Gridworld {
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of( //
        UserHome.Pictures("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 250);
    LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
    Sarsa sarsa = new OriginalSarsa(gridworld, qsa, learningRate);
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        Policy policy = EGreedyPolicy.bestEquiprobable(gridworld, qsa, epsilon.Get(batch));
        // sarsa.supplyPolicy(() -> policy);
        return policy;
      }
    };
    int episode = 0;
    while (exploringStartsStream.batchIndex() < batches) {
      sarsa.setExplore(epsilon.Get(exploringStartsStream.batchIndex()));
      exploringStartsStream.nextEpisode();
      if (episode % 5 == 0) {
        Infoline infoline = Infoline.print(gridworld, episode, ref, qsa);
        gsw.append(ImageFormat.of( //
            StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref)));
        if (infoline.isLossfree())
          break;
      }
      ++episode;
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    int nstep = 1;
    handle(SarsaType.original, nstep, 3);
    handle(SarsaType.expected, nstep, 3);
    handle(SarsaType.qlearning, nstep, 3);
  }
}
