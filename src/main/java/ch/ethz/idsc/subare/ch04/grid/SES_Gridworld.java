// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsStream;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
class SES_Gridworld {
  static void handle(SarsaType sarsaType, int nstep, int BATCHES) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    Tensor epsilon = Subdivide.of(.1, .01, BATCHES); // used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of( //
        UserHome.Pictures("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 100);
    LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
    Sarsa sarsa = new OriginalSarsa(gridworld, qsa, learningRate);
    ExploringStartsStream exploringStartsStream = new ExploringStartsStream(gridworld, nstep, sarsa) {
      @Override
      public Policy batchPolicy() {
        Policy policy = EGreedyPolicy.bestEquiprobable(gridworld, qsa, epsilon.Get(batchIndex()));
        sarsa.supplyPolicy(() -> policy);
        return policy;
      }
    };
    int index = 0;
    while (exploringStartsStream.batchIndex() < BATCHES) {
      exploringStartsStream.nextEpisode();
      if (index % 5 == 0) {
        Infoline.print(gridworld, index, ref, qsa);
        gsw.append(ImageFormat.of( //
            StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref)));
      }
      ++index;
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    int nstep = 3;
    handle(SarsaType.original, nstep, 3);
    handle(SarsaType.expected, nstep, 3);
    handle(SarsaType.qlearning, nstep, 3);
  }
}