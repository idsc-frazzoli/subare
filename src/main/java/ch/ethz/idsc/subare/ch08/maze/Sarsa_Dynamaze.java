// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** determines q(s,a) function for equiprobable "random" policy */
class Sarsa_Dynamaze {
  static void handle(SarsaType sarsaType, int nstep, int EPISODES) throws Exception {
    System.out.println(sarsaType);
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Tensor epsilon = Subdivide.of(.2, .01, EPISODES);
    LearningRate learningRate = DefaultLearningRate.of(5, 0.51);
    Sarsa sarsa = sarsaType.supply(dynamaze, qsa, learningRate);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "n" + nstep + "_qsa_" + sarsaType + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      // if (EPISODES - 10 < index)
      Infoline.print(dynamaze, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(dynamaze, qsa, epsilon.Get(index));
      sarsa.supplyPolicy(() -> policy);
      // for (int count = 0; count < 5; ++count)
      ExploringStarts.batch(dynamaze, policy, nstep, sarsa);
      gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 3, 50);
    handle(SarsaType.expected, 2, 50);
    // handle(SarsaType.qlearning, 1, 50);
  }
}
