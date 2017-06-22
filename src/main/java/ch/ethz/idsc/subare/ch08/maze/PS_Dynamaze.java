// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.PrioritizedSweeping;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** determines q(s,a) function for equiprobable "random" policy */
class PS_Dynamaze {
  static void handle(SarsaType sarsaType, int EPISODES) throws Exception {
    System.out.println(sarsaType);
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Tensor epsilon = Subdivide.of(.2, .01, EPISODES);
    LearningRate learningRate = DefaultLearningRate.of(5, 0.51);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping(
        // TabularDynaQ tabularDynaQ = new TabularDynaQ( //
        sarsaType.supply(dynamaze, qsa, learningRate), 10, RealScalar.ZERO);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "_ps_" + sarsaType + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      // if (EPISODES - 10 < index)
      Infoline.print(dynamaze, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(dynamaze, qsa, epsilon.Get(index));
      prioritizedSweeping.setPolicy(policy);
      // for (int count = 0; count < 5; ++count)
      ExploringStarts.batch(dynamaze, policy, prioritizedSweeping);
      gsw.append(ImageFormat.of(StateRasters.vs_rescale(dynamazeRaster, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 3, 50);
    // handle(SarsaType.expected, 2, 50);
    handle(SarsaType.qlearning, 50);
  }
}
