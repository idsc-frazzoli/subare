// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

class Sarsa_Wireloop {
  static void handle(SarsaType type, int n) throws Exception {
    System.out.println(type);
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.2, .01, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = type.supply(wireloop, qsa, learningRate);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "_qsa_" + type + ".gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index + " " + epsilon.Get(index).map(Round._2));
      Policy policy = EGreedyPolicy.bestEquiprobable(wireloop, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policy);
      ExploringStarts.batch(wireloop, policy, n, sarsa);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.expected, 5);
  }
}
