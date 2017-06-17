// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
class Sarsa_Cliffwalk {
  static void handle(SarsaType type, int EPISODES) throws Exception {
    System.out.println(type);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    System.out.println(qsa.size());
    Tensor epsilon = Subdivide.of(.5, .01, EPISODES);
    Sarsa sarsa = type.supply(cliffwalk, qsa, DefaultLearningRate.of(5, 0.51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_" + type + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteValueFunctions.distance(qsa, ref);
      System.out.println(index + " " + error.map(Round._1));
      Policy policy = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, epsilon.Get(index));
      sarsa.setPolicy(policy);
      ExploringStarts.batch(cliffwalk, policy, sarsa);
      gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(cliffwalk, qsa, ref)));
    }
    gsw.close();
    // qsa.print(Digits._2);
    System.out.println("---");
    Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, qsa);
    EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 100);
    handle(SarsaType.qlearning, 40);
  }
}
