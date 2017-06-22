// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** StepDigest qsa methods applied to cliff walk */
class Sarsa_Cliffwalk {
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    Tensor epsilon = Subdivide.of(.5, .01, batches);
    Sarsa sarsa = sarsaType.supply(cliffwalk, qsa, DefaultLearningRate.of(5, 0.51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_" + sarsaType + ".gif"), 200);
    for (int index = 0; index < batches; ++index) {
      if (batches - 10 < index)
        Infoline.print(cliffwalk, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, epsilon.Get(index));
      sarsa.supplyPolicy(() -> policy);
      ExploringStarts.batch(cliffwalk, policy, sarsa);
      gsw.append(ImageFormat.of( //
          StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref)));
    }
    gsw.close();
    // qsa.print(Digits._2);
    // System.out.println("---");
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, qsa);
    // EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policy);
    // while (mce.hasNext()) {
    // StepInterface stepInterface = mce.step();
    // Tensor state = stepInterface.prevState();
    // System.out.println(state + " then " + stepInterface.action());
    // }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.original, 30);
    handle(SarsaType.expected, 30);
    handle(SarsaType.qlearning, 30);
  }
}
