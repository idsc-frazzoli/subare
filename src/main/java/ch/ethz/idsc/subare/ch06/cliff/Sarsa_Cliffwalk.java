// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
enum Sarsa_Cliffwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk, DoubleScalar.POSITIVE_INFINITY);
    Tensor epsilon = Subdivide.of(.2, .01, batches);
    Sarsa sarsa = sarsaType.supply(cliffwalk, qsa, DefaultLearningRate.of(7, 0.61));
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("cliffwalk_qsa_" + sarsaType + ".gif"), 200)) {
      for (int index = 0; index < batches; ++index) {
        // if (batches - 10 < index)
        Infoline infoline = Infoline.print(cliffwalk, index, ref, qsa);
        Policy policy = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, epsilon.Get(index));
        // sarsa.supplyPolicy(() -> policy);
        sarsa.setExplore(epsilon.Get(index));
        ExploringStarts.batch(cliffwalk, policy, nstep, sarsa);
        gsw.append(StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
    DiscreteUtils.print(qsa, Round._2);
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
    // handle(SarsaType.original, 1, 30);
    // handle(SarsaType.expected, 1, 30);
    handle(SarsaType.QLEARNING, 1, 10);
  }
}
