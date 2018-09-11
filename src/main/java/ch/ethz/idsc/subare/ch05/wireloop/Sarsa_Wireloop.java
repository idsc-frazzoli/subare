// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

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
import ch.ethz.idsc.tensor.io.AnimationWriter;

enum Sarsa_Wireloop {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    Tensor epsilon = Subdivide.of(.2, .01, batches);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    Sarsa sarsa = sarsaType.supply(wireloop, DefaultLearningRate.of(3, 0.51), qsa);
    try (AnimationWriter gsw = AnimationWriter.of( //
        UserHome.Pictures(name + "L_qsa_" + sarsaType + "" + nstep + ".gif"), 250)) {
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
        Policy policy = new EGreedyPolicy(wireloop, qsa, epsilon.Get(index));
        // sarsa.supplyPolicy(() -> policy);
        sarsa.setExplore(epsilon.Get(index));
        ExploringStarts.batch(wireloop, policy, nstep, sarsa);
        gsw.append(WireloopHelper.render(wireloopRaster, ref, qsa));
        if (infoline.isLossfree())
          break;
      }
    }
    System.out.println("---");
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING, 1, 20);
    handle(SarsaType.EXPECTED, 1, 20);
  }
}
