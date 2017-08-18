// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DequeExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;

enum SES_Wireloop {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    Tensor epsilon = Subdivide.of(.1, .01, batches);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    Sarsa sarsa = sarsaType.supply(wireloop, qsa, DefaultLearningRate.of(7, 1.11));
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(wireloop, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        Scalar eps = epsilon.Get(batch);
        System.out.println("policy update " + batchIndex() + " " + eps);
        sarsa.setExplore(eps);
        return EGreedyPolicy.bestEquiprobable(wireloop, qsa, eps);
      }
    };
    AnimationWriter gsw = AnimationWriter.of( //
        UserHome.Pictures(name + "L_qsa_" + sarsaType + "" + nstep + ".gif"), 100);
    int index = 0;
    while (exploringStartsStream.batchIndex() < batches) {
      exploringStartsStream.nextEpisode();
      if (index % 50 == 0) {
        Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
        gsw.append(WireloopHelper.render(wireloopRaster, ref, qsa));
        if (infoline.isLossfree())
          break;
      }
      ++index;
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.qlearning, 1, 3);
    handle(SarsaType.expected, 1, 3);
  }
}
