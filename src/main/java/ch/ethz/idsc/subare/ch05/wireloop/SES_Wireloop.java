// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DequeExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

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
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    Sarsa sarsa = sarsaType.sarsa(wireloop, DefaultLearningRate.of(7, 1.11), qsa, sac, policy);
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(wireloop, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        System.out.println("policy update " + batchIndex());
        return policy;
      }
    };
    try (AnimationWriter animationWriter = new GifAnimationWriter( //
        HomeDirectory.Pictures(name + "L_qsa_" + sarsaType + "" + nstep + ".gif"), 100, TimeUnit.MILLISECONDS)) {
      int index = 0;
      while (exploringStartsStream.batchIndex() < batches) {
        exploringStartsStream.nextEpisode();
        if (index % 50 == 0) {
          Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
          animationWriter.write(WireloopHelper.render(wireloopRaster, ref, qsa));
          if (infoline.isLossfree())
            break;
        }
        ++index;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.qlearning, 1, 3);
    handle(SarsaType.EXPECTED, 1, 3);
  }
}
