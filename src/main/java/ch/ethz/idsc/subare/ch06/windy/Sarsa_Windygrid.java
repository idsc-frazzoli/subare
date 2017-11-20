// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.LearningRate;
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
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** determines q(s,a) function for equiprobable "random" policy */
enum Sarsa_Windygrid {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Tensor epsilon = Subdivide.of(.2, .01, batches);
    LearningRate learningRate = DefaultLearningRate.of(3, 0.51);
    Sarsa sarsa = sarsaType.supply(windygrid, qsa, learningRate);
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("windygrid_qsa_" + sarsaType + ".gif"), 100);
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(windygrid, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(windygrid, qsa, epsilon.Get(index));
      // sarsa.supplyPolicy(() -> policy);
      sarsa.setExplore(epsilon.Get(index));
      for (int count = 0; count < 10; ++count) // because there is only 1 start state
        ExploringStarts.batch(windygrid, policy, sarsa);
      gsw.append(StateActionRasters.qsaLossRef(windygridRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 20);
    // handle(SarsaType.expected, 20);
    handle(SarsaType.QLEARNING, 20);
  }
}
