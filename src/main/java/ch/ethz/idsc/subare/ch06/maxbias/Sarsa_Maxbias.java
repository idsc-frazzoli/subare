// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

enum Sarsa_Maxbias {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println(sarsaType);
    Maxbias maxbias = new Maxbias(3);
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    int batches = 100;
    Tensor epsilon = Subdivide.of(.7, .1, batches);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = sarsaType.supply(maxbias, qsa, learningRate);
    ActionValueStatistics avs = new ActionValueStatistics(maxbias);
    for (int index = 0; index < batches; ++index) {
      if (batches - 10 < index)
        Infoline.print(maxbias, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(maxbias, qsa, epsilon.Get(index));
      // sarsa.supplyPolicy(() -> policy);
      sarsa.setExplore(epsilon.Get(index));
      ExploringStarts.batch(maxbias, policy, nstep, sarsa, avs);
    }
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    DiscreteUtils.print(vs, Round._3);
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.ORIGINAL, 3);
    handle(SarsaType.EXPECTED, 3);
    handle(SarsaType.QLEARNING, 3);
  }
}
