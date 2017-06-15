// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

class Sarsa_Maxbias {
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("--- " + sarsaType);
    Maxbias maxbias = new Maxbias(3);
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.7, .1, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = sarsaType.supply(maxbias, qsa, learningRate);
    ActionValueStatistics avs = new ActionValueStatistics(maxbias);
    for (int index = 0; index < EPISODES; ++index) {
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(maxbias, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policyInterface);
      ExploringStarts.batch(maxbias, policyInterface, n, sarsa, avs);
    }
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    vs.print(Round._3);
    Scalar error = TensorValuesUtils.distance(qsa, ref);
    System.out.println("error = " + error.map(Round._3));
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.original, 3);
    handle(SarsaType.expected, 3);
    handle(SarsaType.qlearning, 3);
  }
}
