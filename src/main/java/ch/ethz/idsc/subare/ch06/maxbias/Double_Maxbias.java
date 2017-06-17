// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

/** Double Sarsa for maximization bias */
class Double_Maxbias {
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("double " + sarsaType);
    Maxbias maxbias = new Maxbias(1);
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(maxbias);
    DiscreteQsa qsa2 = DiscreteQsa.build(maxbias);
    DoubleSarsa doubleSarsa = new DoubleSarsa(sarsaType, maxbias, //
        qsa1, qsa2, //
        DefaultLearningRate.of(5, .51), //
        DefaultLearningRate.of(5, .51));
    for (int index = 0; index < EPISODES; ++index) {
      Scalar explore = epsilon.Get(index);
      Scalar error = TensorValuesUtils.distance(qsa1, ref);
      System.out.println(index + " " + explore.map(Round._2) + " " + error.map(Round._1));
      Policy policy = EGreedyPolicy.bestEquiprobable( //
          maxbias, TensorValuesUtils.average(qsa1, qsa2), explore);
      doubleSarsa.setPolicy(policy);
      ExploringStarts.batch(maxbias, policy, n, doubleSarsa);
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa1);
    Put.of(UserHome.file("gridworld_" + sarsaType), vs.values());
    Policy policy = GreedyPolicy.bestEquiprobable(maxbias, qsa1);
    EpisodeInterface ei = EpisodeKickoff.single(maxbias, policy);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.qlearning, 1);
    // handle(SarsaType.expected, 3);
    // handle(SarsaType.qlearning, 2);
  }
}
