// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.sca.Round;

/** Q-Learning applied to gambler with adaptive learning rate */
/* package */ enum QL_Gambler {
  ;
  static void handle() throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int batches = 100;
    Policy policy = EquiprobablePolicy.create(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policyEGreedy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, qsa, sac);
    policyEGreedy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    System.out.println(qsa.size());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_ql.gif"), 100, TimeUnit.MILLISECONDS)) {
      LearningRate learningRate = DefaultLearningRate.of(2, 0.51);
      Sarsa stepDigest = SarsaType.QLEARNING.sarsa(gambler, learningRate, qsa, sac, policyEGreedy);
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gambler, index, ref, qsa);
        for (int count = 0; count < 1; ++count) {
          ExploringStarts.batch(gambler, policy, 1, stepDigest);
        }
        animationWriter.append(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref));
      }
    }
    DiscreteUtils.print(qsa, Round._2);
    System.out.println("---");
    EpisodeInterface mce = EpisodeKickoff.single(gambler, policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle();
  }
}
