// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.sca.Round;

/** Q-Learning applied to gambler with adaptive learning rate */
enum QL_Gambler {
  ;
  static void handle() throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int batches = 100;
    // Tensor epsilon = Subdivide.of(.2, .001, EPISODES);
    Policy policy = EquiprobablePolicy.create(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gambler_qsa_ql.gif"), 100);
    ExplorationRateDeque lr_scheduler = new ExplorationRateDeque(0.1);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.51);
    Sarsa stepDigest = SarsaType.QLEARNING.supply(gambler, learningRate, qsa);
    for (int index = 0; index < batches; ++index) {
      Infoline.print(gambler, index, ref, qsa);
      Scalar error = DiscreteValueFunctions.distance(qsa, ref);
      lr_scheduler.notifyError(error);
      Scalar eps = lr_scheduler.getEpsilon();
      for (int count = 0; count < 1; ++count) {
        ExploringStarts.batch(gambler, policy, 1, stepDigest);
        policy = EGreedyPolicy.bestEquiprobable(gambler, qsa, eps);
      }
      gsw.append(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref));
    }
    gsw.close();
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
