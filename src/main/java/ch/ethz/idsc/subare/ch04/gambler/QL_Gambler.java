// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** Sarsa applied to gambler */
class QL_Gambler {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  static void handle() throws Exception {
    System.out.println();
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.2, .001, EPISODES);
    PolicyInterface policyInterface = new EquiprobablePolicy(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_ql.gif"), 100);
    LearningRateScheduler lr_scheduler = new LearningRateScheduler(0.6, 16, 0.1, 0.1);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      lr_scheduler.notifyError(error);
      Scalar alpha = lr_scheduler.getRate(); // lr_scheduler.getRate(index);
      Scalar eps = lr_scheduler.getEpsilon(); // epsilon.Get(index);
      // eps = Power.of(eps, 2);
      System.out.println(index + " " + eps.map(ROUND) + " " + error.map(ROUND));
      Sarsa stepDigest = new QLearning(gambler, qsa, alpha);
      for (int count = 0; count < 1; ++count) {
        ExploringStarts.batch(gambler, policyInterface, 1, stepDigest);
        policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, eps);
      }
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, qsa, ref)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    EpisodeInterface mce = EpisodeKickoff.single(gambler, policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(StepDigestType.original);
    // handle(StepDigestType.expected);
    handle();
    System.exit(0);
  }
}
