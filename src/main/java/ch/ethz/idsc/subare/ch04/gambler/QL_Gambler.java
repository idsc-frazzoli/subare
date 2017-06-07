// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.N;
import ch.ethz.idsc.tensor.sca.Power;
import ch.ethz.idsc.tensor.sca.Round;

/** Sarsa applied to gambler */
class QL_Gambler {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  static void handle() throws Exception {
    System.out.println();
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.4, .1, EPISODES);
    PolicyInterface policyInterface = new EquiprobablePolicy(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_ql.gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      Scalar eps = epsilon.Get(index);
      eps = Power.of(eps, 2);
      System.out.println(index + " " + eps.map(ROUND) + " " + error.map(ROUND));
      Sarsa stepDigest = new QLearning(gambler, qsa, Power.of(N.of(RationalScalar.of(1, 16*(index+1))),0.6));
      for (int count = 0; count < 1; ++count) {
        ExploringStartsBatch.apply(gambler, stepDigest, policyInterface);
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
