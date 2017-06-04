// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 4.1, p.82 */
class SD_Gridworld {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  static void handle(StepDigestType type) throws Exception {
    System.out.println(type);
    final Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int EPISODES = 20;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_" + type + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gridworld, qsa, epsilon.Get(index));
      StepDigest stepDigest = type.supply(gridworld, qsa, RealScalar.of(.2), policyInterface);
      if (type.equals(StepDigestType.qlearning)) {
        ExploringStartsBatch.apply(gridworld, stepDigest, policyInterface);
      } else {
        ExploringStartsBatch.apply(gridworld, stepDigest, policyInterface);
        ExploringStartsBatch.apply(gridworld, stepDigest, policyInterface);
      }
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa, ref)));
    }
    gsw.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa);
    Put.of(UserHome.file("gridworld_" + type), vs.values());
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(gridworld, qsa);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyInterface);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(StepDigestType.original);
    handle(StepDigestType.expected);
    // handle(StepDigestType.qlearning);
  }
}
