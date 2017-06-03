// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
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

/** Sarsa applied to gridworld */
class Sarsa_Gridworld {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  static void handle(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    final Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int EPISODES = 50;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES);
    PolicyInterface policyInterface = new EquiprobablePolicy(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_" + sarsaType + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = qsa.distance(ref);
      System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      Sarsa sarsa = sarsaType.supply( //
          gridworld, qsa, RealScalar.of(.2), //
          policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStartsBatch.apply(gridworld, sarsa, policyInterface);
      policyInterface = EGreedyPolicy.bestEquiprobable(gridworld, qsa, epsilon.Get(index));
      gsw.append(ImageFormat.of(GridworldHelper.render(gridworld, qsa)));
    }
    gsw.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa);
    Put.of(UserHome.file("gambler_" + sarsaType), vs.values());
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyInterface);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.original);
    handle(SarsaType.expected);
  }
}
