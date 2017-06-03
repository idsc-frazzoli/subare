// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

/** Sarsa applied to gridworld */
class Sarsa_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gambler = new Gridworld();
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.9, .01, EPISODES);
    PolicyInterface policyInterface = new EquiprobablePolicy(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_sarsa.gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      Sarsa sarsa = new OriginalSarsa( //
          gambler, qsa, RealScalar.of(.2), //
          policyInterface);
      // sarsa.simulate(10);// FIXME
      policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      gsw.append(ImageFormat.of(GridworldHelper.render(gambler, qsa)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gambler, qsa);
    Put.of(UserHome.file("esarsa_gambler"), vs.values());
    EpisodeInterface ei = EpisodeKickoff.create(gambler, policyInterface);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
