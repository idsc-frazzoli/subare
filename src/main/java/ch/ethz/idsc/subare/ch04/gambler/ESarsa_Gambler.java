// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
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

/** Expected Sarsa applied to gambler */
class ESarsa_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.9, .01, EPISODES);
    PolicyInterface policy = new EquiprobablePolicy(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_esarsa.gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      ExpectedSarsa expectedSarsa = new ExpectedSarsa( //
          gambler, policy, //
          gambler, //
          qsa, RealScalar.ONE, RealScalar.of(.2));
      expectedSarsa.simulate(300);
      policy = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      gsw.append(ImageFormat.of(GamblerHelper.render(gambler, qsa)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gambler, qsa);
    Put.of(UserHome.file("esarsa_gambler"), vs.values());
    EpisodeInterface mce = gambler.kickoff(policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
