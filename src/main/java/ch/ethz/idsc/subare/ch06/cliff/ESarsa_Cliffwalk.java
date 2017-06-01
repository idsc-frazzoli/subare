// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** Expected Sarsa applied to cliff walk */
class ESarsa_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    PolicyInterface policy = new EquiprobablePolicy(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_esarsa.gif"), 100);
    for (int c = 0; c < 200; ++c) {
      System.out.println(c);
      ExpectedSarsa expectedSarsa = new ExpectedSarsa( //
          cliffwalk, policy, //
          cliffwalk, //
          qsa, RealScalar.of(.25));
      expectedSarsa.simulate(3);
      policy = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, RealScalar.of(.1));
      if (c % 2 == 0)
        gsw.append(ImageFormat.of(CliffwalkHelper.render(cliffwalk, qsa)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    EpisodeInterface mce = cliffwalk.kickoff(policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
