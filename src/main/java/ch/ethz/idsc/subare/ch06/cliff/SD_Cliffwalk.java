// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
class SD_Cliffwalk {
  static void handle(StepDigestType type) throws Exception {
    System.out.println(type);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_" + type + ".gif"), 100);
    for (int c = 0; c < 200; ++c) {
      System.out.println(c);
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, RealScalar.of(.1));
      StepDigest stepDigest = type.supply(cliffwalk, qsa, RealScalar.of(.25), policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStartsBatch.apply(cliffwalk, stepDigest, policyInterface);
      if (c % 2 == 0)
        gsw.append(ImageFormat.of(CliffwalkHelper.render(cliffwalk, qsa)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(cliffwalk, qsa);
    EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(StepDigestType.original);
    // handle(StepDigestType.expected);
    handle(StepDigestType.qlearning);
  }
}
