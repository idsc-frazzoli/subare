// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
class SD_Cliffwalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  static void handle(SarsaType type, int total) throws Exception {
    System.out.println(type);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_" + type + ".gif"), 100);
    for (int index = 0; index < total; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      System.out.println(index + " " + error.map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(cliffwalk, qsa, RealScalar.of(.1));
      Sarsa sarsa = type.supply(cliffwalk, qsa, RealScalar.of(.25), policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStarts.batch(cliffwalk, policyInterface, sarsa);
      if (index % 2 == 0)
        gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(cliffwalk, qsa, ref)));
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
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 200);
    handle(SarsaType.qlearning, 10);
  }
}
