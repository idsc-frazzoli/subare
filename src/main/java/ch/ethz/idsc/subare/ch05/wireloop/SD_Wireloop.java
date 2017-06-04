// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class SD_Wireloop {
  static void handle(StepDigestType type) throws Exception {
    System.out.println(type);
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.2, .01, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_qsa_" + type + ".gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index + " " + epsilon.Get(index));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(wireloop, qsa, epsilon.Get(index));
      StepDigest stepDigest = type.supply(wireloop, qsa, RealScalar.of(.1), policyInterface);
      ExploringStartsBatch.apply(wireloop, stepDigest, policyInterface);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    handle(StepDigestType.expected);
    // handle(StepDigestType.qlearning);
  }
}
