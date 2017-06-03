// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class QL_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire4";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    PolicyInterface policyInterface = new EquiprobablePolicy(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    QLearning qLearning = new QLearning(wireloop, qsa, RealScalar.of(.2));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_ql.gif"), 250);
    for (int count = 0; count < 60; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
      // qLearning.simulate(200); // FIXME
    }
    gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
    gsw.close();
  }
}
