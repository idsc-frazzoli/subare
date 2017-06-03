// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class Sarsa_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire4";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.5, .01, EPISODES);
    PolicyInterface policy = new EquiprobablePolicy(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_qsa_esarsa.gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index + " " + epsilon.Get(index));
      Sarsa sarsa = new ExpectedSarsa(wireloop, policy, qsa, epsilon.Get(index));
      // sarsa.simulate(500); // FIXME
      policy = EGreedyPolicy.bestEquiprobable(wireloop, qsa, epsilon.Get(index));
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
    }
    gsw.close();
  }
}
