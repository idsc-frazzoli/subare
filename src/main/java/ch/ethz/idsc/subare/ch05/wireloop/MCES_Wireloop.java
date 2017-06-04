// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

class MCES_Wireloop {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_mces.gif"), 100);
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(mces.qsa(), ref);
      System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      for (int count = 0; count < 4; ++count) {
        PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(wireloop, mces.qsa(), epsilon.Get(index));
        ExploringStartsBatch.apply(wireloop, mces, policyInterface);
      }
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, mces.qsa())));
    }
    gsw.close();
  }
}
