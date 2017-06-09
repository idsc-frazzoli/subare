// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** monte carlo is bad in this example, since the steep negative reward biases most episodes */
class MCES_Cliffwalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(mces.qsa(), ref);
      System.out.println(index + " " + error.map(ROUND));
      for (int count = 0; count < 10; ++count) {
        PolicyInterface policyInterface = //
            EGreedyPolicy.bestEquiprobable(cliffwalk, mces.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(cliffwalk, policyInterface, mces);
      }
      gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(cliffwalk, mces.qsa(), ref)));
    }
    gsw.close();
  }
}
