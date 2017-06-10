// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** action value iteration for gambler's dilemma */
class AVI_Gambler2 {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_avi.gif"), 500);
    for (int index = 0; index < 13; ++index) {
      Scalar error = TensorValuesUtils.distance(avi.qsa(), ref);
      System.out.println(index + " " + error.map(ROUND));
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, avi.qsa(), ref)));
      avi.step();
    }
    gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, avi.qsa(), ref)));
    gsw.close();
  }
}
