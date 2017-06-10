// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** action value iteration for cliff walk */
class AVI_Windygrid {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    Export.of(UserHome.file("Pictures/windygrid_qsa_avi.png"), //
        WindygridHelper.render(windygrid, TensorValuesUtils.rescaled(ref)));
    ActionValueIteration avi = new ActionValueIteration(windygrid, windygrid);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/windygrid_qsa_avi.gif"), 250);
    for (int index = 0; index < 20; ++index) {
      Scalar error = TensorValuesUtils.distance(avi.qsa(), ref);
      System.out.println(index + " " + error.map(ROUND));
      gsw.append(ImageFormat.of(WindygridHelper.joinAll(windygrid, avi.qsa(), ref)));
      avi.step();
    }
    gsw.append(ImageFormat.of(WindygridHelper.joinAll(windygrid, avi.qsa(), ref)));
    gsw.close();
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    vs.print();
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(windygrid, ref);
    Policies.print(policyInterface, windygrid.states());
  }
}
