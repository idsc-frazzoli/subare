// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.red.Min;

/** Sarsa applied to gambler */
class Bulk_Gambler {
  private final Gambler gambler;
  private final DiscreteQsa qsa;
  private final Tensor epsilon;
  private final Sarsa sarsa;

  Bulk_Gambler(Gambler gambler, SarsaType sarsaType, //
      int EPISODES, Scalar factor, Scalar exponent) throws Exception {
    this.gambler = gambler;
    epsilon = Subdivide.of(.2, .01, EPISODES);
    qsa = DiscreteQsa.build(gambler); // q-function for training, initialized to 0
    sarsa = sarsaType.supply(gambler, qsa, DefaultLearningRate.of(factor, exponent));
  }

  void step(int index) {
    PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
    sarsa.setPolicyInterface(policyInterface);
    sarsa.getUcbPolicy().setTime(RealScalar.of(index + 1)); // TODO
    PolicyInterface ucbPolicy = sarsa.getUcbPolicy();
    ExploringStarts.batch(gambler, policyInterface, 1, sarsa);
  }

  public static void main(String[] args) throws Exception {
    SarsaType sarsaType = SarsaType.original;
    final int EPISODES = 80;
    final int ERRORCAP = 15;
    Gambler gambler = new Gambler(20, RationalScalar.of(4, 10));
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    Map<Point, Bulk_Gambler> map = new HashMap<>();
    final int RESX = 32;
    final int RESY = 25;
    int x = 0;
    for (Tensor factor : Subdivide.of(.5, 16, RESX - 1)) {
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 2, RESY - 1)) {
        Bulk_Gambler bg = new Bulk_Gambler(gambler, sarsaType, EPISODES, factor.Get(), exponent.Get());
        map.put(new Point(x, y), bg);
        ++y;
      }
      ++x;
    }
    Interpolation interpolation = Colorscheme.classic();
    Tensor image = Array.zeros(RESX, RESY, 4);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gambler_bulk_" + sarsaType + ".gif"), 200);
    final Tensor BASE = Tensors.vector(255);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      for (Entry<Point, Bulk_Gambler> entry : map.entrySet()) {
        Point point = entry.getKey();
        Bulk_Gambler bg = entry.getValue();
        bg.step(index);
        Scalar error = TensorValuesUtils.distance(bg.qsa, ref).multiply(RationalScalar.of(1, ERRORCAP));
        error = Min.of(error, RealScalar.ONE);
        image.set(interpolation.get(BASE.multiply(error)), point.x, point.y);
      }
      gsw.append(ImageFormat.of(ImageResize.of(image, 7)));
    }
    gsw.close();
  }
}
